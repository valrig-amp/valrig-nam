//! `Conv1D` (dilated, grouped) and `Conv1x1` (pointwise, grouped).
//!
//! Both mirror `NAM/conv1d.cpp` and the `Conv1x1` section of `NAM/dsp.cpp`,
//! including the exact order in which floats are drawn from the `.nam` weight
//! array and the exact order in which products are accumulated.
//!
//! # Accumulation order
//!
//! The reference computes a dilated convolution as one Eigen matrix product
//! per kernel tap, accumulated into a zeroed output:
//!
//! ```text
//! out.leftCols(n).setZero();
//! for (k = 0; k < K; k++)
//!     out.leftCols(n).noalias() += weight[k] * ring.Read(n, dilation * (K - 1 - k));
//! out.leftCols(n).colwise() += bias;
//! ```
//!
//! With `+=`, Eigen evaluates each tap's contribution into a temporary and
//! then adds it, so the per-element order is: sum over input channels
//! *within* a tap, then sum across taps, then the bias. That association is
//! reproduced here exactly, because collapsing it into a single running
//! accumulator over `(k, i)` would give different (equally valid, but
//! different) results.
//!
//! # Lanes
//!
//! How a convolution runs is decided once, when it is built, by its shape
//! ([`Taps`]). One whose output is at least a register wide has its lanes
//! across the output channels, and reads whole frames from a ring. One
//! whose output is narrower, the head of an A2 capture or any layer of a
//! three-channel model, has its lanes along time instead, and reads four
//! frames of a channel at once from a planar history; see
//! [`crate::kernels`] for the two kernels and [`crate::history`] for the two
//! histories. Either way each number is the sum above, in that order.
//!
//! # Weights
//!
//! A convolution reads its weights when it is built, in the order the
//! reference's constructor and `set_weights` visit them, and before it
//! allocates storage for them it checks that the file has that many left.
//! So a file whose shapes describe more weights than it carries fails with
//! [`Error::WeightCount`] at the first convolution that cannot be filled,
//! rather than sizing an allocation for a shape that was never real.

use crate::buffer::Buf;
use crate::error::{Error, Result};
use crate::history::{Arena, History, Planes};
use crate::kernels::{
    FRAME_LANE_INPUTS, FRAME_LANE_OUTPUTS, FramesFn, MatMulFn, Planar, Product, depthwise_accum,
    matmul_auto, select_frames, select_matmul,
};
use crate::weights::WeightReader;

/// A borrowed column-major view whose columns may be strided.
///
/// The reference regularly passes `_z.topRows(bottleneck)` into a `Conv1x1`:
/// a block whose column stride is the *source* row count, not its own. This
/// carries that distinction rather than forcing a copy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct View<'a> {
    data: &'a [f32],
    rows: usize,
    stride: usize,
}

impl<'a> View<'a> {
    /// A view over a whole buffer.
    #[inline]
    pub(crate) fn full(b: &'a Buf) -> Self {
        Self {
            data: b.data(),
            rows: b.rows(),
            stride: b.rows(),
        }
    }

    /// A view over the top `rows` channels of a buffer.
    #[inline]
    pub(crate) fn top_rows(b: &'a Buf, rows: usize) -> Self {
        debug_assert!(rows <= b.rows());
        Self {
            data: b.data(),
            rows,
            stride: b.rows(),
        }
    }

    /// Channels in the view.
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// One column.
    #[inline]
    pub(crate) fn col(&self, c: usize) -> &'a [f32] {
        let s = c * self.stride;
        &self.data[s..s + self.rows]
    }
}

/// How the weights of a grouped convolution are stored: one weight per
/// channel per tap when every group is one channel wide, a full matrix per
/// tap otherwise.
///
/// The reference stores a grouped convolution whose groups equal both
/// channel counts as one weight per channel per tap.
fn is_depthwise(in_ch: usize, out_ch: usize, groups: usize) -> bool {
    groups == in_ch && in_ch == out_ch
}

fn check_groups(in_ch: usize, out_ch: usize, groups: usize) -> Result<()> {
    if groups == 0 {
        return Err(Error::Config("groups must be positive".into()));
    }
    if !in_ch.is_multiple_of(groups) {
        return Err(Error::Config(format!(
            "in_channels ({in_ch}) must be divisible by numGroups ({groups})"
        )));
    }
    if !out_ch.is_multiple_of(groups) {
        return Err(Error::Config(format!(
            "out_channels ({out_ch}) must be divisible by numGroups ({groups})"
        )));
    }
    Ok(())
}

/// Floats a convolution of this shape consumes from the weight array, or an
/// error when the product does not fit in `usize`.
fn weight_count(
    in_ch: usize,
    out_ch: usize,
    groups: usize,
    kernel_size: usize,
    bias: bool,
) -> Result<usize> {
    let per_tap = if is_depthwise(in_ch, out_ch, groups) {
        Some(in_ch)
    } else {
        (out_ch / groups)
            .checked_mul(in_ch / groups)
            .and_then(|n| n.checked_mul(groups))
    };
    per_tap
        .and_then(|n| n.checked_mul(kernel_size))
        .and_then(|n| n.checked_add(if bias { out_ch } else { 0 }))
        .ok_or_else(|| {
            Error::Config(format!(
                "a {out_ch}x{in_ch} convolution with kernel size {kernel_size} has more weights than can be counted"
            ))
        })
}

/// The `(row, col)` positions of a grouped weight matrix in the order the
/// reference fills them, *"Crazy ordering because that's how it gets
/// flattened."* For each group, for each output channel in the group, for
/// each input channel in the group.
fn grouped_positions(
    out_ch: usize,
    in_ch: usize,
    groups: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let opg = out_ch / groups;
    let ipg = in_ch / groups;
    (0..groups).flat_map(move |g| {
        (0..opg).flat_map(move |i| (0..ipg).map(move |j| (g * opg + i, g * ipg + j)))
    })
}

/// `out.leftCols(n).colwise() += bias`, on a tightly packed `(out_ch, n)`
/// block.
#[inline]
fn add_bias(out: &mut [f32], bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    for frame in out.chunks_exact_mut(bias.len()) {
        for (v, b) in frame.iter_mut().zip(bias) {
            *v += *b;
        }
    }
}

/// How a [`Conv1D`]'s taps are stored and run, each with the history it
/// reads: which way the lanes go.
#[derive(Debug, Clone)]
enum Taps {
    /// One weight per channel per tap, `[tap][channel]`: the reference's
    /// storage for a convolution whose groups equal both channel counts.
    Depthwise {
        w: Vec<Vec<f32>>,
        history: Option<History>,
    },
    /// A matrix per tap, column-major `(out_ch, in_ch)`, with the lanes
    /// across the output channels.
    Channels {
        w: Vec<Vec<f32>>,
        /// The batched kernel for this width, resolved once at load time so
        /// `process()` does no shape dispatch. `None` falls back to
        /// `matmul_dyn`.
        kernel: Option<MatMulFn>,
        history: Option<History>,
    },
    /// For an output narrower than a register: the lanes along time.
    Frames {
        /// Per tap and output, a weight from every input, in registers of
        /// four, the last padded.
        w: Vec<[f32; 4]>,
        /// The kernel for this many outputs.
        kernel: FramesFn,
        history: Option<Planes>,
        /// The block's sums before the bias: a row per output, as long as
        /// the largest block rounded up to whole registers.
        sums: Vec<f32>,
    },
}

/// A 1D dilated, optionally grouped convolution with its own input history.
#[derive(Debug, Clone)]
pub(crate) struct Conv1D {
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    taps: Taps,
    /// One per output channel; empty when the convolution has no bias.
    bias: Vec<f32>,
    cached_prewarm_state: Vec<f32>,
    has_cached_prewarm_state: bool,
    /// `(out_ch, max_buffer)`: the convolution's result for the current block.
    pub(crate) output: Buf,
}

impl Conv1D {
    /// Build with the given shape and read the weights from `r`: the taps
    /// in the reference's grouped order, then the bias if `bias` is set.
    ///
    /// Fails with [`Error::Config`] when the groups do not divide the
    /// channel counts and [`Error::WeightCount`] when `r` cannot supply the
    /// shape.
    pub(crate) fn new(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        bias: bool,
        dilation: usize,
        groups: usize,
        r: &mut WeightReader<'_>,
    ) -> Result<Self> {
        check_groups(in_ch, out_ch, groups)?;
        r.check(weight_count(in_ch, out_ch, groups, kernel_size, bias)?)?;
        // In both layouts the tap index varies fastest.
        let taps = if is_depthwise(in_ch, out_ch, groups) {
            let mut w = vec![vec![0.0; in_ch]; kernel_size];
            for c in 0..in_ch {
                for tap in w.iter_mut() {
                    tap[c] = r.next()?;
                }
            }
            Taps::Depthwise { w, history: None }
        } else {
            let mut w = vec![vec![0.0; out_ch * in_ch]; kernel_size];
            for (row, col) in grouped_positions(out_ch, in_ch, groups) {
                for tap in w.iter_mut() {
                    tap[row + col * out_ch] = r.next()?;
                }
            }
            Taps::of_matrices(w, out_ch, in_ch)
        };
        let mut b = vec![0.0; if bias { out_ch } else { 0 }];
        r.fill(&mut b)?;
        Ok(Self {
            in_ch,
            out_ch,
            kernel_size,
            dilation,
            groups,
            taps,
            bias: b,
            cached_prewarm_state: vec![0.0; in_ch],
            has_cached_prewarm_state: false,
            output: Buf::new(),
        })
    }

    /// Input channels.
    pub(crate) fn in_channels(&self) -> usize {
        self.in_ch
    }

    /// Output channels.
    pub(crate) fn out_channels(&self) -> usize {
        self.out_ch
    }

    /// Kernel size.
    pub(crate) fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Dilation factor.
    pub(crate) fn dilation(&self) -> usize {
        self.dilation
    }

    /// The weights per tap, column-major `(out_ch, in_ch)`, whichever way
    /// the convolution keeps them; empty when depthwise.
    pub(crate) fn tap_matrices(&self) -> Vec<Vec<f32>> {
        match &self.taps {
            Taps::Depthwise { .. } => Vec::new(),
            Taps::Channels { w, .. } => w.clone(),
            Taps::Frames { w, .. } => {
                let per = self.in_ch.div_ceil(4);
                w.chunks_exact(self.out_ch * per)
                    .map(|tap| {
                        let mut m = vec![0.0; self.out_ch * self.in_ch];
                        for (o, row) in tap.chunks_exact(per).enumerate() {
                            for (i, w) in row.as_flattened().iter().take(self.in_ch).enumerate() {
                                m[o + i * self.out_ch] = *w;
                            }
                        }
                        m
                    })
                    .collect()
            }
        }
    }

    /// The bias, one per output channel; empty when the layer has none.
    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Whether the convolution is plain: one group and a full matrix, which
    /// a 1-channel convolution is not, since the reference stores that as
    /// depthwise.
    pub(crate) fn is_plain(&self) -> bool {
        self.groups == 1 && !matches!(self.taps, Taps::Depthwise { .. })
    }

    /// How many frames of history this convolution reaches back:
    /// `(kernel_size - 1) * dilation`, or an error when that does not fit.
    pub(crate) fn receptive_field(&self) -> Result<usize> {
        self.kernel_size
            .saturating_sub(1)
            .checked_mul(self.dilation)
            .ok_or_else(|| {
                Error::Config(format!(
                    "a kernel of {} at dilation {} reaches further back than can be counted",
                    self.kernel_size, self.dilation
                ))
            })
    }

    /// Floats of input history this convolution keeps:
    /// `in_channels × receptive_field`, or an error when that does not fit.
    pub(crate) fn history_floats(&self) -> Result<usize> {
        self.in_ch
            .checked_mul(self.receptive_field()?)
            .ok_or_else(|| {
                Error::Config(format!(
                    "{} channels of {}-frame history is more than can be counted",
                    self.in_ch, self.kernel_size
                ))
            })
    }

    /// Reserve history and output storage for blocks of up to `max_buffer`
    /// frames. Allocating; never call it from `process`.
    ///
    /// The receptive field has been checked by [`Conv1D::receptive_field`]
    /// before this is reached.
    pub(crate) fn set_max_buffer_size(&mut self, arena: &mut Arena, max_buffer: usize) {
        let lookback = self.kernel_size.saturating_sub(1) * self.dilation;
        match &mut self.taps {
            Taps::Depthwise { history, .. } | Taps::Channels { history, .. } => {
                *history = Some(History::reserve(arena, self.in_ch, lookback, max_buffer));
            }
            Taps::Frames { history, sums, .. } => {
                let padded = max_buffer.div_ceil(4) * 4;
                *history = Some(Planes::new(self.in_ch, lookback, padded));
                *sums = vec![0.0; self.out_ch * padded];
            }
        }
        self.output.resize(self.out_ch, max_buffer);
        // The cached prewarm column is one frame wide and independent of the
        // block size, so resizing does not invalidate it. The reference keeps
        // it too, which is what lets a second Reset() skip the prewarm run.
    }

    /// True when a steady-state history has been cached by
    /// [`Conv1D::cache_prewarm_state`].
    pub(crate) fn has_cached_prewarm_state(&self) -> bool {
        self.has_cached_prewarm_state
    }

    /// Snapshot the last written input column as the steady prewarm state.
    pub(crate) fn cache_prewarm_state(&mut self, arena: &Arena) {
        let cached = &mut self.cached_prewarm_state;
        match &self.taps {
            Taps::Depthwise {
                history: Some(h), ..
            }
            | Taps::Channels {
                history: Some(h), ..
            } => h.cache_last_written(arena, cached),
            Taps::Frames {
                history: Some(h), ..
            } => h.last_written(cached),
            // Never sized, so there is nothing to snapshot.
            _ => return,
        }
        self.has_cached_prewarm_state = true;
    }

    /// Refill the history from the cached steady state, skipping a prewarm run.
    pub(crate) fn prewarm_from_cache(&mut self, arena: &mut Arena) {
        debug_assert!(self.has_cached_prewarm_state);
        let cached = &self.cached_prewarm_state;
        match &mut self.taps {
            Taps::Depthwise {
                history: Some(h), ..
            }
            | Taps::Channels {
                history: Some(h), ..
            } => h.fill_with_sample(arena, cached),
            Taps::Frames {
                history: Some(h), ..
            } => h.fill_with_sample(cached),
            _ => {}
        }
    }

    /// Process `n` frames of `input` into [`Conv1D::output`].
    ///
    /// Panics if [`Conv1D::set_max_buffer_size`] has not run or `n` exceeds
    /// the size it was given.
    pub(crate) fn process(&mut self, arena: &mut Arena, input: &Buf, n: usize) {
        debug_assert_eq!(input.rows(), self.in_ch);
        const UNSIZED: &str = "Conv1D::process before set_max_buffer_size";
        let (in_ch, out_ch) = (self.in_ch, self.out_ch);
        let reach = (self.kernel_size, self.dilation);
        let out = self.output.left_mut(n);
        match &mut self.taps {
            Taps::Depthwise { w, history } => {
                let hist = history.as_mut().expect(UNSIZED);
                each_run(hist, arena, input, out, n, reach, |k, src, o, count| {
                    depthwise_accum(&w[k], src, in_ch, o, in_ch, count);
                });
            }
            Taps::Channels { w, kernel, history } => {
                let hist = history.as_mut().expect(UNSIZED);
                each_run(hist, arena, input, out, n, reach, |k, src, o, count| {
                    // The run is contiguous and column-major, so its stride
                    // is just `in_ch`. Batching over it is what keeps each
                    // weight column resident across the tile's frames
                    // instead of being re-streamed per frame; see
                    // `kernels::matmul`.
                    let p = Product {
                        w: &w[k],
                        x: src,
                        x_stride: in_ch,
                        out_ch,
                        in_ch,
                        n: count,
                    };
                    matmul_auto(*kernel, p, o);
                });
            }
            Taps::Frames {
                w,
                kernel,
                history,
                sums,
            } => {
                let hist = history.as_mut().expect(UNSIZED);
                if n == 0 {
                    return;
                }
                // Whole registers of frames. The ones past `n` come out of
                // whatever the rows hold there and are not kept.
                let groups = n.div_ceil(4);
                hist.begin(4 * groups);
                let pos = hist.pos();
                // The block, a row per channel.
                for (c, row) in hist.rows_mut().enumerate() {
                    let frames = input.data().chunks_exact(in_ch);
                    for (x, frame) in row[pos..pos + n].iter_mut().zip(frames) {
                        *x = frame[c];
                    }
                }
                let padded = sums.len() / out_ch;
                {
                    let mut rows = hist.data().chunks_exact(hist.stride());
                    let rows: [&[f32]; FRAME_LANE_INPUTS] =
                        std::array::from_fn(|_| rows.next().unwrap_or(&[]));
                    let taps = Planar {
                        rows: &rows[..in_ch],
                        w,
                        dilation: reach.1,
                        pos,
                    };
                    let mut rows = sums.chunks_exact_mut(padded);
                    let mut rows: [&mut [[f32; 4]]; FRAME_LANE_OUTPUTS] =
                        std::array::from_fn(|_| {
                            rows.next()
                                .map_or(&mut [][..], |row| row.as_chunks_mut::<4>().0)
                        });
                    kernel(&taps, &mut rows[..out_ch], groups);
                }
                // Back to frames. The sums started from zero, as the output
                // does on the other two paths.
                for (c, row) in sums.chunks_exact(padded).enumerate() {
                    for (frame, s) in out.chunks_exact_mut(out_ch).zip(row) {
                        frame[c] = *s;
                    }
                }
                hist.advance(n);
            }
        }
        add_bias(out, &self.bias);
    }
}

impl Taps {
    /// The form for a convolution of full matrices, one per tap,
    /// column-major `(out_ch, in_ch)`: frame lanes when the output is
    /// narrower than a register, channel lanes otherwise.
    fn of_matrices(w: Vec<Vec<f32>>, out_ch: usize, in_ch: usize) -> Self {
        let Some(kernel) = select_frames(out_ch, in_ch) else {
            return Taps::Channels {
                w,
                kernel: select_matmul(out_ch),
                history: None,
            };
        };
        let per = in_ch.div_ceil(4);
        let mut registers = vec![[0.0; 4]; w.len() * out_ch * per];
        for (tap, m) in registers.chunks_exact_mut(out_ch * per).zip(&w) {
            for (o, row) in tap.chunks_exact_mut(per).enumerate() {
                for (i, w) in row.as_flattened_mut().iter_mut().take(in_ch).enumerate() {
                    *w = m[o + i * out_ch];
                }
            }
        }
        Taps::Frames {
            w: registers,
            kernel,
            history: None,
            sums: Vec::new(),
        }
    }
}

/// What a convolution that reads a ring does around its kernel: write the
/// block of `n` frames, zero `out`, and call `tap(k, run, out, frames)` for
/// every run of every tap in order, with the run's frames and the part of
/// the output they fill. `reach` is the kernel size and the dilation.
fn each_run(
    hist: &mut History,
    arena: &mut Arena,
    input: &Buf,
    out: &mut [f32],
    n: usize,
    (kernel_size, dilation): (usize, usize),
    mut tap: impl FnMut(usize, &[f32], &mut [f32], usize),
) {
    hist.write(arena, input.data(), n);
    out.fill(0.0);
    let out_ch = out.len().checked_div(n).unwrap_or(0);
    let k_last = kernel_size.saturating_sub(1);
    for k in 0..kernel_size {
        let lookback = dilation * (k_last - k);
        let mut frame = 0usize;
        for (col, count) in hist.read_runs(n, lookback) {
            if count == 0 {
                continue;
            }
            let src = hist.run(arena, col, count);
            tap(
                k,
                src,
                &mut out[frame * out_ch..(frame + count) * out_ch],
                count,
            );
            frame += count;
        }
    }
    hist.advance(n);
}

/// How a [`Conv1x1`]'s weights are stored.
#[derive(Debug, Clone)]
enum Weights {
    /// One weight per channel: the reference's storage for a convolution
    /// whose groups equal both channel counts.
    Depthwise(Vec<f32>),
    /// Column-major `(out_ch, in_ch)`.
    Matrix {
        w: Vec<f32>,
        /// The batched kernel for this width, resolved once at load time so
        /// `process()` does no shape dispatch. `None` falls back to
        /// `matmul_dyn`.
        kernel: Option<MatMulFn>,
    },
}

/// A pointwise (1x1) convolution: a per-frame linear layer.
#[derive(Debug, Clone)]
pub(crate) struct Conv1x1 {
    in_ch: usize,
    out_ch: usize,
    groups: usize,
    weights: Weights,
    /// One per output channel; empty when the convolution has no bias.
    bias: Vec<f32>,
    /// `(out_ch, max_buffer)`.
    pub(crate) output: Buf,
}

impl Conv1x1 {
    /// Build with the given shape and read the weights from `r`: the matrix
    /// in the reference's grouped order, then the bias if `bias` is set.
    ///
    /// Fails as [`Conv1D::new`] does.
    pub(crate) fn new(
        in_ch: usize,
        out_ch: usize,
        bias: bool,
        groups: usize,
        r: &mut WeightReader<'_>,
    ) -> Result<Self> {
        check_groups(in_ch, out_ch, groups)?;
        r.check(weight_count(in_ch, out_ch, groups, 1, bias)?)?;
        let weights = if is_depthwise(in_ch, out_ch, groups) {
            let mut w = vec![0.0; in_ch];
            r.fill(&mut w)?;
            Weights::Depthwise(w)
        } else {
            let mut w = vec![0.0; out_ch * in_ch];
            for (row, col) in grouped_positions(out_ch, in_ch, groups) {
                w[row + col * out_ch] = r.next()?;
            }
            Weights::Matrix {
                w,
                kernel: select_matmul(out_ch),
            }
        };
        let mut b = vec![0.0; if bias { out_ch } else { 0 }];
        r.fill(&mut b)?;
        Ok(Self {
            in_ch,
            out_ch,
            groups,
            weights,
            bias: b,
            output: Buf::new(),
        })
    }

    /// Input channels.
    pub(crate) fn in_channels(&self) -> usize {
        self.in_ch
    }

    /// Output channels.
    pub(crate) fn out_channels(&self) -> usize {
        self.out_ch
    }

    /// The weights, column-major `(out_ch, in_ch)`; empty when depthwise.
    pub(crate) fn weight(&self) -> &[f32] {
        match &self.weights {
            Weights::Depthwise(_) => &[],
            Weights::Matrix { w, .. } => w,
        }
    }

    /// The bias, one per output channel; empty when the layer has none.
    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Whether the convolution is plain: one group and a full matrix, which
    /// a 1-channel convolution is not, since the reference stores that as
    /// depthwise.
    pub(crate) fn is_plain(&self) -> bool {
        self.groups == 1 && matches!(self.weights, Weights::Matrix { .. })
    }

    /// Reserve output storage for blocks of up to `max_buffer` frames.
    /// Allocating.
    pub(crate) fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.output.resize(self.out_ch, max_buffer);
    }

    /// Process `n` frames of a whole buffer into [`Conv1x1::output`].
    pub(crate) fn process(&mut self, input: &Buf, n: usize) {
        self.process_view(View::full(input), n);
    }

    /// Process `n` frames of a possibly strided view into
    /// [`Conv1x1::output`].
    pub(crate) fn process_view(&mut self, input: View<'_>, n: usize) {
        debug_assert_eq!(input.rows(), self.in_ch);
        let out = self.output.left_mut(n);
        match &self.weights {
            Weights::Depthwise(w) => {
                for f in 0..n {
                    let x = input.col(f);
                    let o = &mut out[f * self.out_ch..(f + 1) * self.out_ch];
                    for c in 0..self.in_ch {
                        o[c] = w[c] * x[c];
                    }
                }
            }
            Weights::Matrix { w, kernel } => {
                // The reference assigns (`=`) rather than accumulating;
                // this zeroes first and accumulates, so what is written is
                // `0.0 + v`.
                out.fill(0.0);
                // `input` may be a `topRows()`-style block, so its column
                // stride is the source's row count rather than `in_ch`.
                let p = Product {
                    w,
                    x: input.data,
                    x_stride: input.stride,
                    out_ch: self.out_ch,
                    in_ch: self.in_ch,
                    n,
                };
                matmul_auto(*kernel, p, out);
            }
        }
        add_bias(out, &self.bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::splitmix_stream;

    fn seq_weights(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect()
    }

    /// A convolution of this shape with weights from `seq_weights`.
    fn conv(in_ch: usize, out_ch: usize, k: usize, dilation: usize) -> Conv1D {
        let w = seq_weights(weight_count(in_ch, out_ch, 1, k, true).unwrap());
        let mut r = WeightReader::new(&w);
        let c = Conv1D::new(in_ch, out_ch, k, true, dilation, 1, &mut r).unwrap();
        r.finish().unwrap();
        c
    }

    /// The same convolution with its lanes across the output channels,
    /// whatever its shape would have chosen.
    fn channel_lanes(c: &Conv1D) -> Conv1D {
        let mut c = c.clone();
        c.taps = Taps::Channels {
            w: c.tap_matrices(),
            kernel: select_matmul(c.out_ch),
            history: None,
        };
        c
    }

    #[test]
    fn conv1d_matches_direct_definition() {
        // out[o, t] = bias[o] + sum_k sum_i w[k][o, i] * x[i, t - (K-1-k)*dilation]
        // A narrow output and a wide one: either way the lanes go.
        for (in_ch, out_ch, k, dil) in [(3usize, 2usize, 3usize, 2usize), (3, 5, 3, 2)] {
            let mut c = conv(in_ch, out_ch, k, dil);
            let w = c.tap_matrices();
            let mut arena = Arena::new();
            c.set_max_buffer_size(&mut arena, 8);

            // Feed a single block; history before it is zero.
            let n = 8;
            let mut input = Buf::zeros(in_ch, n);
            for t in 0..n {
                for i in 0..in_ch {
                    input.set(i, t, (t * in_ch + i) as f32 * 0.25 - 1.0);
                }
            }
            c.process(&mut arena, &input, n);

            for t in 0..n {
                for o in 0..out_ch {
                    let mut acc = 0.0f32;
                    for (tap, m) in w.iter().enumerate() {
                        let lag = (k - 1 - tap) * dil;
                        let mut inner = 0.0f32;
                        for i in 0..in_ch {
                            let x = if t >= lag { input.at(i, t - lag) } else { 0.0 };
                            inner += m[o + i * out_ch] * x;
                        }
                        acc += inner;
                    }
                    acc += c.bias[o];
                    assert_eq!(
                        c.output.at(o, t).to_bits(),
                        acc.to_bits(),
                        "{in_ch} to {out_ch}: mismatch at o={o} t={t}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_narrow_output_has_its_lanes_along_time() {
        // The head of an A2 capture, and a layer of a three-channel model.
        assert!(matches!(conv(8, 1, 16, 1).taps, Taps::Frames { .. }));
        assert!(matches!(conv(3, 3, 6, 7).taps, Taps::Frames { .. }));
        // A register of outputs or more keeps its channel lanes, and so
        // does a narrow output from more inputs than the kernel takes.
        assert!(matches!(conv(8, 4, 3, 1).taps, Taps::Channels { .. }));
        assert!(matches!(conv(16, 16, 3, 1).taps, Taps::Channels { .. }));
        assert!(matches!(conv(17, 1, 3, 1).taps, Taps::Channels { .. }));
    }

    /// One block of `signal` from frame `at` through both convolutions,
    /// each with its arena, and their outputs compared to the bit.
    fn step(
        frames: &mut (Conv1D, Arena),
        channels: &mut (Conv1D, Arena),
        signal: &[f32],
        at: &mut usize,
        n: usize,
    ) {
        let in_ch = frames.0.in_ch;
        let mut input = Buf::zeros(in_ch, n.max(1));
        input
            .left_mut(n)
            .copy_from_slice(&signal[*at * in_ch..(*at + n) * in_ch]);
        frames.0.process(&mut frames.1, &input, n);
        channels.0.process(&mut channels.1, &input, n);
        let (a, b) = (frames.0.output.left(n), channels.0.output.left(n));
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{in_ch} to {}, {} taps at {}: block at frame {}, element {i}",
                frames.0.out_ch,
                frames.0.kernel_size,
                frames.0.dilation,
                *at
            );
        }
        *at += n;
    }

    /// Which way the lanes go changes which numbers are computed side by
    /// side and nothing else: the same weights give the same bits either
    /// way, over ragged blocks, a resize, and a history restored from the
    /// cached steady state.
    #[test]
    fn both_lane_axes_give_the_same_bits() {
        for (in_ch, out_ch, k, dil) in [
            (8usize, 1usize, 16usize, 1usize),
            (8, 1, 1, 1),
            (1, 2, 4, 3),
            (3, 3, 6, 7),
            (5, 2, 2, 40),
            (16, 3, 5, 2),
        ] {
            let frames = conv(in_ch, out_ch, k, dil);
            assert!(
                matches!(frames.taps, Taps::Frames { .. }),
                "{in_ch} to {out_ch}"
            );
            let channels = channel_lanes(&frames);
            assert_eq!(frames.tap_matrices(), channels.tap_matrices());
            let (mut frames, mut channels) = ((frames, Arena::new()), (channels, Arena::new()));

            let mut signal = splitmix_stream(0xC0 + (in_ch * 31 + out_ch) as u64, in_ch * 400);
            // A stretch of silence and one of tiny values, where a sum can
            // underflow or come out a zero of either sign.
            signal[in_ch * 100..in_ch * 140].fill(0.0);
            for v in &mut signal[in_ch * 200..in_ch * 240] {
                *v *= 1.0e-38;
            }

            let mut at = 0;
            frames.0.set_max_buffer_size(&mut frames.1, 33);
            channels.0.set_max_buffer_size(&mut channels.1, 33);
            for n in [33usize, 1, 7, 0, 33, 2, 5, 33, 33, 4, 33, 30] {
                step(&mut frames, &mut channels, &signal, &mut at, n);
            }
            // The last frame becomes the steady state; a resize clears the
            // histories and the cache refills them.
            frames.0.cache_prewarm_state(&frames.1);
            channels.0.cache_prewarm_state(&channels.1);
            assert_eq!(
                frames.0.cached_prewarm_state,
                channels.0.cached_prewarm_state
            );
            frames.1.clear();
            channels.1.clear();
            frames.0.set_max_buffer_size(&mut frames.1, 8);
            channels.0.set_max_buffer_size(&mut channels.1, 8);
            frames.0.prewarm_from_cache(&mut frames.1);
            channels.0.prewarm_from_cache(&mut channels.1);
            for n in [8usize, 3, 8, 8, 1, 8, 6] {
                step(&mut frames, &mut channels, &signal, &mut at, n);
            }
        }
    }

    #[test]
    fn depthwise_is_selected_when_groups_equal_channels() {
        assert_eq!(weight_count(8, 8, 8, 3, false).unwrap(), 3 * 8);
        assert_eq!(weight_count(8, 8, 4, 3, false).unwrap(), 3 * 2 * 2 * 4);
        let w = seq_weights(24);
        let c = Conv1D::new(8, 8, 3, false, 1, 8, &mut WeightReader::new(&w)).unwrap();
        assert!(matches!(&c.taps, Taps::Depthwise { w, .. } if w.len() == 3));
        assert!(c.tap_matrices().is_empty());
        let w = seq_weights(48);
        let c = Conv1D::new(8, 8, 3, false, 1, 4, &mut WeightReader::new(&w)).unwrap();
        assert!(matches!(c.taps, Taps::Channels { .. }));
        assert_eq!(c.tap_matrices().len(), 3);
        // One channel in, one out, one group: depthwise by the reference's
        // rule, so not "plain".
        let w = seq_weights(3);
        let c = Conv1D::new(1, 1, 3, false, 1, 1, &mut WeightReader::new(&w)).unwrap();
        assert!(matches!(c.taps, Taps::Depthwise { .. }) && !c.is_plain());
    }

    #[test]
    fn grouped_weights_land_on_the_block_diagonal() {
        let w: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let c = Conv1x1::new(4, 4, false, 2, &mut WeightReader::new(&w)).unwrap();
        // groups=2 => two 2x2 blocks; off-diagonal blocks stay zero.
        let at = |o: usize, i: usize| c.weight()[o + i * 4];
        assert_eq!(at(0, 0), 1.0);
        assert_eq!(at(0, 1), 2.0);
        assert_eq!(at(1, 0), 3.0);
        assert_eq!(at(1, 1), 4.0);
        assert_eq!(at(2, 2), 5.0);
        assert_eq!(at(3, 3), 8.0);
        assert_eq!(at(0, 2), 0.0);
        assert_eq!(at(2, 0), 0.0);
        assert!(!c.is_plain(), "two groups are not one");
        // Groups equal to both channel counts are a weight per channel.
        let c = Conv1x1::new(4, 4, false, 4, &mut WeightReader::new(&w[..4])).unwrap();
        assert!(matches!(&c.weights, Weights::Depthwise(w) if w.len() == 4));
        assert!(c.weight().is_empty() && !c.is_plain());
    }

    #[test]
    fn a_shape_the_file_cannot_fill_is_refused_before_allocating() {
        // 16 x 16 x 3 taps = 768 weights; the file has 10.
        let w = seq_weights(10);
        match Conv1D::new(16, 16, 3, false, 1, 1, &mut WeightReader::new(&w)) {
            Err(Error::WeightCount { expected, found }) => assert_eq!((expected, found), (768, 10)),
            other => panic!("{other:?}"),
        }
        // A product that does not fit in usize is a config error, not a wrap.
        assert!(matches!(
            weight_count(usize::MAX / 2, 4, 1, 3, false),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn groups_must_divide_the_channel_counts() {
        let w = seq_weights(64);
        assert!(matches!(
            Conv1x1::new(4, 4, false, 0, &mut WeightReader::new(&w)),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            Conv1x1::new(4, 4, false, 3, &mut WeightReader::new(&w)),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            Conv1x1::new(6, 4, false, 3, &mut WeightReader::new(&w)),
            Err(Error::Config(_))
        ));
    }
}
