//! The fused path for a plain layer array: the layers of NAM's standard
//! architectures, run a layer at a time rather than a pass at a time.
//!
//! A *plain* layer is what every A1 and A2 capture is made of: a dilated
//! convolution with a bias, a mixin from a one-channel condition, one
//! activation, a 1x1 back to the layer's width with a bias, and the residual
//! sum. No gating, no FiLM, no groups, no bottleneck, no head 1x1.
//! `wavenet::Layer` runs one as it runs any layer, as a dozen passes over
//! the block: zero the output, a kernel call per tap, the bias, the mixin
//! into a buffer of its own, the sum of the two, the activation, the 1x1,
//! its bias, the residual, the copy into the next layer's history. Each pass
//! is cheap, and together they are a quarter of a standard A1 capture's
//! time, a third of an A2's and half of a small one's, because at eight
//! channels a core runs out of instructions per cycle before it runs out of
//! multiply-adds.
//!
//! Here a layer is three passes. The first takes a few frames at a time
//! through every tap, with the running sum in registers, adds the bias and
//! the mixin there, and writes the pre-activation once. The second is the
//! activation, the same call the generic path makes. The third is the 1x1
//! with its bias and the residual, written straight into the next layer's
//! history.
//!
//! # Layout
//!
//! Channels are lanes, as in `kernels`, so a layer of `C` channels works in
//! `V = ceil(C / 4)` registers per frame. A history frame is `4 * V` floats
//! whatever `C` is: the lanes past `C` carry zero weights, are never read as
//! inputs, and what lands in them is never looked at. That keeps every load
//! and store a whole register.
//!
//! Each layer's input history is a ring, as `history::History` is and for
//! the same reason: nothing is ever copied, so no block costs more than
//! another. A tap's frames for a block are then one run or two, and a block
//! is cut into *segments* in which every tap's frames and the frames being
//! written are one run each; the kernels below take one segment. A cut is
//! not free, and every tap makes one each time the ring comes round, so the
//! ring is at least [`ROOM`] frames longer than the lookback however short
//! the block. The layer before writes this layer's frames in place, so
//! there is no copy into the history either.
//!
//! The pre-activation is the one buffer that is not padded: it is `C` floats
//! a frame, so the activation runs over exactly the values the generic path
//! hands it, in the same positions. A frame of it is still written and read
//! as whole registers; the lanes past `C` are the start of the next frame,
//! written before that frame is and so overwritten by it.
//!
//! # Bit-exactness
//!
//! For every output channel and frame, each number is computed in the order
//! the generic path computes it: a tap's contribution is a fused chain over
//! the inputs from zero, added to the running sum tap by tap from zero, the
//! bias last; the mixin is one product onto zero, added to zero as the
//! generic path's separate buffer does, then to the sum; the 1x1 is the same
//! chain, added to zero, then its bias, then the layer's input. The
//! additions of zero are kept because they are not all no-ops: a sum that
//! underflows to a negative zero leaves one of them positive. Which frames
//! advance together is the only difference, and frames are independent. The
//! tests below hold the two paths to each other bit for bit, and the
//! recorded cases hold the result to the C++ reference.

use crate::activations::Activation;
use crate::buffer::Buf;
use crate::format::MAX_HISTORY_FLOATS;
use crate::lanes::F4;

/// The widest layer this path takes, in registers per frame: sixteen
/// channels, NAM's standard width. A tile of frames has to fit the register
/// file twice over, once for the tap in hand and once for the running sum.
const MAX_LANES: usize = 4;

/// Frames a ring is longer than its lookback, at the least.
///
/// A ring of `lookback + max_buffer` frames is all the history needs, and
/// for a short kernel at a small buffer it is barely longer than the block:
/// some tap wraps in nearly every block, the block is cut there, and the
/// frames between two cuts that do not fill a tile are taken one at a time.
/// Room makes the ring come round less often. Measured on the eight-channel
/// A2 capture, against none: 1024 frames is 12% less time at blocks of 16,
/// 6% at 64 and 3% at 128, and 4096 is a percent better again at best, for
/// four times the memory.
///
/// The unit tests run with a few frames instead, so that their rings come
/// round every block or two and each case crosses many cuts; the
/// integration tests and the benchmark run the real one.
const ROOM: usize = if cfg!(test) { 8 } else { 1024 };

/// One plain layer as the file describes it, lifted from the built layer so
/// the weights are read once, one way.
#[derive(Debug, Clone)]
pub(crate) struct PlainLayer {
    pub kernel_size: usize,
    pub dilation: usize,
    /// Per tap, column-major `(channels, channels)`.
    pub conv_w: Vec<Vec<f32>>,
    pub conv_b: Vec<f32>,
    /// The condition mixin, `channels` weights from the one condition.
    pub mixin_w: Vec<f32>,
    /// The 1x1 after the activation, column-major `(channels, channels)`.
    pub l1x1_w: Vec<f32>,
    pub l1x1_b: Vec<f32>,
    pub activation: Activation,
}

/// One fused layer: its weights with every column padded to whole
/// registers, and its input history.
#[derive(Debug, Clone)]
struct Layer {
    kernel_size: usize,
    dilation: usize,
    /// `kernel_size * channels` columns of `4 * V`: tap, then input.
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    mixin_w: Vec<f32>,
    /// `channels` columns of `4 * V`.
    l1x1_w: Vec<f32>,
    l1x1_b: Vec<f32>,
    activation: Activation,
    /// The input history: `cols` frames of `4 * V` floats.
    ring: Vec<f32>,
    cols: usize,
    /// The frame the current block starts at.
    write: usize,
    /// The input this layer sees once the model has settled on silence.
    cached: Vec<f32>,
    has_cached: bool,
}

/// `v` padded with zeros to `width`.
fn padded(v: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0; width];
    out[..v.len()].copy_from_slice(v);
    out
}

/// A column-major `(rows, _)` matrix with every column padded to `width`.
fn padded_columns(m: &[f32], rows: usize, width: usize) -> Vec<f32> {
    m.chunks_exact(rows)
        .flat_map(|c| padded(c, width))
        .collect()
}

/// A flat run of padded frames, as frames of `V` registers.
#[inline(always)]
fn frames<const V: usize>(flat: &[f32]) -> &[[[f32; 4]; V]] {
    flat.as_chunks::<4>().0.as_chunks::<V>().0
}

/// One padded frame, as `V` registers.
#[inline(always)]
fn frame<const V: usize>(flat: &[f32]) -> &[[f32; 4]; V] {
    flat.as_chunks::<4>()
        .0
        .try_into()
        .expect("a frame is V registers")
}

impl Layer {
    fn new(p: PlainLayer, channels: usize, width: usize) -> Self {
        Self {
            kernel_size: p.kernel_size,
            dilation: p.dilation,
            conv_w: p
                .conv_w
                .iter()
                .flat_map(|tap| padded_columns(tap, channels, width))
                .collect(),
            conv_b: padded(&p.conv_b, width),
            mixin_w: padded(&p.mixin_w, width),
            l1x1_w: padded_columns(&p.l1x1_w, channels, width),
            l1x1_b: padded(&p.l1x1_b, width),
            activation: p.activation,
            ring: Vec::new(),
            cols: 0,
            write: 0,
            cached: vec![0.0; width],
            has_cached: false,
        }
    }

    /// How far back the convolution reaches, in frames.
    fn lookback(&self) -> usize {
        (self.kernel_size - 1) * self.dilation
    }

    /// Copy `n` frames of `channels` floats into the history at the write
    /// position, wrapping at the end.
    fn write_frames(&mut self, src: &[f32], channels: usize, n: usize) {
        let width = self.cached.len();
        let first = (self.cols - self.write).min(n);
        for (from, to, count) in [(0, self.write, first), (first, 0, n - first)] {
            let src = &src[from * channels..(from + count) * channels];
            let dst = &mut self.ring[to * width..(to + count) * width];
            if channels == width {
                dst.copy_from_slice(src);
            } else {
                for (d, s) in dst.chunks_exact_mut(width).zip(src.chunks_exact(channels)) {
                    d[..channels].copy_from_slice(s);
                }
            }
        }
    }

    /// The first pass over `n` frames: every tap, the bias and the mixin,
    /// into `z`, a segment at a time.
    #[inline(always)]
    fn convolve<const V: usize, const T: usize, const FUSE: bool>(
        &self,
        channels: usize,
        cond: &[f32],
        z: &mut [f32],
        starts: &mut [usize],
        n: usize,
    ) {
        let taps = Taps::<V> {
            w: frames(&self.conv_w),
            channels,
            bias: frame(&self.conv_b),
            mixin: frame(&self.mixin_w),
            ring: frames(&self.ring),
        };
        let cols = self.cols;
        // Where each tap's frames for this block begin: the last tap reads
        // the block itself, each one before it `dilation` further back.
        let starts = &mut starts[..self.kernel_size];
        for (k, s) in starts.iter_mut().enumerate() {
            let back = (self.kernel_size - 1 - k) * self.dilation;
            *s = (self.write + cols - back) % cols;
        }
        let mut done = 0;
        while done < n {
            // As many frames as every tap can read without wrapping.
            let mut len = n - done;
            for &s in starts.iter() {
                len = len.min(cols - s);
            }
            let cond = &cond[done..done + len];
            let z = &mut z[done * channels..];
            let at = conv_tiles::<V, T, FUSE>(&taps, starts, cond, z, 0);
            conv_tiles::<V, 1, FUSE>(&taps, starts, cond, z, at);
            for s in starts.iter_mut() {
                *s += len;
                if *s == cols {
                    *s = 0;
                }
            }
            done += len;
        }
    }

    /// The third pass over `n` frames: the 1x1, its bias and the residual,
    /// into `to`, a segment at a time.
    #[inline(always)]
    fn mix<const V: usize, const T: usize, const FUSE: bool>(
        &self,
        channels: usize,
        z: &[f32],
        to: Sink<'_>,
        n: usize,
    ) {
        let ring = frames::<V>(&self.ring);
        let one = OneByOne::<V> {
            w: frames(&self.l1x1_w),
            channels,
            bias: frame(&self.l1x1_b),
        };
        let (mut src, mut dst) = (self.write, to.at);
        let mut done = 0;
        while done < n {
            // As many frames as can be read and written without wrapping.
            let len = (n - done).min(self.cols - src).min(to.cols - dst);
            let x = &ring[src..src + len];
            let z = &z[done * channels..];
            let out = &mut to.frames[dst * to.stride..];
            let at = mix_tiles::<V, T, FUSE>(&one, z, x, out, to.stride, 0);
            mix_tiles::<V, 1, FUSE>(&one, z, x, out, to.stride, at);
            src += len;
            if src == self.cols {
                src = 0;
            }
            dst += len;
            if dst == to.cols {
                dst = 0;
            }
            done += len;
        }
    }
}

/// What the first pass reads of a layer: its taps and its history, as
/// frames of `V` registers.
struct Taps<'a, const V: usize> {
    /// One padded column per tap and input, a tap's inputs together.
    w: &'a [[[f32; 4]; V]],
    /// Inputs a tap.
    channels: usize,
    bias: &'a [[f32; 4]; V],
    mixin: &'a [[f32; 4]; V],
    /// The layer's input history.
    ring: &'a [[[f32; 4]; V]],
}

/// What the third pass reads of a layer: its 1x1.
struct OneByOne<'a, const V: usize> {
    /// One padded column per input.
    w: &'a [[[f32; 4]; V]],
    /// Inputs.
    channels: usize,
    bias: &'a [[f32; 4]; V],
}

/// Where the third pass writes: frames `stride` floats apart, in a buffer
/// that wraps after `cols` of them, from frame `at`.
struct Sink<'a> {
    frames: &'a mut [f32],
    stride: usize,
    cols: usize,
    at: usize,
}

/// One tap's contribution to a tile of `T` frames: for each frame and each
/// output register, `acc += w[i] * x[i]` over the inputs `i`, ascending.
///
/// `w` is one padded column per input and `x` the tile's frames. The inputs
/// are taken four at a time, one load per frame, each then multiplying as a
/// lane of it; the last one to three are lanes of one more load.
#[inline(always)]
fn chain<const V: usize, const T: usize, const FUSE: bool>(
    acc: &mut [[F4; V]; T],
    w: &[[[f32; 4]; V]],
    x: [&[[f32; 4]; V]; T],
) {
    // One input: its column against one lane of each frame's register.
    macro_rules! input {
        ($column:expr, $x:expr, $lane:literal) => {{
            let column: [F4; V] = std::array::from_fn(|v| F4::load(&$column[v]));
            for (a, x) in acc.iter_mut().zip(&$x) {
                for (a, w) in a.iter_mut().zip(&column) {
                    *a = a.macc_lane::<FUSE>(*w, *x, $lane);
                }
            }
        }};
    }
    let (w4, w1) = w.as_chunks::<4>();
    // There are at most `V` fours of inputs, which `take` and `min` say in
    // a way the compiler can see, so a frame's registers index unchecked.
    for (q, w) in w4.iter().enumerate().take(V) {
        let x: [F4; T] = std::array::from_fn(|t| F4::load(&x[t][q]));
        input!(w[0], x, 0);
        input!(w[1], x, 1);
        input!(w[2], x, 2);
        input!(w[3], x, 3);
    }
    if let Some(w0) = w1.first() {
        let q = w4.len().min(V - 1);
        let x: [F4; T] = std::array::from_fn(|t| F4::load(&x[t][q]));
        input!(w0, x, 0);
        if let Some(w) = w1.get(1) {
            input!(w, x, 1);
        }
        if let Some(w) = w1.get(2) {
            input!(w, x, 2);
        }
    }
}

/// The first pass over one segment, `T` frames at a time from frame `at`
/// for as many whole tiles as `cond` has frames left; returns the frame it
/// stopped at.
///
/// For each tile: every tap's [`chain`] from zero, added to the running sum
/// from zero in tap order; then the bias; then the mixin, one product onto
/// zero and added to zero. `starts[k]` is the ring frame tap `k` reads
/// first in this segment, and `z` the segment's frames, `channels` apart.
#[inline(always)]
fn conv_tiles<const V: usize, const T: usize, const FUSE: bool>(
    taps: &Taps<'_, V>,
    starts: &[usize],
    cond: &[f32],
    z: &mut [f32],
    mut at: usize,
) -> usize {
    let zero = F4::zero();
    let channels = taps.channels;
    let bias: [F4; V] = std::array::from_fn(|v| F4::load(&taps.bias[v]));
    let mixin: [F4; V] = std::array::from_fn(|v| F4::load(&taps.mixin[v]));
    while at + T <= cond.len() {
        let mut sum = [[zero; V]; T];
        for (w, &start) in taps.w.chunks_exact(channels).zip(starts) {
            // The tile's frames for this tap, checked once here.
            let x: &[[[f32; 4]; V]; T] = taps.ring[start + at..start + at + T]
                .try_into()
                .expect("a tile of frames");
            let mut acc = [[zero; V]; T];
            chain::<V, T, FUSE>(&mut acc, w, x.each_ref());
            for (s, a) in sum.iter_mut().zip(&acc) {
                for (s, a) in s.iter_mut().zip(a) {
                    *s = s.add(*a);
                }
            }
        }
        for (t, s) in sum.iter().enumerate() {
            let c = F4::splat(cond[at + t]);
            // A frame is `channels` floats but is written as whole
            // registers; the lanes past it are the next frame's, which is
            // written after this one.
            let frame = &mut z[(at + t) * channels..(at + t) * channels + 4 * V];
            let (out, _) = frame.as_chunks_mut::<4>();
            for v in 0..V {
                let mix = zero.add(zero.macc::<FUSE>(mixin[v], c));
                s[v].add(bias[v]).add(mix).store(&mut out[v]);
            }
        }
        at += T;
    }
    at
}

/// The third pass over one segment, `T` frames at a time from frame `at`
/// for as many whole tiles as `x` has frames left; returns the frame it
/// stopped at.
///
/// For each tile: the 1x1's [`chain`] over the activated frame from zero,
/// added to zero, then its bias, then the layer's input `x`. `z` is the
/// segment's activated frames, `channels` apart, and `out` where the
/// segment's results go, `out_stride` apart.
#[inline(always)]
fn mix_tiles<const V: usize, const T: usize, const FUSE: bool>(
    one: &OneByOne<'_, V>,
    z: &[f32],
    x: &[[[f32; 4]; V]],
    out: &mut [f32],
    out_stride: usize,
    mut at: usize,
) -> usize {
    let zero = F4::zero();
    let channels = one.channels;
    let bias: [F4; V] = std::array::from_fn(|v| F4::load(&one.bias[v]));
    while at + T <= x.len() {
        // An activated frame is `channels` floats but is read as whole
        // registers; the lanes past it are never used as inputs. The frames
        // are `channels` apart, so they are sliced one by one.
        let zf = |t: usize| frame::<V>(&z[(at + t) * channels..(at + t) * channels + 4 * V]);
        let mut zt = [zf(0); T];
        for (t, f) in zt.iter_mut().enumerate().skip(1) {
            *f = zf(t);
        }
        let mut acc = [[zero; V]; T];
        chain::<V, T, FUSE>(&mut acc, one.w, zt);
        let x: &[[[f32; 4]; V]; T] = x[at..at + T].try_into().expect("a tile of frames");
        for (t, a) in acc.iter().enumerate() {
            let frame = &mut out[(at + t) * out_stride..(at + t) * out_stride + 4 * V];
            let (o, _) = frame.as_chunks_mut::<4>();
            for v in 0..V {
                let residual = zero.add(a[v]).add(bias[v]);
                F4::load(&x[t][v]).add(residual).store(&mut o[v]);
            }
        }
        at += T;
    }
    at
}

/// The layers of a plain array, fused.
#[derive(Debug, Clone)]
pub(crate) struct FusedLayers {
    channels: usize,
    /// Registers per frame.
    lanes: usize,
    layers: Vec<Layer>,
    /// The block's pre-activation, then its activation: `channels` floats a
    /// frame and a register of slack.
    z: Vec<f32>,
    /// `(channels, max_buffer + 1)`: the last layer's output, for the array
    /// after this one. The extra column is a register of slack.
    output: Buf,
    /// Whether anything reads `output`. Nothing reads the last array's.
    wants_output: bool,
    /// Scratch for a layer's tap positions, one per tap.
    starts: Vec<usize>,
}

impl FusedLayers {
    /// Fuse `layers`, each `channels` wide, or `None` when the array is
    /// wider than this path takes or its padded histories would exceed what
    /// a model may keep.
    ///
    /// `wants_output` says whether another array follows and reads the last
    /// layer's output; when none does, that layer's 1x1 and residual are
    /// never computed, since nothing can see them.
    pub(crate) fn new(
        channels: usize,
        layers: Vec<PlainLayer>,
        wants_output: bool,
    ) -> Option<Self> {
        let lanes = channels.div_ceil(4);
        if channels < 2 || lanes > MAX_LANES || layers.is_empty() {
            return None;
        }
        let width = 4 * lanes;
        let layers: Vec<Layer> = layers
            .into_iter()
            .map(|l| Layer::new(l, channels, width))
            .collect();
        // The file's histories were counted at `channels` floats a frame
        // when the model was built; these are `width`, and have their room.
        let history = layers.iter().try_fold(0usize, |sum, l| {
            sum.checked_add(l.lookback().checked_add(ROOM)?.checked_mul(width)?)
        })?;
        if history > MAX_HISTORY_FLOATS {
            return None;
        }
        let taps = layers.iter().map(|l| l.kernel_size).max().unwrap_or(0);
        Some(Self {
            channels,
            lanes,
            layers,
            z: Vec::new(),
            output: Buf::new(),
            wants_output,
            starts: vec![0; taps],
        })
    }

    /// Reserve the histories and the block buffers for blocks of up to
    /// `max_buffer` frames, cleared. Allocating.
    pub(crate) fn set_max_buffer_size(&mut self, max_buffer: usize) {
        let width = 4 * self.lanes;
        for l in &mut self.layers {
            l.cols = l.lookback() + max_buffer.max(ROOM);
            l.ring = vec![0.0; l.cols * width];
            l.write = 0;
            // The cached steady state is one frame and does not depend on
            // the block size, so it survives, as `Conv1D`'s does.
        }
        self.z = vec![0.0; max_buffer * self.channels + width];
        if self.wants_output {
            self.output.resize(self.channels, max_buffer + 1);
        }
    }

    /// The last layer's output for the block, `channels` rows.
    pub(crate) fn output(&self) -> &Buf {
        &self.output
    }

    /// True when a steady state has been cached for every layer.
    pub(crate) fn has_cached_prewarm_state(&self) -> bool {
        self.layers.iter().all(|l| l.has_cached)
    }

    /// Snapshot each layer's last input frame as its steady state.
    pub(crate) fn cache_prewarm_state(&mut self) {
        let width = 4 * self.lanes;
        for l in &mut self.layers {
            if l.cols == 0 {
                continue;
            }
            let last = (l.write + l.cols - 1) % l.cols;
            l.cached
                .copy_from_slice(&l.ring[last * width..(last + 1) * width]);
            l.has_cached = true;
        }
    }

    /// Refill every history from the cached steady state.
    pub(crate) fn prewarm_from_cache(&mut self) {
        for l in &mut self.layers {
            debug_assert!(l.has_cached);
            for f in l.ring.chunks_exact_mut(l.cached.len()) {
                f.copy_from_slice(&l.cached);
            }
            l.write = 0;
        }
    }

    /// Run the layers over `n` frames.
    ///
    /// `input` is the array's rechannelled input, `channels` rows;
    /// `condition` the one-row condition; and `head` the array's skip
    /// accumulator, `channels` rows, to which every layer's activation is
    /// added. The last layer's output is left in [`FusedLayers::output`]
    /// when the array was built to keep it.
    pub(crate) fn process(&mut self, input: &Buf, condition: &Buf, head: &mut Buf, n: usize) {
        if n == 0 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if crate::kernels::fused() {
            // SAFETY: `fused()` reported the CPU has FMA and AVX.
            return unsafe { self.process_fma(input, condition, head, n) };
        }
        self.process_lanes::<{ !cfg!(target_arch = "x86_64") }>(input, condition, head, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx,fma")]
    unsafe fn process_fma(&mut self, input: &Buf, condition: &Buf, head: &mut Buf, n: usize) {
        self.process_lanes::<true>(input, condition, head, n)
    }

    /// The pass for this array's width. The tile is the frames whose sums
    /// stay in registers: four, and eight where a frame is one register.
    #[inline(always)]
    fn process_lanes<const FUSE: bool>(
        &mut self,
        input: &Buf,
        condition: &Buf,
        head: &mut Buf,
        n: usize,
    ) {
        match self.lanes {
            1 => self.run::<1, 8, FUSE>(input, condition, head, n),
            2 => self.run::<2, 4, FUSE>(input, condition, head, n),
            3 => self.run::<3, 4, FUSE>(input, condition, head, n),
            _ => self.run::<4, 4, FUSE>(input, condition, head, n),
        }
    }

    #[inline(always)]
    fn run<const V: usize, const T: usize, const FUSE: bool>(
        &mut self,
        input: &Buf,
        condition: &Buf,
        head: &mut Buf,
        n: usize,
    ) {
        let Self {
            channels,
            layers,
            z,
            output,
            wants_output,
            starts,
            ..
        } = self;
        let c = *channels;
        debug_assert_eq!(input.rows(), c);
        debug_assert_eq!(condition.rows(), 1);
        debug_assert_eq!(head.rows(), c);
        let cond = condition.left(n);
        layers[0].write_frames(input.left(n), c, n);
        for i in 0..layers.len() {
            let (done, rest) = layers.split_at_mut(i + 1);
            let layer = &mut done[i];
            layer.convolve::<V, T, FUSE>(c, cond, z, starts, n);
            layer.activation.apply(&mut z[..n * c]);
            // The skip path takes the activation itself.
            for (h, z) in head.left_mut(n).iter_mut().zip(&z[..n * c]) {
                *h += *z;
            }
            match rest.first_mut() {
                // Straight into the next layer's history, at its block.
                Some(next) => {
                    let to = Sink {
                        frames: &mut next.ring,
                        stride: 4 * V,
                        cols: next.cols,
                        at: next.write,
                    };
                    layer.mix::<V, T, FUSE>(c, z, to, n);
                }
                // A block buffer starts at its first frame and never wraps.
                None if *wants_output => {
                    let to = Sink {
                        frames: output.data_mut(),
                        stride: c,
                        cols: usize::MAX,
                        at: 0,
                    };
                    layer.mix::<V, T, FUSE>(c, z, to, n);
                }
                None => {}
            }
            layer.write = (layer.write + n) % layer.cols;
        }
    }
}

#[cfg(test)]
mod tests {
    //! The fused path against the generic path: bit for bit, over every
    //! width it takes, random signals in ragged blocks, with and without a
    //! prewarm, and through a second reset.

    use super::{FusedLayers, PlainLayer, ROOM};
    use crate::activations::Activation;
    use crate::engine::Engine;
    use crate::error::Error;
    use crate::format::{ArchConfig, MAX_HISTORY_FLOATS, parse_value};
    use crate::testutil::splitmix_stream;
    use crate::wavenet::WaveNet;
    use serde_json::{Value, json};
    use std::path::Path;

    /// The model as the file describes it, every layer on the generic path.
    fn generic(v: &Value) -> WaveNet {
        let file = parse_value(v).unwrap();
        match file.config {
            ArchConfig::WaveNet(cfg) => WaveNet::new(cfg, &file.weights).unwrap(),
            _ => panic!("not a WaveNet"),
        }
    }

    /// `doc` with as many weights as it consumes, found by asking: a model
    /// built with too few says how many the next convolution wanted.
    fn with_weights(mut doc: Value, seed: u64, scale: f32) -> Value {
        let mut n = 0;
        loop {
            doc["weights"] = json!(vec![0.0; n]);
            let file = parse_value(&doc).unwrap();
            let ArchConfig::WaveNet(cfg) = file.config else {
                panic!("not a WaveNet")
            };
            match WaveNet::new(cfg, &file.weights) {
                Ok(_) => break,
                Err(Error::WeightCount { expected, .. }) => n = expected,
                Err(e) => panic!("{e}"),
            }
        }
        let w: Vec<f32> = splitmix_stream(seed, n).iter().map(|u| u * scale).collect();
        doc["weights"] = json!(w);
        doc
    }

    /// Two arrays the way NAM's A1 presets have them: `c0` channels then
    /// `c1`, one activation, the legacy one-tap heads.
    fn a1(c0: usize, c1: usize, kernel: usize, activation: Value) -> Value {
        json!({
            "version": "0.5.4", "architecture": "WaveNet", "sample_rate": 48000,
            "config": {"layers": [
                {"input_size": 1, "condition_size": 1, "head_size": c1, "channels": c0,
                 "kernel_size": kernel, "dilations": [1, 2, 5, 11], "activation": activation,
                 "gated": false, "head_bias": false},
                {"input_size": c0, "condition_size": 1, "head_size": 1, "channels": c1,
                 "kernel_size": kernel, "dilations": [7, 1, 3], "activation": activation,
                 "gated": false, "head_bias": true},
            ], "head": null, "head_scale": 0.02},
        })
    }

    /// One array the way A2 has it: a kernel per layer, a head convolution.
    fn a2(c: usize, kernels: &[usize], dilations: &[usize], activation: Value) -> Value {
        json!({
            "version": "0.7.0", "architecture": "WaveNet", "sample_rate": 48000,
            "config": {"layers": [{
                "input_size": 1, "condition_size": 1,
                "head": {"out_channels": 1, "kernel_size": 5, "bias": true},
                "channels": c, "bottleneck": c, "kernel_sizes": kernels,
                "dilations": dilations, "activation": activation,
                "layer1x1": {"active": true, "groups": 1},
            }], "head": null, "head_scale": 0.01},
        })
    }

    fn bundled(name: &str) -> Value {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("assets/models/{name}.nam"));
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    /// What is done to a model between building it and the end of a case.
    enum Step<'a> {
        Size(usize),
        Reset(usize),
        Prewarm,
        /// Process this signal in blocks cycling through this schedule.
        Run(&'a [f32], &'a [usize]),
    }

    fn run(engine: &mut dyn Engine, steps: &[Step<'_>]) -> Vec<f32> {
        let mut out = Vec::new();
        for step in steps {
            match step {
                Step::Size(n) => engine.set_max_buffer_size(*n),
                Step::Reset(n) => engine.reset(48_000.0, *n),
                Step::Prewarm => engine.prewarm(),
                Step::Run(input, blocks) => {
                    let (mut pos, mut i) = (0, 0);
                    while pos < input.len() {
                        let n = blocks[i % blocks.len()].min(input.len() - pos);
                        let mut block = vec![0.0f32; n];
                        engine.process(&[&input[pos..pos + n]], &mut [&mut block], n);
                        out.extend_from_slice(&block);
                        pos += n;
                        i += 1;
                    }
                }
            }
        }
        out
    }

    /// Both paths through the same steps, compared to the bit.
    fn same(name: &str, doc: &Value, steps: &[Step<'_>]) {
        let want = run(&mut generic(doc), steps);
        let fused = generic(doc).fused();
        assert!(fused.fused_arrays() > 0, "{name}: nothing was fused");
        let mut fused = fused;
        let got = run(&mut fused, steps);
        assert_eq!(want.len(), got.len(), "{name}");
        let first = want
            .iter()
            .zip(&got)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        if let Some(i) = first {
            panic!(
                "{name}: first difference at frame {i}: {} ({:#010x}) vs {} ({:#010x})",
                want[i],
                want[i].to_bits(),
                got[i],
                got[i].to_bits()
            );
        }
        assert!(
            want.iter().any(|v| v.abs() > 1e-6),
            "{name}: the model made no sound"
        );
    }

    #[test]
    fn a_plain_array_is_fused_and_any_other_is_left_alone() {
        let tanh = json!("Tanh");
        let count = |doc: Value| generic(&with_weights(doc, 1, 0.1)).fused().fused_arrays();
        assert_eq!(count(a1(16, 8, 3, tanh.clone())), 2);
        assert_eq!(count(a1(12, 6, 3, tanh.clone())), 2);
        assert_eq!(count(a1(2, 2, 3, tanh.clone())), 2);
        // Seventeen channels is past the widest this path takes; the array
        // after it is still plain.
        assert_eq!(count(a1(17, 8, 3, tanh.clone())), 1);
        // One channel is stored depthwise, which is other arithmetic.
        assert_eq!(count(a1(1, 4, 3, tanh.clone())), 1);
        assert_eq!(count(a2(8, &[6, 15], &[1, 13], tanh.clone())), 1);

        let mut gated = a1(8, 4, 3, tanh.clone());
        gated["config"]["layers"][0]["gated"] = json!(true);
        assert_eq!(count(gated), 1);

        let mut grouped = a2(8, &[3], &[1], tanh.clone());
        grouped["config"]["layers"][0]["groups_input"] = json!(2);
        assert_eq!(count(grouped), 0);

        let mut bottleneck = a2(8, &[3], &[1], tanh);
        bottleneck["config"]["layers"][0]["bottleneck"] = json!(4);
        assert_eq!(count(bottleneck), 0);

        // The bundled captures: the standard one and both A2 children are
        // plain throughout; the one that conditions on a nested model's
        // three channels and the one with every A2 feature are not.
        for (name, want) in [
            ("wavenet_a1_standard", 2),
            ("wavenet", 2),
            ("wavenet_condition_dsp", 0),
            ("wavenet_a2_feature_test", 0),
        ] {
            assert_eq!(
                generic(&bundled(name)).fused().fused_arrays(),
                want,
                "{name}"
            );
        }
        for child in bundled("a2")["config"]["submodels"].as_array().unwrap() {
            assert_eq!(generic(&child["model"]).fused().fused_arrays(), 1);
        }
    }

    /// A two-channel plain layer of this shape, every weight zero.
    fn silent(kernel_size: usize, dilation: usize) -> PlainLayer {
        PlainLayer {
            kernel_size,
            dilation,
            conv_w: vec![vec![0.0; 4]; kernel_size],
            conv_b: vec![0.0; 2],
            mixin_w: vec![0.0; 2],
            l1x1_w: vec![0.0; 4],
            l1x1_b: vec![0.0; 2],
            activation: Activation::ReLU,
        }
    }

    #[test]
    fn a_ring_has_room_past_its_lookback_however_short_the_block() {
        let mut fused = FusedLayers::new(2, vec![silent(6, 1), silent(3, 512)], false).unwrap();
        let cols = |f: &FusedLayers| f.layers.iter().map(|l| l.cols).collect::<Vec<_>>();
        // A block shorter than the room: the room.
        fused.set_max_buffer_size(ROOM / 2);
        assert_eq!(cols(&fused), [5 + ROOM, 1024 + ROOM]);
        // A block longer than it: the block, which the history cannot do
        // without.
        fused.set_max_buffer_size(4 * ROOM);
        assert_eq!(cols(&fused), [5 + 4 * ROOM, 1024 + 4 * ROOM]);
        // Two channels are one register a frame.
        assert!(fused.layers.iter().all(|l| l.ring.len() == 4 * l.cols));
    }

    #[test]
    fn histories_too_long_with_their_room_are_left_to_the_generic_path() {
        // One register a frame, so a quarter of the limit in frames; the
        // limit counts the room, which is memory like any other.
        let fits = MAX_HISTORY_FLOATS / 4 - ROOM;
        let fused = |dilation: usize| FusedLayers::new(2, vec![silent(2, dilation)], false);
        assert!(fused(fits).is_some());
        assert!(fused(fits + 1).is_none());
        // Every layer has its own.
        let two =
            |dilation: usize| FusedLayers::new(2, vec![silent(2, dilation), silent(1, 1)], false);
        assert!(two(fits - ROOM).is_some());
        assert!(two(fits - ROOM + 1).is_none());
    }

    #[test]
    fn the_fused_path_is_the_generic_path_bit_for_bit_at_every_width() {
        let input = splitmix_stream(7, 700);
        let input: Vec<f32> = input.iter().map(|u| u * 0.5).collect();
        let leaky = json!({"type": "LeakyReLU", "negative_slope": 0.01});
        for c in 2..=16usize {
            // A register and a bit, a kernel of one tap and a long one, and
            // dilations either side of the block.
            let half = c.div_ceil(2).max(2);
            let scale = 0.5 / (c as f32).sqrt();
            let shapes = [
                (format!("a1 {c}/{half}"), a1(c, half, 3, json!("Tanh"))),
                (
                    format!("a2 {c}"),
                    a2(c, &[6, 1, 15, 2], &[1, 9, 2, 40], leaky.clone()),
                ),
            ];
            for (name, doc) in shapes {
                let doc = with_weights(doc, c as u64, scale);
                for (blocks, max) in [
                    (&[64usize][..], 64),
                    (&[1, 7, 64, 3, 128, 33][..], 128),
                    (&[1][..], 1),
                    (&[5][..], 6),
                ] {
                    let steps = [Step::Reset(max), Step::Run(&input, blocks)];
                    same(&format!("{name}, blocks {blocks:?}"), &doc, &steps);
                }
                // Without a prewarm, from cleared histories.
                let steps = [Step::Size(32), Step::Run(&input, &[32, 9])];
                same(&format!("{name}, unsettled"), &doc, &steps);
            }
        }
    }

    #[test]
    fn every_activation_runs_over_the_frames_the_generic_path_gives_it() {
        // The pre-activation is kept at the layer's own width for this: a
        // per-channel slope table is indexed by position, so the positions
        // have to be the generic path's.
        let input: Vec<f32> = splitmix_stream(11, 500).iter().map(|u| u * 0.8).collect();
        for activation in [
            json!("Tanh"),
            json!("Hardtanh"),
            json!("Fasttanh"),
            json!("ReLU"),
            json!("LeakyReLU"),
            json!({"type": "PReLU", "negative_slopes": [0.1, 0.2, 0.05, 0.3, 0.15, 0.25]}),
            json!({"type": "PReLU", "negative_slopes": [0.1, 0.2, 0.05, 0.3]}),
            json!("Sigmoid"),
            json!("SiLU"),
            json!("Hardswish"),
            json!({"type": "LeakyHardtanh", "min_val": -0.5, "max_val": 0.7,
                   "min_slope": 0.02, "max_slope": 0.03}),
            json!("Softsign"),
        ] {
            for c in [3usize, 6, 8] {
                let doc = with_weights(a1(c, 6, 2, activation.clone()), 3, 0.25);
                let steps = [Step::Reset(48), Step::Run(&input, &[48, 5, 17])];
                same(&format!("{activation} at {c} channels"), &doc, &steps);
            }
        }
    }

    #[test]
    fn the_bundled_captures_come_out_the_same() {
        let input: Vec<f32> = splitmix_stream(5, 2600).iter().map(|u| u * 0.4).collect();
        let mut docs = vec![
            (
                "wavenet_a1_standard".to_string(),
                bundled("wavenet_a1_standard"),
            ),
            ("wavenet".to_string(), bundled("wavenet")),
        ];
        for child in bundled("a2")["config"]["submodels"].as_array().unwrap() {
            let c = &child["model"]["config"]["layers"][0]["channels"];
            docs.push((format!("a2, {c} channels"), child["model"].clone()));
        }
        for (name, doc) in docs {
            let steps = [Step::Reset(64), Step::Run(&input, &[64, 1, 37])];
            same(&name, &doc, &steps);
        }
    }

    #[test]
    fn a_second_reset_restores_the_steady_state_the_first_one_cached() {
        let a: Vec<f32> = splitmix_stream(21, 400).iter().map(|u| u * 0.5).collect();
        let b: Vec<f32> = splitmix_stream(22, 400).iter().map(|u| u * 0.5).collect();
        for c in [3usize, 8, 13] {
            let doc = with_weights(a1(c, 5, 3, json!("Tanh")), 9, 0.2);
            // A reset, a run, a reset at another size, a run; then a bare
            // prewarm, which restores without resizing, and a run.
            let steps = [
                Step::Reset(64),
                Step::Run(&a, &[64, 3]),
                Step::Reset(17),
                Step::Run(&b, &[17, 2]),
                Step::Prewarm,
                Step::Run(&a, &[9]),
            ];
            same(&format!("{c} channels"), &doc, &steps);
        }
    }

    #[test]
    fn sums_that_underflow_keep_the_sign_the_generic_path_gives_them() {
        // Weights and a signal small enough that products underflow, and
        // stretches of exact silence: where a negative zero can arise, and
        // where the additions of zero the kernels keep are not no-ops.
        let mut input: Vec<f32> = splitmix_stream(31, 600)
            .iter()
            .map(|u| u * 1.0e-30)
            .collect();
        input[100..250].fill(0.0);
        input[300..320].fill(-0.0);
        for c in [2usize, 4, 7, 16] {
            for activation in [
                json!("ReLU"),
                json!({"type": "LeakyReLU", "negative_slope": 0.5}),
            ] {
                let mut doc = with_weights(a1(c, 4, 3, activation), 13, 1.0e-12);
                // Half the weights exactly zero, of either sign, so some
                // sums are zero outright rather than by underflow.
                for (i, w) in doc["weights"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    match i % 4 {
                        0 => *w = json!(0.0),
                        1 => *w = json!(-0.0),
                        _ => {}
                    }
                }
                let want = run(
                    &mut generic(&doc),
                    &[Step::Size(64), Step::Run(&input, &[64, 5])],
                );
                let got = run(
                    &mut generic(&doc).fused(),
                    &[Step::Size(64), Step::Run(&input, &[64, 5])],
                );
                for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{c} channels, frame {i}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
