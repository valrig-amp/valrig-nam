//! Convolution kernels.
//!
//! # Shape specialisation
//!
//! The C++ reference carries hand-written specialisations for a handful of
//! channel counts and kernel sizes, and anything outside that set falls into
//! a generic path. The model shape is known at load time and never changes,
//! so it is a code-generation problem: [`matmul`] is written once and
//! monomorphised for every output width in `for_each_width!`, and
//! [`select_matmul`] hands a layer the function pointer for its width when it
//! is built. The *input* count stays a run-time number: it is only a loop
//! bound, where the output width is the shape of the accumulator and has to
//! be known for that to live in registers. So one kernel per width serves
//! every input count, and a capture whose layers are twelve channels wide
//! runs the same code as one whose layers are sixteen. Widths outside the
//! table run [`matmul_dyn`], the same arithmetic with runtime bounds, written
//! to be good rather than to be a cliff.
//!
//! # Weight reuse
//!
//! A 16-channel layer streams its whole weight matrix per frame for a few
//! hundred multiply-adds. The batched kernels load a weight column once and
//! apply it to a tile of frames before touching the next one, cutting weight
//! traffic by the tile's size; [`tile_for`] picks the tile per width.
//!
//! # Vectorisation
//!
//! The accumulator is indexed by *output channel*, and distinct output
//! channels are independent sums, so widening that loop to a SIMD register
//! cannot reorder any addition. The row step, one input's contribution to
//! every output channel, is written out in [`F4`] lanes, because rustc's
//! auto-vectoriser does not take that loop at 16 channels and the scalar
//! form it emits spills.
//!
//! The inputs are read four at a time as well: one load brings in four
//! consecutive inputs of a frame, and each multiply then takes one lane of
//! it. A scalar load per input per frame is one load for every one to four
//! multiply-adds, and at that ratio the loads cost as much as the
//! arithmetic: on an Apple M3, `kernel_bench` below measured 34
//! multiply-adds per nanosecond that way at 8 × 8 and 39 this way, and 37
//! against 45 at 8 × 16.
//!
//! Two bounds checks the compiler cannot prove away are hoisted: a tile's
//! output is taken as whole `[f32; OUT]` frames once so the write-back loop
//! vectorises, and a tile's input frames are sliced once per tile rather
//! than indexed through a runtime stride inside the inner loop.
//!
//! # Frames as lanes
//!
//! An output of one to three channels has no lanes to speak of: [`matmul`]
//! runs it one multiply-add at a time. Those outputs take the other axis.
//! [`frames`] computes four consecutive *frames* of one output channel in
//! one register: each input is read from a planar history
//! (`history::Planes`), four frames to a load, and multiplied by one lane of
//! the output's register of weights. Distinct frames are independent sums
//! as distinct channels are, so nothing is added in a different order; what
//! each frame of each output gets is [`matmul`]'s chain over the inputs from
//! zero, tap by tap. The head of an A2 capture is this shape, one output
//! from sixteen taps, and so is every layer of a model three channels wide,
//! which `a2_fast` runs through the same kernel.
//!
//! # Bit-exactness
//!
//! Every kernel in this module accumulates in exactly the same order:
//! `acc[o] += w[o, i] * x[i]` for `i` ascending, starting from `0.0`, with
//! the bias added last and separately by the caller. The specialised,
//! dynamic and batched paths differ only in loop bounds, in which frames
//! advance together and in how an input reaches the multiplier, so they are
//! bit-identical to *each other* by construction. The tests below check it
//! anyway, and `tests/reference.rs` checks the whole-model consequence
//! against output from the C++ reference.
//!
//! # Fused multiply-add
//!
//! The multiply-accumulate step fuses where the machine has the instruction:
//! one rounding, where `a * b + c` is two. The C++ reference makes the same
//! choice at its own release flags, where Eigen's `pmadd` lowers to a fused
//! multiply-add. Rust never contracts on its own, so every fusion here is a
//! written one, through [`macc`].
//!
//! What it costs is bit-identical output against one pinned build of the
//! reference; what stays is agreement to within that reference's own
//! build-to-build spread, recorded per case in `assets/expectations.txt`.
//! What it buys is up to a quarter of the throughput on the large captures.
//!
//! Only the *vector-lane* accumulations of the convolutions fuse. Where the
//! accumulator is a single scalar carried across a loop, in the
//! `out_ch > DYN_ACC` branch of [`matvec_dyn_body`], fusing is slower,
//! because the whole loop is one serial dependency chain and a fused
//! multiply-add is a longer link in it than an add; that site is written
//! unfused. So is the LSTM's gate product, in `lstm`, which the reference
//! computes unfused and whose vectors are pinned that way.
//!
//! On x86-64 the fused instructions are not in the baseline, so [`fused`]
//! asks the CPU once and every kernel follows its answer: each has a twin
//! compiled for AVX and FMA, entered only after that check, and the unfused
//! form otherwise. On aarch64 NEON is baseline and always fuses.
//!
//! # Layout
//!
//! Weight matrices are column-major `(out_channels, in_channels)`, matching
//! `Eigen::MatrixXf`: `w[o, i]` lives at `w[o + i * OUT]`. Frames are
//! columns, so one frame of activations is contiguous.

use crate::lanes::F4;

/// One multiply-accumulate step, fused: one instruction and one rounding,
/// where `acc + a * b` would be two of each.
#[inline(always)]
pub(crate) fn macc(acc: f32, a: f32, b: f32) -> f32 {
    a.mul_add(b, acc)
}

/// One multiply-accumulate step, fused or not by the const.
#[inline(always)]
fn step<const FUSE: bool>(acc: f32, a: f32, b: f32) -> f32 {
    if FUSE { macc(acc, a, b) } else { acc + a * b }
}

/// Whether this machine fuses: yes wherever the fused instruction is in the
/// target's baseline, and on x86-64 exactly when the CPU has FMA and AVX.
///
/// One answer per process, and every kernel here follows it, so they stay
/// bit-identical to each other on every machine. Without the instruction,
/// fusing would mean a library call per multiply-accumulate; the reference
/// does not fuse on such a machine either.
#[inline(always)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn fused() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// Four lanes of `acc[k] = acc[k] + w[k] * x`, fused by `FUSE`, as one
/// vector operation where the target has one.
///
/// A lane is one output channel, so the vector form adds nothing in a
/// different order: it is the scalar loop bit for bit.
#[inline(always)]
fn lane4<const FUSE: bool>(acc: &mut [f32; 4], w: &[f32; 4], x: f32) {
    F4::load(acc)
        .macc::<FUSE>(F4::load(w), F4::splat(x))
        .store(acc);
}

/// One input's contribution to every output channel of one frame:
/// `acc[o] = acc[o] + w[o] * x` for each `o`, fused by `FUSE`, four lanes at
/// a time with the odd lanes one at a time.
#[inline(always)]
pub(crate) fn macc_row<const OUT: usize, const FUSE: bool>(
    acc: &mut [f32; OUT],
    w: &[f32; OUT],
    x: f32,
) {
    let (a4, ar) = acc.as_chunks_mut::<4>();
    let (w4, wr) = w.as_chunks::<4>();
    for (a, w) in a4.iter_mut().zip(w4) {
        lane4::<FUSE>(a, w, x);
    }
    for (a, w) in ar.iter_mut().zip(wr) {
        *a = step::<FUSE>(*a, *w, x);
    }
}

/// [`macc_row`] with the input arriving as one lane of four: `x` holds four
/// consecutive inputs of a frame and `lane` says which this step is. The
/// whole vectors of `acc` multiply by that lane directly; the odd lanes
/// take `xs`, the same input as a scalar.
#[inline(always)]
fn macc_row_lane<const OUT: usize, const FUSE: bool>(
    acc: &mut [f32; OUT],
    w: &[f32; OUT],
    x: F4,
    lane: usize,
    xs: f32,
) {
    let (a4, ar) = acc.as_chunks_mut::<4>();
    let (w4, wr) = w.as_chunks::<4>();
    for (a, w) in a4.iter_mut().zip(w4) {
        F4::load(a).macc_lane::<FUSE>(F4::load(w), x, lane).store(a);
    }
    for (a, w) in ar.iter_mut().zip(wr) {
        *a = step::<FUSE>(*a, *w, xs);
    }
}

/// Defines `$name` as the machine-appropriate face of `$body::<…, FUSE>`.
///
/// On x86-64 it also defines `$fma`, the body compiled for AVX and FMA, and
/// enters it when [`fused`] reports the CPU has them; otherwise it runs the
/// unfused body. On every other target it runs the body fused, because the
/// fused instruction is in the baseline there.
macro_rules! fused_dispatch {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident / $fma:ident $(<$(const $g:ident: usize),+>)?
            ($($arg:ident: $ty:ty),* $(,)?) => $body:ident
    ) => {
        $(#[$meta])*
        $vis fn $name$(<$(const $g: usize),+>)?($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            if fused() {
                // SAFETY: `fused()` reported the CPU has FMA and AVX.
                return unsafe { $fma$(::<$($g),+>)?($($arg),*) };
            }
            $body::<$($($g,)+)? { !cfg!(target_arch = "x86_64") }>($($arg),*)
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx,fma")]
        unsafe fn $fma$(<$(const $g: usize),+>)?($($arg: $ty),*) {
            $body::<$($($g,)+)? true>($($arg),*)
        }
    };
}

/// The output widths that get a monomorphised kernel.
///
/// Every width up to 16, which is every layer of NAM's standard, lite,
/// feather and nano WaveNets (16, 12, 8, 6, 4 and 2 channels) and of A2 (8
/// and 3), plus the wider multiples of four that gating and FiLM double
/// those into. The input count is not part of the table: one kernel per
/// width takes any. Anything else runs [`matmul_dyn`].
/// `the_bundled_captures_stay_on_the_table` checks the table against the
/// bundled models and the trainer's presets, so a common width cannot
/// quietly fall back.
macro_rules! for_each_width {
    ($mac:ident) => {
        $mac!(1);
        $mac!(2);
        $mac!(3);
        $mac!(4);
        $mac!(5);
        $mac!(6);
        $mac!(7);
        $mac!(8);
        $mac!(9);
        $mac!(10);
        $mac!(11);
        $mac!(12);
        $mac!(13);
        $mac!(14);
        $mac!(15);
        $mac!(16);
        $mac!(20);
        $mac!(24);
        $mac!(28);
        $mac!(32);
    };
}

/// One tile of a batched product: `T` frames against every weight column,
/// accumulated into `out`, which holds exactly those `T` frames.
///
/// `cols` is one weight column per input and each of `xt` is one frame of
/// that many inputs. The inputs are taken four at a time while there are
/// four, one load per frame, and each of the four then multiplies as a lane
/// of it; the last one to three are read one by one.
#[inline(always)]
fn tile<const OUT: usize, const T: usize, const FUSE: bool>(
    cols: &[[f32; OUT]],
    xt: [&[f32]; T],
    out: &mut [f32],
) {
    // One accumulator per frame in the tile, each indexed by output channel,
    // the dimension the lanes run over: distinct output channels are
    // independent sums.
    let mut acc = [[0.0f32; OUT]; T];
    let (w4, w1) = cols.as_chunks::<4>();
    let xq: [(&[[f32; 4]], &[f32]); T] = xt.map(|x| x.as_chunks::<4>());
    for (q, w) in w4.iter().enumerate() {
        let x: [F4; T] = std::array::from_fn(|t| F4::load(&xq[t].0[q]));
        // Each column is loaded once, then used T times. This is the whole
        // point of the tile.
        macro_rules! lane {
            ($l:literal) => {
                for (t, a) in acc.iter_mut().enumerate() {
                    macc_row_lane::<OUT, FUSE>(a, &w[$l], x[t], $l, xq[t].0[q][$l]);
                }
            };
        }
        lane!(0);
        lane!(1);
        lane!(2);
        lane!(3);
    }
    for (i, w) in w1.iter().enumerate() {
        for (t, a) in acc.iter_mut().enumerate() {
            macc_row::<OUT, FUSE>(a, w, xq[t].1[i]);
        }
    }
    // Whole frames, so the write-back loop has no bounds check per element
    // and vectorises.
    let (frames, _) = out.as_chunks_mut::<OUT>();
    for (o, a) in frames.iter_mut().zip(&acc) {
        for k in 0..OUT {
            o[k] += a[k];
        }
    }
}

/// Largest output-channel count the dynamic kernel accumulates on the stack.
///
/// Comfortably above anything a real capture uses: NAM's widest standard
/// WaveNet is 16 channels, doubled to 32 by gating.
const DYN_ACC: usize = 64;

#[inline(always)]
fn matvec_dyn_body<const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    out: &mut [f32],
    out_ch: usize,
    in_ch: usize,
) {
    debug_assert!(w.len() >= out_ch * in_ch && x.len() >= in_ch && out.len() >= out_ch);
    if out_ch <= DYN_ACC {
        let mut acc = [0.0f32; DYN_ACC];
        let acc = &mut acc[..out_ch];
        for i in 0..in_ch {
            let xi = x[i];
            let wcol = &w[i * out_ch..i * out_ch + out_ch];
            for o in 0..out_ch {
                acc[o] = step::<FUSE>(acc[o], wcol[o], xi);
            }
        }
        for o in 0..out_ch {
            out[o] += acc[o];
        }
    } else {
        for o in 0..out_ch {
            let mut acc = 0.0f32;
            for i in 0..in_ch {
                acc += w[o + i * out_ch] * x[i];
            }
            out[o] += acc;
        }
    }
}

/// Frames a batched kernel accumulates before moving to the next weight
/// column, chosen by the output width.
///
/// The tile is the accumulator, `[[f32; OUT]; TILE]`, which has to stay in
/// registers: NEON has 32 of them, four lanes each. Measured on an Apple M3
/// core by `tile_bench` below, ns per frame at block 64:
///
/// | shape | 1 | 2 | 4 | 8 |
/// |---|---|---|---|---|
/// | 4 × 4 | 1.2 | 0.9 | 0.8 | 1.5 |
/// | 8 × 8 | 2.5 | 1.8 | 1.7 | 2.5 |
/// | 8 × 16 | 4.7 | 3.3 | 2.9 | 4.1 |
/// | 12 × 12 | 4.6 | 3.5 | 3.4 | 5.7 |
/// | 16 × 16 | 7.8 | 5.9 | 5.5 | 10.5 |
/// | 24 × 12 | 8.4 | 6.6 | 7.9 | 13.9 |
/// | 32 × 16 | 14.6 | 11.2 | 18.3 | 26.1 |
///
/// Four is as fast as two or faster at every width up to 16 channels, and
/// eight is slower everywhere. Past 16 channels four frames are more
/// accumulator than there are registers and it spills, so those widths keep
/// two.
pub(crate) const fn tile_for(out_ch: usize) -> usize {
    if out_ch > 16 { 2 } else { 4 }
}

/// The dynamic kernel's tile: its accumulator is `DYN_ACC` wide whatever the
/// shape, so it keeps the conservative choice.
const DYN_TILE: usize = 2;

fused_dispatch! {
    /// Fixed-width matrix-matrix product over `n` frames, accumulating into
    /// `out`: `out[o + f * OUT] += sum_i w[o + i * OUT] * x[i + f * x_stride]`
    /// for `i` below `in_ch`.
    ///
    /// `x_stride` is the input's column stride, which is not always `in_ch` -
    /// the reference regularly hands a `Conv1x1` a `topRows()` block whose
    /// stride is the source's row count. `out` is always tightly packed at
    /// `OUT`.
    #[inline(always)]
    pub(crate) fn matmul / matmul_fma<const OUT: usize, const TILE: usize>
        (w: &[f32], x: &[f32], x_stride: usize, in_ch: usize, out: &mut [f32], n: usize)
        => matmul_body
}

/// The safe face of `matmul_fma`, for the dispatch table.
#[cfg(target_arch = "x86_64")]
fn matmul_x86_fma<const OUT: usize, const TILE: usize>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    in_ch: usize,
    out: &mut [f32],
    n: usize,
) {
    // SAFETY: `select_matmul` returns this only after `fused()` reported the
    // CPU has FMA and AVX, and nothing else names it.
    unsafe { matmul_fma::<OUT, TILE>(w, x, x_stride, in_ch, out, n) }
}

#[inline(always)]
fn matmul_body<const OUT: usize, const TILE: usize, const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    in_ch: usize,
    out: &mut [f32],
    n: usize,
) {
    debug_assert!(w.len() >= OUT * in_ch);
    debug_assert!(n == 0 || x.len() >= (n - 1) * x_stride + in_ch);
    debug_assert!(out.len() >= n * OUT);

    let (cols, _) = w.as_chunks::<OUT>();
    let cols = &cols[..in_ch];
    // A frame's inputs, sliced once here. Indexing `x[f * x_stride + i]` in
    // the tile costs a bounds check per element that the compiler cannot
    // elide, because `x_stride` is a runtime value.
    let frame = |f: usize| &x[f * x_stride..f * x_stride + in_ch];
    let mut f = 0;
    while f + TILE <= n {
        let xt: [&[f32]; TILE] = std::array::from_fn(|t| frame(f + t));
        tile::<OUT, TILE, FUSE>(cols, xt, &mut out[f * OUT..(f + TILE) * OUT]);
        f += TILE;
    }
    // Tail: fewer than TILE frames left, so one frame at a time rather than
    // a partial-tile branch through the hot loop.
    while f < n {
        tile::<OUT, 1, FUSE>(cols, [frame(f)], &mut out[f * OUT..(f + 1) * OUT]);
        f += 1;
    }
}

fused_dispatch! {
    /// Runtime-shape counterpart to [`matmul`], for shapes outside the table.
    pub(crate) fn matmul_dyn / matmul_dyn_fma
        (w: &[f32], x: &[f32], x_stride: usize, out: &mut [f32], out_ch: usize, in_ch: usize, n: usize)
        => matmul_dyn_body
}

#[inline(always)]
fn matmul_dyn_body<const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    out: &mut [f32],
    out_ch: usize,
    in_ch: usize,
    n: usize,
) {
    debug_assert!(w.len() >= out_ch * in_ch);
    debug_assert!(out.len() >= n * out_ch);

    if out_ch > DYN_ACC {
        // Too wide to tile without spilling; the per-frame kernel already has
        // a sensible shape for this case.
        for f in 0..n {
            matvec_dyn_body::<FUSE>(w, &x[f * x_stride..], &mut out[f * out_ch..], out_ch, in_ch);
        }
        return;
    }

    let mut f = 0;
    while f + DYN_TILE <= n {
        let mut acc = [[0.0f32; DYN_ACC]; DYN_TILE];
        for i in 0..in_ch {
            let wcol = &w[i * out_ch..i * out_ch + out_ch];
            for (t, a) in acc.iter_mut().enumerate() {
                let xi = x[(f + t) * x_stride + i];
                for o in 0..out_ch {
                    a[o] = step::<FUSE>(a[o], wcol[o], xi);
                }
            }
        }
        for (t, a) in acc.iter().enumerate() {
            let o = &mut out[(f + t) * out_ch..(f + t + 1) * out_ch];
            for k in 0..out_ch {
                o[k] += a[k];
            }
        }
        f += DYN_TILE;
    }
    while f < n {
        matvec_dyn_body::<FUSE>(w, &x[f * x_stride..], &mut out[f * out_ch..], out_ch, in_ch);
        f += 1;
    }
}

/// A monomorphised [`matmul`], selected once at load time:
/// `(w, x, x_stride, in_ch, out, n)`.
pub(crate) type MatMulFn = fn(&[f32], &[f32], usize, usize, &mut [f32], usize);

/// Pick the specialised batched kernel for `out_ch` outputs, if there is one.
///
/// Resolved once at load time and stored in the layer, so `process()` does no
/// shape dispatch at all.
pub(crate) fn select_matmul(out_ch: usize) -> Option<MatMulFn> {
    #[cfg(target_arch = "x86_64")]
    let fma = fused();
    macro_rules! arm {
        ($o:expr) => {
            if out_ch == $o {
                #[cfg(target_arch = "x86_64")]
                if fma {
                    return Some(matmul_x86_fma::<$o, { tile_for($o) }>);
                }
                return Some(matmul::<$o, { tile_for($o) }>);
            }
        };
    }
    for_each_width!(arm);
    None
}

/// The arguments of one batched product, so that a convolution can hand its
/// shape to [`matmul_auto`] without a positional list eight long.
pub(crate) struct Product<'a> {
    /// The column-major `(out_ch, in_ch)` weight matrix.
    pub w: &'a [f32],
    /// The input, `n` columns of `x_stride` floats, the first `in_ch` of each
    /// meaningful.
    pub x: &'a [f32],
    /// Distance in floats between consecutive input columns.
    pub x_stride: usize,
    /// Output channels.
    pub out_ch: usize,
    /// Input channels.
    pub in_ch: usize,
    /// Frames.
    pub n: usize,
}

/// Run the batched kernel chosen at load time, or the dynamic one,
/// accumulating into `out`, which is tightly packed at `out_ch`.
#[inline(always)]
pub(crate) fn matmul_auto(kernel: Option<MatMulFn>, p: Product<'_>, out: &mut [f32]) {
    match kernel {
        Some(f) => f(p.w, p.x, p.x_stride, p.in_ch, out, p.n),
        None => matmul_dyn(p.w, p.x, p.x_stride, out, p.out_ch, p.in_ch, p.n),
    }
}

fused_dispatch! {
    /// Batched depthwise accumulate: `out[c + f * ch] += w[c] * x[c + f * x_stride]`.
    ///
    /// The weight vector is only `channels` long, so there is nothing to
    /// tile: the whole of `w` stays resident across the frame loop on its own.
    pub(crate) fn depthwise_accum / depthwise_accum_fma
        (w: &[f32], x: &[f32], x_stride: usize, out: &mut [f32], channels: usize, n: usize)
        => depthwise_accum_body
}

#[inline(always)]
fn depthwise_accum_body<const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    out: &mut [f32],
    channels: usize,
    n: usize,
) {
    for f in 0..n {
        let x = &x[f * x_stride..f * x_stride + channels];
        let o = &mut out[f * channels..(f + 1) * channels];
        for c in 0..channels {
            o[c] = step::<FUSE>(o[c], w[c], x[c]);
        }
    }
}

/// The most outputs the frame-lane kernel is chosen for: one short of a
/// register, which is where channel lanes start to pay.
pub(crate) const FRAME_LANE_OUTPUTS: usize = 3;

/// The most inputs the frame-lane kernel is chosen for: sixteen, NAM's
/// standard width. Past that a narrow output's weights are a small part of
/// the convolution, and it keeps its channel lanes.
pub(crate) const FRAME_LANE_INPUTS: usize = 16;

/// Registers of frames the frame-lane kernel takes through every tap
/// together, with their sums in registers: four registers of four outputs
/// are sixteen, which leaves room for the inputs and the weights.
const FRAME_TILE: usize = 4;

/// What a frame-lane convolution reads: a planar history and the taps.
pub(crate) struct Planar<'a> {
    /// The history, one row per input: at least one.
    pub rows: &'a [&'a [f32]],
    /// The weights in registers of four, a tap after another: for each tap,
    /// for each output, a weight from every input, the last register padded.
    /// The last tap reads the block itself and each one before it
    /// `dilation` frames further back.
    pub w: &'a [[f32; 4]],
    /// Frames from one tap to the next.
    pub dilation: usize,
    /// The frame of every row at which the block starts.
    pub pos: usize,
}

/// One register of weights against the frames of up to four inputs:
/// `acc + x[0] * w[0] + x[1] * w[1] + ...`, a step at a time in that order,
/// fused by `FUSE`. Each of `x` is four consecutive frames of one input.
#[inline(always)]
pub(crate) fn dot<const FUSE: bool>(acc: F4, x: &[F4], w: F4) -> F4 {
    let mut p = acc;
    if let Some(x) = x.first() {
        p = p.macc_lane::<FUSE>(*x, w, 0);
    }
    if let Some(x) = x.get(1) {
        p = p.macc_lane::<FUSE>(*x, w, 1);
    }
    if let Some(x) = x.get(2) {
        p = p.macc_lane::<FUSE>(*x, w, 2);
    }
    if let Some(x) = x.get(3) {
        p = p.macc_lane::<FUSE>(*x, w, 3);
    }
    p
}

fused_dispatch! {
    /// Frame-lane convolution over a block: `groups` registers of frames of
    /// `O` outputs, written to `out`, one row per output.
    ///
    /// For each output and frame: every tap's chain over the inputs from
    /// zero, added to a running sum from zero in tap order. A block that is
    /// not a whole number of registers is rounded up; the frames past it
    /// are computed from whatever the history holds there.
    pub(crate) fn frames / frames_fma<const O: usize>
        (p: &Planar<'_>, out: &mut [&mut [[f32; 4]]], groups: usize) => frames_body
}

/// [`frames`] with the fusing given, for a caller that is already inside
/// its own choice of it.
#[inline(always)]
pub(crate) fn frames_body<const O: usize, const FUSE: bool>(
    p: &Planar<'_>,
    out: &mut [&mut [[f32; 4]]],
    groups: usize,
) {
    debug_assert!(!p.rows.is_empty());
    let mut g = 0;
    while g + FRAME_TILE <= groups {
        frame_tile::<O, FRAME_TILE, FUSE>(p, out, g);
        g += FRAME_TILE;
    }
    while g < groups {
        frame_tile::<O, 1, FUSE>(p, out, g);
        g += 1;
    }
}

/// `G` registers of frames through every tap, from register `g` of the
/// block: the sums stay in registers across the taps and are written to
/// `out` once.
#[inline(always)]
fn frame_tile<const O: usize, const G: usize, const FUSE: bool>(
    p: &Planar<'_>,
    out: &mut [&mut [[f32; 4]]],
    g: usize,
) {
    // Registers of weights per output and tap; the inputs before the last
    // of them, four to a register; and the inputs the last holds, one to
    // four.
    let inputs = p.rows.len();
    let per = inputs.div_ceil(4);
    let whole = 4 * (per - 1);
    let tail = (inputs - whole).min(4);
    let taps = p.w.len() / (O * per);
    let first = p.pos + 4 * g;
    let mut sum = [[F4::zero(); O]; G];
    for (k, w) in p.w.chunks_exact(O * per).enumerate() {
        let at = first - (taps - 1 - k) * p.dilation;
        // The tile's frames of one input for this tap, checked once here
        // rather than at every load.
        let input = |c: usize| -> &[[f32; 4]; G] {
            let (x, _) = p.rows[c][at..at + 4 * G].as_chunks::<4>();
            x.try_into().expect("a tile of frames")
        };
        // Every register of weights but the last, four inputs each. A
        // narrow model has none, and its chains start in the last.
        let mut acc = [[F4::zero(); O]; G];
        for q in 0..per - 1 {
            let x: [&[[f32; 4]; G]; 4] = std::array::from_fn(|l| input(4 * q + l));
            let w: [F4; O] = std::array::from_fn(|o| F4::load(&w[o * per + q]));
            for (t, acc) in acc.iter_mut().enumerate() {
                // This register's inputs, loaded once for every output.
                let x: [F4; 4] = std::array::from_fn(|l| F4::load(&x[l][t]));
                for (a, w) in acc.iter_mut().zip(&w) {
                    *a = dot::<FUSE>(*a, &x, *w);
                }
            }
        }
        // The last register, whose chains end in the tap's sum. Its weights
        // are loaded once for the tile.
        let x: [&[[f32; 4]; G]; 4] = std::array::from_fn(|l| input(whole + l.min(tail - 1)));
        let w: [F4; O] = std::array::from_fn(|o| F4::load(&w[o * per + per - 1]));
        for (t, (s, acc)) in sum.iter_mut().zip(&acc).enumerate() {
            let x: [F4; 4] = std::array::from_fn(|l| F4::load(&x[l][t]));
            for ((s, a), w) in s.iter_mut().zip(acc).zip(&w) {
                *s = s.add(dot::<FUSE>(*a, &x[..tail], *w));
            }
        }
    }
    for (row, o) in out.iter_mut().zip(0..O) {
        let row: &mut [[f32; 4]; G] = (&mut row[g..g + G]).try_into().expect("a tile of frames");
        for (t, s) in sum.iter().enumerate() {
            s[o].store(&mut row[t]);
        }
    }
}

/// A monomorphised [`frames`], selected once at load time:
/// `(history and taps, out, groups)`.
pub(crate) type FramesFn = fn(&Planar<'_>, &mut [&mut [[f32; 4]]], usize);

/// The frame-lane kernel for a convolution of this shape, if it is one
/// whose output is narrower than a register ([`FRAME_LANE_OUTPUTS`]) and
/// whose inputs the kernel takes ([`FRAME_LANE_INPUTS`]); otherwise the
/// convolution's lanes are its output channels.
pub(crate) fn select_frames(out_ch: usize, in_ch: usize) -> Option<FramesFn> {
    if !(1..=FRAME_LANE_INPUTS).contains(&in_ch) {
        return None;
    }
    match out_ch {
        1 => Some(frames::<1>),
        2 => Some(frames::<2>),
        3 => Some(frames::<3>),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::splitmix_stream as rng_stream;

    // The per-frame kernels, exposed whole as oracles for the tests below:
    // a tile of one frame, which is what the batched kernels finish their
    // tails with, and the dynamic one.
    #[inline(always)]
    fn matvec_body<const OUT: usize, const FUSE: bool>(
        w: &[f32],
        x: &[f32],
        in_ch: usize,
        out: &mut [f32],
    ) {
        let (cols, _) = w.as_chunks::<OUT>();
        tile::<OUT, 1, FUSE>(&cols[..in_ch], [&x[..in_ch]], &mut out[..OUT]);
    }

    fused_dispatch! {
        /// Fixed-width matrix-vector product, accumulating into `out`:
        /// `out[o] += sum_i w[o + i * OUT] * x[i]` for `i` below `in_ch`.
        #[inline(always)]
        fn matvec / matvec_fma<const OUT: usize>
            (w: &[f32], x: &[f32], in_ch: usize, out: &mut [f32]) => matvec_body
    }

    fused_dispatch! {
        /// Runtime-shape matrix-vector product, accumulating into `out`.
        ///
        /// Identical arithmetic to [`matvec`], same order and same
        /// association, with runtime bounds. The accumulator block is what makes the loop
        /// order possible, and the loop order is what makes it fast: `i`-outer
        /// walks one contiguous column of `w` per input channel, where `o`-outer
        /// would read `w` with stride `out_ch`, several times slower at 16×16.
        /// The `o`-outer form is kept only for shapes too wide for the stack
        /// block; it cannot accumulate straight into `out`, which already holds
        /// earlier kernel taps, without re-associating the sum.
        #[inline]
        fn matvec_dyn / matvec_dyn_fma
            (w: &[f32], x: &[f32], out: &mut [f32], out_ch: usize, in_ch: usize) => matvec_dyn_body
    }

    /// One step the way this machine does it, for the scalar oracles below.
    fn step_here(acc: f32, a: f32, b: f32) -> f32 {
        if fused() {
            macc(acc, a, b)
        } else {
            acc + a * b
        }
    }

    /// Input counts either side of every group of four, so the inputs read
    /// as a lane of a vector, the ones read singly and a kernel with only
    /// one kind are all covered.
    const INPUTS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17];

    /// The batched kernels exist only to change *when* weights are loaded
    /// and how an input reaches the multiplier, not what is computed. Tiling
    /// over frames cannot reorder an addition, since distinct frames are
    /// independent output columns, so batched and per-frame must agree bit
    /// for bit, not merely to within a tolerance.
    ///
    /// Exercised for every tabulated width and one that is not, at frame
    /// counts either side of every tile so the tiled body, the one-frame
    /// tail and the empty case are all covered, and with a column stride
    /// wider than `in_ch`, which is what a `topRows()`-style view hands a
    /// `Conv1x1`. The per-frame oracle is [`matvec_dyn`], which
    /// `specialised_and_dynamic_kernels_agree_bit_for_bit` ties to
    /// [`matvec`] and to the definition.
    #[test]
    fn batched_kernels_are_bit_identical_to_the_per_frame_ones() {
        let mut widths = vec![17usize];
        macro_rules! width {
            ($o:expr) => {
                widths.push($o);
            };
        }
        for_each_width!(width);
        for out_ch in widths {
            for in_ch in INPUTS {
                for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33] {
                    for extra_stride in [0usize, 5] {
                        let stride = in_ch + extra_stride;
                        let tag = (out_ch * 131 + in_ch * 17 + n * 7 + stride) as u64;
                        let w = rng_stream(0xBA7C4E5 ^ tag, out_ch * in_ch);
                        let x = rng_stream(0xC0FFEE ^ tag, n.max(1) * stride + in_ch);
                        // Seed both outputs identically and non-zero, so an
                        // accumulate that silently became an assign would show.
                        let seed = rng_stream(0xD00D ^ tag, n * out_ch);

                        let mut want = seed.clone();
                        for f in 0..n {
                            matvec_dyn(
                                &w,
                                &x[f * stride..],
                                &mut want[f * out_ch..],
                                out_ch,
                                in_ch,
                            );
                        }

                        let mut got = seed.clone();
                        let p = Product {
                            w: &w,
                            x: &x,
                            x_stride: stride,
                            out_ch,
                            in_ch,
                            n,
                        };
                        matmul_auto(select_matmul(out_ch), p, &mut got);
                        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "{out_ch}x{in_ch}, n={n}, stride={stride}, element {i}"
                            );
                        }

                        // And the dynamic path must match the specialised one too,
                        // for the widths that have both.
                        let mut dynamic = seed.clone();
                        matmul_dyn(&w, &x, stride, &mut dynamic, out_ch, in_ch, n);
                        for (i, (a, b)) in want.iter().zip(&dynamic).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "dyn {out_ch}x{in_ch}, n={n}, stride={stride}, element {i}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The definition, as a scalar loop, with this machine's fusion policy:
    /// what is pinned is the accumulation *order*.
    fn naive(w: &[f32], x: &[f32], out_ch: usize, in_ch: usize) -> Vec<f32> {
        (0..out_ch)
            .map(|o| {
                let mut acc = 0.0f32;
                for i in 0..in_ch {
                    acc = step_here(acc, w[o + i * out_ch], x[i]);
                }
                acc
            })
            .collect()
    }

    /// Every monomorphised kernel in the table, against the dynamic one and
    /// against the definition, at every input count in [`INPUTS`].
    #[test]
    fn specialised_and_dynamic_kernels_agree_bit_for_bit() {
        macro_rules! check {
            ($o:expr) => {{
                let o: usize = $o;
                for i in INPUTS {
                    let w = rng_stream(0xC0FFEE + o as u64 * 131 + i as u64, o * i);
                    let x = rng_stream(0xBEEF + o as u64 * 17 + i as u64, i);
                    let expected = naive(&w, &x, o, i);

                    let mut a = vec![0.0f32; o];
                    matvec::<$o>(&w, &x, i, &mut a);
                    let mut b = vec![0.0f32; o];
                    matvec_dyn(&w, &x, &mut b, o, i);

                    for k in 0..o {
                        assert_eq!(a[k].to_bits(), b[k].to_bits(), "shape {o}x{i} lane {k}");
                        assert_eq!(
                            a[k].to_bits(),
                            expected[k].to_bits(),
                            "shape {o}x{i} lane {k}"
                        );
                    }
                }
            }};
        }
        for_each_width!(check);
    }

    /// A sum that lands on a negative zero keeps its sign through the
    /// kernel: the accumulator starts from positive zero and the first step
    /// is a real multiply-accumulate onto it, in every form the row step
    /// takes.
    #[test]
    fn a_product_that_underflows_keeps_the_sign_the_definition_gives_it() {
        macro_rules! check {
            ($o:expr) => {{
                let o: usize = $o;
                for i in INPUTS {
                    let w = vec![1.0e-30f32; o * i];
                    let x = vec![-1.0e-30f32; i];
                    let expected = naive(&w, &x, o, i);
                    let mut a = vec![0.0f32; o];
                    matvec::<$o>(&w, &x, i, &mut a);
                    let p = Product {
                        w: &w,
                        x: &x,
                        x_stride: i,
                        out_ch: o,
                        in_ch: i,
                        n: 1,
                    };
                    let mut b = vec![0.0f32; o];
                    matmul_auto(select_matmul(o), p, &mut b);
                    for k in 0..o {
                        // `out` starts at positive zero and the kernel adds
                        // its sum to it, so what is written is `0 + sum`.
                        let want = 0.0 + expected[k];
                        assert_eq!(a[k].to_bits(), want.to_bits(), "{o}x{i} lane {k}");
                        assert_eq!(b[k].to_bits(), want.to_bits(), "{o}x{i} lane {k}");
                    }
                }
            }};
        }
        for_each_width!(check);
    }

    /// The frame-lane kernel against the definition: for every output and
    /// frame, each tap's chain over the inputs from zero, summed from zero
    /// in tap order. Every count of outputs it is built for, input counts
    /// either side of every register of weights, one tap and many, and
    /// blocks either side of the tile.
    #[test]
    fn the_frame_lane_kernel_matches_the_definition() {
        macro_rules! check {
            ($o:literal) => {{
                let outputs: usize = $o;
                for inputs in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 16] {
                    for (taps, dilation) in [(1usize, 1usize), (2, 3), (5, 1), (16, 1), (3, 7)] {
                        for groups in [1usize, 2, 4, 5, 9] {
                            let tag = (outputs * 1009 + inputs * 131 + taps * 17 + groups) as u64;
                            let lookback = (taps - 1) * dilation;
                            let (pos, n) = (lookback + 2, 4 * groups);
                            // Rows longer than they need be, so the stride
                            // is not the block.
                            let stride = pos + n + 3;
                            let rows = rng_stream(0xF00D ^ tag, inputs * stride);
                            let per = inputs.div_ceil(4);
                            let flat = rng_stream(0xFEED ^ tag, taps * outputs * inputs);
                            let weight =
                                |k: usize, o: usize, i: usize| flat[(k * outputs + o) * inputs + i];
                            let mut w = vec![[0.0f32; 4]; taps * outputs * per];
                            for k in 0..taps {
                                for o in 0..outputs {
                                    for i in 0..inputs {
                                        w[(k * outputs + o) * per + i / 4][i % 4] = weight(k, o, i);
                                    }
                                }
                            }
                            let row: Vec<&[f32]> = rows.chunks_exact(stride).collect();
                            let p = Planar {
                                rows: &row,
                                w: &w,
                                dilation,
                                pos,
                            };
                            let mut got = vec![vec![[9.0f32; 4]; groups]; outputs];
                            {
                                let mut out: Vec<&mut [[f32; 4]]> =
                                    got.iter_mut().map(|r| r.as_mut_slice()).collect();
                                frames::<$o>(&p, &mut out, groups);
                            }
                            for o in 0..outputs {
                                for t in 0..n {
                                    let mut want = 0.0f32;
                                    for k in 0..taps {
                                        let at = pos + t - (taps - 1 - k) * dilation;
                                        let mut acc = 0.0f32;
                                        for i in 0..inputs {
                                            acc = step_here(
                                                acc,
                                                rows[i * stride + at],
                                                weight(k, o, i),
                                            );
                                        }
                                        want += acc;
                                    }
                                    assert_eq!(
                                        got[o][t / 4][t % 4].to_bits(),
                                        want.to_bits(),
                                        "{outputs} from {inputs}, {taps} taps at {dilation}, \
                                         {groups} registers: output {o} frame {t}"
                                    );
                                }
                            }
                        }
                    }
                }
            }};
        }
        check!(1);
        check!(2);
        check!(3);
        check!(4);
    }

    #[test]
    fn the_frame_lane_kernel_is_for_narrow_outputs_and_inputs_it_can_hold() {
        for out_ch in 1..=FRAME_LANE_OUTPUTS {
            assert!(select_frames(out_ch, 1).is_some());
            assert!(select_frames(out_ch, FRAME_LANE_INPUTS).is_some());
            assert!(select_frames(out_ch, FRAME_LANE_INPUTS + 1).is_none());
            assert!(select_frames(out_ch, 0).is_none());
        }
        assert!(select_frames(FRAME_LANE_OUTPUTS + 1, 8).is_none());
        assert!(select_frames(0, 8).is_none());
    }

    #[test]
    fn depthwise_matches_the_definition() {
        let (channels, stride, n) = (5usize, 7usize, 6usize);
        let w = rng_stream(1, channels);
        let x = rng_stream(2, n * stride);
        let seed = rng_stream(3, n * channels);
        let mut got = seed.clone();
        depthwise_accum(&w, &x, stride, &mut got, channels, n);
        for f in 0..n {
            for c in 0..channels {
                let want = step_here(seed[f * channels + c], w[c], x[f * stride + c]);
                assert_eq!(
                    got[f * channels + c].to_bits(),
                    want.to_bits(),
                    "f={f} c={c}"
                );
            }
        }
    }

    #[test]
    fn every_tabulated_width_resolves() {
        macro_rules! check {
            ($o:expr) => {
                assert!(select_matmul($o).is_some(), "missing kernel {}", $o);
            };
        }
        for_each_width!(check);
        assert!(select_matmul(17).is_none());
        assert!(select_matmul(0).is_none());
    }

    /// The table is maintained by hand, so this holds it to the widths the
    /// bundled captures and the trainer's four presets instantiate; a width
    /// that falls off the table here is a width real captures pay for.
    #[test]
    fn the_bundled_captures_stay_on_the_table() {
        for o in [16usize, 12, 8, 6, 4, 3, 2, 1] {
            assert!(
                select_matmul(o).is_some(),
                "{o} channels run on the dynamic kernel"
            );
        }
    }

    /// Nanoseconds per frame at each tile size, for the widths on either
    /// side of every step of [`tile_for`]: where its table comes from.
    /// Prints; asserts nothing.
    ///
    /// ```text
    /// cargo test --release --lib tile_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measures rather than checks; run with --release --ignored"]
    fn tile_bench() {
        use std::hint::black_box;
        use std::time::Instant;
        let n = 64;
        macro_rules! width {
            ($o:literal, $i:expr) => {{
                let (o, i): (usize, usize) = ($o, $i);
                let w: Vec<f32> = (0..o * i).map(|k| (k as f32 * 0.37).sin()).collect();
                let x: Vec<f32> = (0..n * i).map(|k| (k as f32 * 0.11).cos()).collect();
                let mut out = vec![0.0f32; n * o];
                let kernels: [(usize, MatMulFn); 5] = [
                    (1, matmul::<$o, 1>),
                    (2, matmul::<$o, 2>),
                    (4, matmul::<$o, 4>),
                    (8, matmul::<$o, 8>),
                    (16, matmul::<$o, 16>),
                ];
                let mut line = format!("({o:2},{i:2})");
                for (tile, f) in kernels {
                    let f = black_box(f);
                    for _ in 0..2000 {
                        f(&w, &x, i, i, &mut out, n);
                    }
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        let reps = 20_000;
                        let t = Instant::now();
                        for _ in 0..reps {
                            f(&w, black_box(&x), i, i, &mut out, n);
                        }
                        best = best.min(t.elapsed().as_nanos() as f64 / (reps as f64 * n as f64));
                    }
                    line += &format!("  tile {tile:2}: {best:6.2}");
                }
                println!("{line}");
            }};
        }
        width!(1, 8);
        width!(2, 2);
        width!(3, 3);
        width!(4, 4);
        width!(6, 6);
        width!(8, 8);
        width!(8, 16);
        width!(12, 12);
        width!(16, 16);
        width!(16, 8);
        width!(20, 16);
        width!(24, 12);
        width!(32, 16);
    }

    /// Nanoseconds per frame for a spread of shapes at a block of 64,
    /// through the function pointer a loaded model holds, so a compiler
    /// change can be pinned to a shape. Prints; asserts nothing.
    ///
    /// It lives here rather than in `tests/benchmark.rs`, as the other
    /// crates' measurements do, because it measures `select_matmul` and the
    /// kernels behind it, which are internal: moving it would mean widening
    /// the crate's API to relocate a test.
    ///
    /// ```text
    /// cargo test --release --lib kernel_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measures rather than checks; run with --release --ignored"]
    fn kernel_bench() {
        use std::hint::black_box;
        use std::time::Instant;
        let shapes = [
            (16usize, 16usize),
            (16, 8),
            (16, 1),
            (12, 12),
            (8, 16),
            (8, 8),
            (8, 1),
            (6, 12),
            (6, 6),
            (4, 4),
            (3, 3),
            (2, 2),
            (1, 8),
            (1, 16),
            (32, 16),
        ];
        let n = 64;
        for (o, i) in shapes {
            let f = black_box(select_matmul(o).expect("tabulated"));
            let w: Vec<f32> = (0..o * i).map(|k| (k as f32 * 0.37).sin()).collect();
            let x: Vec<f32> = (0..n * i).map(|k| (k as f32 * 0.11).cos()).collect();
            let mut out = vec![0.0f32; n * o];
            for _ in 0..2000 {
                f(&w, &x, i, i, &mut out, n);
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let reps = 40_000;
                let t = Instant::now();
                for _ in 0..reps {
                    f(&w, black_box(&x), i, i, &mut out, n);
                }
                best = best.min(t.elapsed().as_nanos() as f64 / (reps as f64 * n as f64));
            }
            println!(
                "({o:2},{i:2}) {best:7.2} ns/frame  {:5.1} fma/ns  checksum {:.3}",
                (o * i) as f64 / best,
                out.iter().sum::<f32>()
            );
        }
    }
}
