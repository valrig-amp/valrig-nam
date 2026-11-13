//! Activation functions, transliterated from `NAM/activations.h`.
//!
//! Every function here reproduces the reference expression term for term,
//! in the same order and the same precision. Floating-point addition is not
//! associative, so the association implied by the C++ parse is part of the
//! contract, not an implementation detail.
//!
//! `Sigmoid` and `SiLU` route through the platform libm (`expf`), exactly
//! as the reference does, and so does the LSTM's cell. Bit-exactness against
//! the C++ reference therefore holds for those when both are linked against
//! the same libm, which is true by construction when comparing builds on
//! one machine, and is the condition under which the reference vectors in
//! `assets/` were produced.
//!
//! `Tanh`, the activation NAM's standard architecture applies to every
//! channel of every layer, is this crate's own: [`tanh`], plain
//! single-precision arithmetic evaluated identically on every machine, a
//! block of values at a time. At every float there is, it is within two and
//! a half units in the last place of the true `tanh` — so never more than
//! two floats from the correctly rounded one, and the correctly rounded
//! float itself on 92% of them, as `tests/activations.rs` measures — and
//! the recorded bounds hold the whole-model consequence to the reference.
//!
//! ```
//! use valrig_nam::activations::{Activation, tanh};
//!
//! let mut block = [0.5f32, -2.0, 0.0];
//! Activation::Tanh.apply(&mut block);
//! assert_eq!(block[0], tanh(0.5));
//! assert!((block[0] - 0.5f32.tanh()).abs() < 1e-6);
//! ```

use crate::error::{Error, Result};

/// `relu`: `max(x, 0)`.
#[inline]
pub(crate) fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// `sigmoid`: `1 / (1 + expf(-x))`, through the platform libm.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `hard_tanh`: clamp to `[-1, 1]`.
///
/// Written as the reference's pair of comparisons rather than `f32::clamp`,
/// which panics on a NaN bound and orders its comparisons differently.
#[allow(clippy::manual_clamp)]
#[inline]
pub fn hard_tanh(x: f32) -> f32 {
    let t = if x < -1.0 { -1.0 } else { x };
    if t > 1.0 { 1.0 } else { t }
}

/// `leaky_hardtanh`: linear between `min_val` and `max_val`, with slopes
/// `min_slope` below and `max_slope` above.
#[inline]
pub fn leaky_hardtanh(x: f32, min_val: f32, max_val: f32, min_slope: f32, max_slope: f32) -> f32 {
    if x < min_val {
        (x - min_val) * min_slope + min_val
    } else if x > max_val {
        (x - max_val) * max_slope + max_val
    } else {
        x
    }
}

/// `fast_tanh`: the reference's rational approximation of `tanh`.
///
/// The coefficients carry more decimal digits than an `f32` can distinguish.
/// That is deliberate: they are transcribed character for character from
/// `NAM/activations.h`, so that both compilers round the same decimal string
/// to the same `f32`. Shortening them is exactly the kind of tidy-up that
/// would silently move the last bit.
///
/// Its worst absolute error against `tanh` is 4.4e-4, at x = -0.582. A file
/// selects it as the activation named `Fasttanh`; nothing else in this crate
/// substitutes it for `tanh`.
#[allow(clippy::excessive_precision)]
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;

    x * (2.455_507_507_029_56_f32
        + 2.455_507_507_029_56_f32 * ax
        + (0.893_229_853_513_558_f32 + 0.821_226_666_969_744_f32 * ax) * x2)
        / (2.445_066_346_522_99_f32
            + (2.445_066_346_522_99_f32 + x2) * (x + 0.814_642_734_961_073_f32 * x * ax).abs())
}

/// This crate's `tanh`: `expm1(2|x|) / (expm1(2|x|) + 2)` with the sign put
/// back, in plain single-precision multiplies, adds and one division, the
/// same on every machine, so a model's output does not depend on which libm
/// it was linked against. Within two and a half units in the last place of
/// the true `tanh` at every float in ±10, which puts it never more than two
/// floats from the correctly rounded one and on the correctly rounded float
/// itself 92% of the time, as `tests/activations.rs` measures over the whole
/// range; beyond ±10 the input is clamped and ±1 is what `tanh` correctly
/// rounds to anyway. Nothing is fused, so the rounding is the same with and
/// without an FMA unit. NaN in, NaN out; ±∞ in, ±1 out.
///
/// `expm1` is Cephes' `expf` without its leading 1: the exponent split off
/// by rounding `2|x|·log2(e)` to an integer (done by adding and subtracting
/// 1.5·2²³, which is exact and needs no rounding instruction), the remainder
/// brought back in two constants (Cody and Waite), a polynomial of degree
/// seven on what is left, and the exponent put into the float's bits.
/// Keeping the 1 out of the polynomial is what preserves small inputs: for
/// them `expm1(2x) ≈ 2x` and the quotient is `x` to the last bit, where
/// `1 - 2 / (e^{2x} + 1)` would have cancelled it away.
///
/// Where it matters: NAM's standard architecture applies this at every one
/// of its twenty layers, and libm's `tanhf` there is two thirds of that
/// model's time.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let (n, r) = tanh_reduce(x);
    tanh_finish(x, n, tanh_expm1(r))
}

/// The first third of [`tanh`]: `2|x|` split into `n` whole powers of two
/// and a remainder `r`, so that `e^{2|x|} = 2^n · e^r`.
#[inline(always)]
fn tanh_reduce(x: f32) -> (f32, f32) {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const ROUNDER: f32 = 12_582_912.0;
    let ax = x.abs();
    let ax = if ax > 10.0 { 10.0 } else { ax };
    let t = ax + ax;
    let n = (t * LOG2E + ROUNDER) - ROUNDER;
    let r = t - n * LN2_HI;
    let r = r - n * LN2_LO;
    (n, r)
}

/// The middle third of [`tanh`]: `e^r - 1` for the remainder.
#[inline(always)]
fn tanh_expm1(r: f32) -> f32 {
    let mut p = 1.987_569_2e-4;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 0.5;
    p * (r * r) + r
}

/// The last third of [`tanh`]: the power of two put back, the quotient, and
/// the sign of `x`.
#[inline(always)]
fn tanh_finish(x: f32, n: f32, q: f32) -> f32 {
    let scale = f32::from_bits(((n as i32 + 127) << 23) as u32);
    let e = q * scale + (scale - 1.0);
    (e / (e + 2.0)).copysign(x)
}

/// Elements [`tanh_lanes`] takes through one third of [`tanh`] before
/// starting them on the next.
const STAGE: usize = 32;

/// [`tanh`] over a slice.
///
/// One element's `tanh` is some thirty operations, each waiting on the one
/// before, and a core only looks so far ahead for work it can start: run
/// element after element, it holds a handful of those chains at once and
/// its multipliers sit idle between their links. So a block of [`STAGE`]
/// elements goes through the reduction, then the polynomial, then the rest,
/// and within a stage no chain is long. The operations and their order are
/// [`tanh`]'s own, element by element, so the result is the same to the
/// bit. On an Apple M3 this measures 0.70 ns an element, against 0.82 for
/// the whole function sixteen elements at a time and 2.0 for libm.
///
/// What is left over goes sixteen at a time, then singly.
pub(crate) fn tanh_lanes(data: &mut [f32]) {
    let (blocks, rest) = data.as_chunks_mut::<STAGE>();
    for b in blocks {
        let mut n = [0.0f32; STAGE];
        let mut r = [0.0f32; STAGE];
        for i in 0..STAGE {
            (n[i], r[i]) = tanh_reduce(b[i]);
        }
        let mut q = [0.0f32; STAGE];
        for i in 0..STAGE {
            q[i] = tanh_expm1(r[i]);
        }
        for i in 0..STAGE {
            b[i] = tanh_finish(b[i], n[i], q[i]);
        }
    }
    let (chunks, rest) = rest.as_chunks_mut::<16>();
    for c in chunks {
        let mut r = [0.0f32; 16];
        for i in 0..16 {
            r[i] = tanh(c[i]);
        }
        *c = r;
    }
    for v in rest {
        *v = tanh(*v);
    }
}

/// `leaky_relu`: `x` above zero, `negative_slope * x` at or below.
#[inline]
pub(crate) fn leaky_relu(x: f32, negative_slope: f32) -> f32 {
    if x > 0.0 { x } else { negative_slope * x }
}

/// `swish` / SiLU: `x * sigmoid(x)`.
#[inline]
pub fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `hardswish`: `x * clamp(x + 3, 0, 6) * (1/6)`.
///
/// The comparisons are the reference's, not `f32::clamp`; see [`hard_tanh`].
#[allow(clippy::manual_clamp)]
#[inline]
pub fn hardswish(x: f32) -> f32 {
    let t = x + 3.0;
    let clamped = if t < 0.0 {
        0.0
    } else if t > 6.0 {
        6.0
    } else {
        t
    };
    x * clamped * (1.0f32 / 6.0f32)
}

/// `softsign`: `x / (1 + |x|)`.
#[inline]
pub fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// A configured activation function.
///
/// This is both the parsed config and the runnable function; the reference
/// splits these across `ActivationConfig` and an `Activation` subclass, but
/// the parameters are immutable after load so one type suffices. A file
/// names one either as a bare string (`"Tanh"`) or as an object with a
/// `type` and optional parameters; [`crate::format::activation_from_json`]
/// reads both spellings.
#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    /// This crate's [`tanh`].
    Tanh,
    /// Clamp to `[-1, 1]`.
    Hardtanh,
    /// The rational approximation in [`fast_tanh`].
    Fasttanh,
    /// `max(x, 0)`.
    ReLU,
    /// Leaky ReLU with one slope for all channels.
    LeakyReLU {
        /// Slope applied where `x <= 0`. The file's default is 0.01.
        negative_slope: f32,
    },
    /// Parametric ReLU with a per-channel slope table.
    ///
    /// Matches the reference's flat-buffer path, which indexes the slope
    /// table by `position % slopes.len()` over a column-major
    /// `(channels, frames)` buffer, which is to say by channel.
    PReLU {
        /// One slope per channel. Never empty.
        negative_slopes: Vec<f32>,
    },
    /// Logistic sigmoid.
    Sigmoid,
    /// SiLU / Swish.
    SiLU,
    /// Hard swish.
    Hardswish,
    /// Leaky hard tanh; see [`leaky_hardtanh`].
    LeakyHardtanh {
        /// Lower knee. The file's default is -1.
        min_val: f32,
        /// Upper knee. The file's default is 1.
        max_val: f32,
        /// Slope below `min_val`. The file's default is 0.01.
        min_slope: f32,
        /// Slope above `max_val`. The file's default is 0.01.
        max_slope: f32,
    },
    /// `x / (1 + |x|)`.
    Softsign,
    /// Pass-through. Not reachable from a `.nam` file; used internally where
    /// the reference wires up `ActivationIdentity`.
    Identity,
}

impl Activation {
    /// Parse the bare-string form (`"Tanh"`), as the reference's `type_map`
    /// does, with the parameter defaults the reference's singletons carry.
    ///
    /// Both `LeakyHardtanh` and `LeakyHardTanh` spellings are accepted,
    /// matching the reference's duplicate map entry. Fails with
    /// [`Error::Config`] on any other name.
    pub(crate) fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "Tanh" => Activation::Tanh,
            "Hardtanh" => Activation::Hardtanh,
            "Fasttanh" => Activation::Fasttanh,
            "ReLU" => Activation::ReLU,
            // The reference's bare-string path resolves to the default-constructed
            // singletons, whose slopes are 0.01.
            "LeakyReLU" => Activation::LeakyReLU {
                negative_slope: 0.01,
            },
            "PReLU" => Activation::PReLU {
                negative_slopes: vec![0.01],
            },
            "Sigmoid" => Activation::Sigmoid,
            "SiLU" => Activation::SiLU,
            "Hardswish" => Activation::Hardswish,
            "LeakyHardtanh" | "LeakyHardTanh" => Activation::LeakyHardtanh {
                min_val: -1.0,
                max_val: 1.0,
                min_slope: 0.01,
                max_slope: 0.01,
            },
            "Softsign" => Activation::Softsign,
            other => return Err(Error::Config(format!("Unknown activation type: {other}"))),
        })
    }

    /// Apply in place to a column-major `(channels, frames)` buffer.
    ///
    /// `data.len()` must be `channels * frames`; `channels` is only consulted
    /// by [`Activation::PReLU`], and even there the reference indexes by
    /// `pos % slopes.len()`, which this reproduces. Allocation-free.
    #[inline]
    pub fn apply(&self, data: &mut [f32]) {
        match self {
            Activation::Tanh => tanh_lanes(data),
            Activation::Hardtanh => map(data, hard_tanh),
            Activation::Fasttanh => map(data, fast_tanh),
            Activation::ReLU => map(data, relu),
            Activation::LeakyReLU { negative_slope } => {
                map(data, |x| leaky_relu(x, *negative_slope))
            }
            Activation::PReLU { negative_slopes } => {
                let n = negative_slopes.len();
                debug_assert!(n > 0, "PReLU with no slopes");
                for (pos, v) in data.iter_mut().enumerate() {
                    *v = leaky_relu(*v, negative_slopes[pos % n]);
                }
            }
            Activation::Sigmoid => map(data, sigmoid),
            Activation::SiLU => map(data, swish),
            Activation::Hardswish => map(data, hardswish),
            Activation::LeakyHardtanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
            } => map(data, |x| {
                leaky_hardtanh(x, *min_val, *max_val, *min_slope, *max_slope)
            }),
            Activation::Softsign => map(data, softsign),
            Activation::Identity => {}
        }
    }
}

/// Apply `f` to every element in place.
///
/// Generic rather than a function pointer, so that each activation's body is
/// inlined into its own copy of the loop.
#[inline(always)]
fn map(data: &mut [f32], f: impl Fn(f32) -> f32) {
    for v in data.iter_mut() {
        *v = f(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_tanh_tracks_tanh_within_its_documented_error() {
        let mut worst = 0.0f32;
        for i in -400..=400 {
            let x = i as f32 * 0.02;
            worst = worst.max((fast_tanh(x) - x.tanh()).abs());
        }
        // The reference's own approximation peaks at 4.361e-4 (at x = -0.582);
        // this bounds our reproduction of it, not the approximation's quality.
        assert!(
            worst < 5e-4,
            "worst absolute error over [-8, 8] was {worst}"
        );
    }

    #[test]
    fn fast_tanh_matches_pinned_reference_bits() {
        // Produced by the C++ reference's fast_tanh, built the way
        // `tools/regen-assets.sh` builds it. Pinned so that a change to the
        // f32/f64 promotion is caught here rather than in a whole-model diff.
        let cases: [(f32, u32); 5] = [
            (0.0, 0x0000_0000),
            (1.0, 0x3F43_082D),
            (-1.0, 0xBF43_082D),
            (0.5, 0x3EEC_687A),
            (3.25, 0x3F7F_4608),
        ];
        for (x, bits) in cases {
            assert_eq!(
                fast_tanh(x).to_bits(),
                bits,
                "fast_tanh({x}) = {} (0x{:08X})",
                fast_tanh(x),
                fast_tanh(x).to_bits()
            );
        }
    }

    #[test]
    fn scalar_activations_match_pinned_reference_bits() {
        // Also produced by the C++ reference. These pin the libm-backed paths
        // (tanhf / expf) as well as the hand-written ones, so a libm
        // difference shows up as a targeted failure instead of a whole-model
        // mismatch.
        let cases: [(f32, [u32; 5]); 2] = [
            (
                0.5,
                [
                    0x3EEC_9A9F,
                    0x3F1F_597F,
                    0x3E9F_597F,
                    0x3E95_5556,
                    0x3EAA_AAAB,
                ],
            ),
            (
                -2.25,
                [
                    0xBF7A_5FEB,
                    0x3DC3_4695,
                    0xBE5B_AF68,
                    0xBE90_0000,
                    0xBF31_3B14,
                ],
            ),
        ];
        for (x, [t, s, w, h, ss]) in cases {
            // Without `black_box` LLVM folds these calls at compile time through
            // its double-precision folder, and the test would pin the folder
            // rather than the libm the model actually calls.
            let x = std::hint::black_box(x);
            assert_eq!(x.tanh().to_bits(), t, "tanh({x})");
            assert_eq!(sigmoid(x).to_bits(), s, "sigmoid({x})");
            assert_eq!(swish(x).to_bits(), w, "swish({x})");
            assert_eq!(hardswish(x).to_bits(), h, "hardswish({x})");
            assert_eq!(softsign(x).to_bits(), ss, "softsign({x})");
        }
    }

    #[test]
    fn prelu_indexes_by_channel() {
        let a = Activation::PReLU {
            negative_slopes: vec![0.5, 0.25],
        };
        // Column-major (2 channels, 2 frames): [c0f0, c1f0, c0f1, c1f1]
        let mut d = [-1.0f32, -1.0, -2.0, -2.0];
        a.apply(&mut d);
        assert_eq!(d, [-0.5, -0.25, -1.0, -0.5]);
    }
}
