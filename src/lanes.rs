//! Four `f32` lanes in one register.
//!
//! Every kernel in this crate that runs lanes side by side does it through
//! [`F4`]: a NEON register on aarch64, an SSE register on x86-64, and four
//! floats in an array everywhere else. It has the handful of operations the
//! kernels are written in and nothing more, so the three targets differ in
//! this file and nowhere else.
//!
//! # Bit-exactness
//!
//! A lane never sees its neighbours: each operation here is the scalar one
//! done four times, with IEEE 754's one correctly rounded result per lane.
//! So a kernel written in lanes computes, for each lane, exactly what the
//! scalar loop computes for that element, and the scalar fallback at the
//! bottom of this file is that loop written out.
//!
//! [`F4::macc_lane`] is the one operation without a scalar spelling. It
//! multiplies by a single lane of its third argument, which is how a kernel
//! reads four inputs with one load: the alternative, a scalar load per
//! input per frame, makes the loads the bottleneck rather than the
//! arithmetic.
//!
//! # Fusing
//!
//! `FUSE` picks between one rounding and two, as it does for
//! `kernels::macc`; see that module for when each is used. On x86-64 the
//! fused form needs FMA, which is not in the baseline, so it may only be
//! instantiated inside a function compiled for it.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use core::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_laneq_f32, vdupq_n_f32, vfmaq_f32, vfmaq_laneq_f32,
        vld1q_f32, vmulq_f32, vst1q_f32,
    };

    /// Four `f32` lanes.
    #[derive(Clone, Copy)]
    pub(crate) struct F4(float32x4_t);

    // SAFETY, for every block in this impl: the module is compiled only
    // where NEON is a baseline feature of the target, so these instructions
    // exist on every machine the crate can run on, and the two that take a
    // pointer are given a whole `[f32; 4]`.
    impl F4 {
        /// All lanes zero.
        #[inline(always)]
        pub(crate) fn zero() -> Self {
            Self(unsafe { vdupq_n_f32(0.0) })
        }

        /// All lanes `x`.
        #[inline(always)]
        pub(crate) fn splat(x: f32) -> Self {
            Self(unsafe { vdupq_n_f32(x) })
        }

        /// The four floats of `a`.
        #[inline(always)]
        pub(crate) fn load(a: &[f32; 4]) -> Self {
            Self(unsafe { vld1q_f32(a.as_ptr()) })
        }

        /// Write the lanes to `a`.
        #[inline(always)]
        pub(crate) fn store(self, a: &mut [f32; 4]) {
            unsafe { vst1q_f32(a.as_mut_ptr(), self.0) }
        }

        /// `self + b`.
        #[inline(always)]
        pub(crate) fn add(self, b: Self) -> Self {
            Self(unsafe { vaddq_f32(self.0, b.0) })
        }

        /// `self + w * x`, fused by `FUSE`.
        #[inline(always)]
        pub(crate) fn macc<const FUSE: bool>(self, w: Self, x: Self) -> Self {
            Self(unsafe {
                if FUSE {
                    vfmaq_f32(self.0, w.0, x.0)
                } else {
                    vaddq_f32(self.0, vmulq_f32(w.0, x.0))
                }
            })
        }

        /// `self + w * x[lane]`, fused by `FUSE`: every lane of `w` against
        /// the one lane of `x`. `lane` is 0 to 3 and is a constant at every
        /// call, so the match below is resolved when the kernel is compiled.
        #[inline(always)]
        pub(crate) fn macc_lane<const FUSE: bool>(self, w: Self, x: Self, lane: usize) -> Self {
            Self(unsafe {
                if FUSE {
                    match lane {
                        0 => vfmaq_laneq_f32::<0>(self.0, w.0, x.0),
                        1 => vfmaq_laneq_f32::<1>(self.0, w.0, x.0),
                        2 => vfmaq_laneq_f32::<2>(self.0, w.0, x.0),
                        _ => vfmaq_laneq_f32::<3>(self.0, w.0, x.0),
                    }
                } else {
                    let xl = match lane {
                        0 => vdupq_laneq_f32::<0>(x.0),
                        1 => vdupq_laneq_f32::<1>(x.0),
                        2 => vdupq_laneq_f32::<2>(x.0),
                        _ => vdupq_laneq_f32::<3>(x.0),
                    };
                    vaddq_f32(self.0, vmulq_f32(w.0, xl))
                }
            })
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::{
        __m128, _mm_add_ps, _mm_fmadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
        _mm_shuffle_ps, _mm_storeu_ps,
    };

    /// Four `f32` lanes.
    #[derive(Clone, Copy)]
    pub(crate) struct F4(__m128);

    // SAFETY, for every block in this impl but the fused one: SSE2 is a
    // baseline feature of x86-64, so these instructions exist on every
    // machine the crate can run on, and the two that take a pointer are
    // given a whole `[f32; 4]`.
    impl F4 {
        /// All lanes zero.
        #[inline(always)]
        pub(crate) fn zero() -> Self {
            Self(unsafe { _mm_setzero_ps() })
        }

        /// All lanes `x`.
        #[inline(always)]
        pub(crate) fn splat(x: f32) -> Self {
            Self(unsafe { _mm_set1_ps(x) })
        }

        /// The four floats of `a`.
        #[inline(always)]
        pub(crate) fn load(a: &[f32; 4]) -> Self {
            Self(unsafe { _mm_loadu_ps(a.as_ptr()) })
        }

        /// Write the lanes to `a`.
        #[inline(always)]
        pub(crate) fn store(self, a: &mut [f32; 4]) {
            unsafe { _mm_storeu_ps(a.as_mut_ptr(), self.0) }
        }

        /// `self + b`.
        #[inline(always)]
        pub(crate) fn add(self, b: Self) -> Self {
            Self(unsafe { _mm_add_ps(self.0, b.0) })
        }

        /// `self + w * x`, fused by `FUSE`.
        #[inline(always)]
        pub(crate) fn macc<const FUSE: bool>(self, w: Self, x: Self) -> Self {
            // SAFETY: the unfused form is SSE2, as above. The fused form
            // needs FMA, which is not baseline: it is only ever instantiated
            // inside a twin compiled with `target_feature(enable =
            // "avx,fma")` that is entered after `kernels::fused()` reported
            // the CPU has it. Nothing in this crate instantiates
            // `macc::<true>` anywhere else, and nothing outside it can.
            Self(unsafe {
                if FUSE {
                    _mm_fmadd_ps(w.0, x.0, self.0)
                } else {
                    _mm_add_ps(self.0, _mm_mul_ps(w.0, x.0))
                }
            })
        }

        /// `self + w * x[lane]`, fused by `FUSE`: every lane of `w` against
        /// the one lane of `x`. `lane` is 0 to 3 and is a constant at every
        /// call, so the match below is resolved when the kernel is compiled.
        #[inline(always)]
        pub(crate) fn macc_lane<const FUSE: bool>(self, w: Self, x: Self, lane: usize) -> Self {
            let xl = Self(unsafe {
                match lane {
                    0 => _mm_shuffle_ps::<0x00>(x.0, x.0),
                    1 => _mm_shuffle_ps::<0x55>(x.0, x.0),
                    2 => _mm_shuffle_ps::<0xAA>(x.0, x.0),
                    _ => _mm_shuffle_ps::<0xFF>(x.0, x.0),
                }
            });
            self.macc::<FUSE>(w, xl)
        }
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64"
)))]
mod imp {
    /// Four `f32` lanes.
    #[derive(Clone, Copy)]
    pub(crate) struct F4([f32; 4]);

    impl F4 {
        /// All lanes zero.
        #[inline(always)]
        pub(crate) fn zero() -> Self {
            Self([0.0; 4])
        }

        /// All lanes `x`.
        #[inline(always)]
        pub(crate) fn splat(x: f32) -> Self {
            Self([x; 4])
        }

        /// The four floats of `a`.
        #[inline(always)]
        pub(crate) fn load(a: &[f32; 4]) -> Self {
            Self(*a)
        }

        /// Write the lanes to `a`.
        #[inline(always)]
        pub(crate) fn store(self, a: &mut [f32; 4]) {
            *a = self.0;
        }

        /// `self + b`.
        #[inline(always)]
        pub(crate) fn add(self, b: Self) -> Self {
            Self(std::array::from_fn(|k| self.0[k] + b.0[k]))
        }

        /// `self + w * x`, fused by `FUSE`.
        #[inline(always)]
        pub(crate) fn macc<const FUSE: bool>(self, w: Self, x: Self) -> Self {
            Self(std::array::from_fn(|k| {
                if FUSE {
                    w.0[k].mul_add(x.0[k], self.0[k])
                } else {
                    self.0[k] + w.0[k] * x.0[k]
                }
            }))
        }

        /// `self + w * x[lane]`, fused by `FUSE`: every lane of `w` against
        /// the one lane of `x`. `lane` is 0 to 3.
        #[inline(always)]
        pub(crate) fn macc_lane<const FUSE: bool>(self, w: Self, x: Self, lane: usize) -> Self {
            self.macc::<FUSE>(w, Self::splat(x.0[lane]))
        }
    }
}

pub(crate) use imp::F4;

#[cfg(test)]
mod tests {
    use super::*;

    fn lanes(v: F4) -> [f32; 4] {
        let mut out = [0.0; 4];
        v.store(&mut out);
        out
    }

    /// Each operation is its scalar counterpart on every lane, to the bit,
    /// including the two things lanes could plausibly get wrong: the sign of
    /// a zero sum and which rounding a multiply-accumulate gets.
    #[test]
    fn every_operation_is_the_scalar_one_per_lane() {
        let a = [1.5f32, -0.0, 3.0e-30, -7.25];
        let w = [0.1f32, 0.0, -2.0e-20, 3.25];
        let x = [0.7f32, -5.0, 1.0e-20, 1.125];
        let (fa, fw, fx) = (F4::load(&a), F4::load(&w), F4::load(&x));
        let bits = |v: F4| lanes(v).map(f32::to_bits);

        assert_eq!(bits(F4::zero()), [0; 4]);
        assert_eq!(lanes(F4::splat(-2.5)), [-2.5; 4]);
        assert_eq!(bits(fa), a.map(f32::to_bits));
        let sum: [f32; 4] = std::array::from_fn(|k| a[k] + w[k]);
        assert_eq!(bits(fa.add(fw)), sum.map(f32::to_bits));
        let unfused: [f32; 4] = std::array::from_fn(|k| a[k] + w[k] * x[k]);
        assert_eq!(bits(fa.macc::<false>(fw, fx)), unfused.map(f32::to_bits));
        for (lane, xl) in x.iter().enumerate() {
            let by_lane: [f32; 4] = std::array::from_fn(|k| a[k] + w[k] * xl);
            assert_eq!(
                bits(fa.macc_lane::<false>(fw, fx, lane)),
                by_lane.map(f32::to_bits),
                "unfused, by lane {lane}"
            );
        }
        // The fused forms, on the targets that have them unconditionally.
        // On x86-64 they may only be reached through a kernel's own twin,
        // and the kernel tests reach them there.
        #[cfg(not(target_arch = "x86_64"))]
        {
            let fused: [f32; 4] = std::array::from_fn(|k| w[k].mul_add(x[k], a[k]));
            assert_eq!(bits(fa.macc::<true>(fw, fx)), fused.map(f32::to_bits));
            for (lane, xl) in x.iter().enumerate() {
                let by_lane: [f32; 4] = std::array::from_fn(|k| w[k].mul_add(*xl, a[k]));
                assert_eq!(
                    bits(fa.macc_lane::<true>(fw, fx, lane)),
                    by_lane.map(f32::to_bits),
                    "fused, by lane {lane}"
                );
            }
        }
    }
}
