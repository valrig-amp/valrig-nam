//! Activation functions, against values produced by the C++ reference.
//!
//! `assets/activations.f32` holds eight functions evaluated at 8001 inputs,
//! written by a program linked against `NAM/activations.h` itself. Two of
//! them, `tanh` and the `expf` inside `sigmoid` and `swish`, come from the
//! platform's libm rather than from the reference's own source, so this is
//! also the test that fails first if the crate is built somewhere whose libm
//! rounds differently. That is worth catching here, as one obvious failure,
//! rather than as a whole-model mismatch nobody can localise.
//!
//! Layout: 8001 records of 8 `f32`, in the order
//! `fast_tanh, sigmoid, swish, hardswish, softsign, tanh, hard_tanh,
//! leaky_hardtanh(-0.5, 0.75, 0.02, 0.03)`, at `x = i * 0.002` for
//! `i` in `-4000..=4000`.

use std::path::Path;
use valverig_nam::activations::*;

const FUNCTIONS: usize = 8;
const FIRST: i32 = -4000;
const LAST: i32 = 4000;

fn load() -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/activations.f32");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "macos")),
    ignore = "assets/activations.f32 was produced with arm64 macOS's libm, whose expf rounds differently from other libms; run with --ignored to see by how much"
)]
fn every_activation_matches_the_reference() {
    // Bit for bit, except the crate's own `tanh`, which is held to two ULP of
    // the reference's `tanhf` and reported.
    let expected = load();
    let count = (LAST - FIRST + 1) as usize;
    assert_eq!(
        expected.len(),
        count * FUNCTIONS,
        "assets/activations.f32 is the wrong length; regenerate it"
    );

    let names = [
        "fast_tanh",
        "sigmoid",
        "swish",
        "hardswish",
        "softsign",
        "tanh",
        "hard_tanh",
        "leaky_hardtanh",
    ];
    let mut mismatches = [0usize; FUNCTIONS];
    let mut near = 0usize;

    for (row, i) in (FIRST..=LAST).enumerate() {
        let x = i as f32 * 0.002;
        let got = [
            fast_tanh(x),
            sigmoid(x),
            swish(x),
            hardswish(x),
            softsign(x),
            tanh(x),
            hard_tanh(x),
            leaky_hardtanh(x, -0.5, 0.75, 0.02, 0.03),
        ];
        for (f, g) in got.iter().enumerate() {
            let want = expected[row * FUNCTIONS + f];
            if g.to_bits() == want.to_bits() {
                continue;
            }
            if names[f] == "tanh" && ulps(*g, want) <= 2 {
                near += 1;
            } else {
                mismatches[f] += 1;
            }
        }
    }

    let report: Vec<String> = mismatches
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .map(|(f, n)| format!("{}: {n}/{count} inputs differ", names[f]))
        .collect();
    eprintln!(
        "tanh: {near} of {count} inputs a unit or two from the reference's tanhf, the rest exact"
    );
    assert!(
        report.is_empty(),
        "activations disagree with the C++ reference:\n  {}\n\
         If only sigmoid/swish differ, this platform's libm rounds \
         differently from the one assets/activations.f32 was produced on.",
        report.join("\n  ")
    );
}

#[test]
fn the_activation_enum_agrees_with_the_free_functions() {
    // `Activation::apply` is what the model actually calls; the free functions
    // are what the reference data above pins. They must not drift apart.
    for i in -500..=500 {
        let x = i as f32 * 0.01;
        let cases: [(Activation, f32); 7] = [
            (Activation::Tanh, tanh(x)),
            (Activation::Fasttanh, fast_tanh(x)),
            (Activation::Sigmoid, sigmoid(x)),
            (Activation::SiLU, swish(x)),
            (Activation::Hardswish, hardswish(x)),
            (Activation::Softsign, softsign(x)),
            (Activation::Hardtanh, hard_tanh(x)),
        ];
        for (act, want) in cases {
            let mut buf = [x];
            act.apply(&mut buf);
            assert_eq!(buf[0].to_bits(), want.to_bits(), "{act:?} at x={x}");
        }
    }
}

/// Units in the last place between two finite floats of one sign.
fn ulps(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    let (a, b) = if a.abs() < b.abs() { (a, b) } else { (b, a) };
    // Same sign, or one of them is zero: the bit patterns are ordered.
    assert!(
        a.signum() == b.signum() || a == 0.0,
        "{a} and {b} differ in sign"
    );
    b.to_bits().abs_diff(a.to_bits())
}

/// One unit in the last place of an `f32` at `v`, as an `f64`. On a binade
/// boundary the two neighbours are not the same distance away; take the
/// nearer, so the measure below is never flattered.
fn ulp_size(v: f32) -> f64 {
    let m = v.abs() as f64;
    let b = v.abs().to_bits();
    let up = f32::from_bits(b + 1) as f64 - m;
    let down = if b == 0 {
        up
    } else {
        m - f32::from_bits(b - 1) as f64
    };
    up.min(down)
}

#[test]
fn the_crates_tanh_is_within_two_and_a_half_ulps_of_the_true_tanh() {
    let mut worst = (0.0f64, 0.0f32);
    let mut histogram = [0usize; 8];
    let mut check = |x: f32| {
        let want = (x as f64).tanh();
        let rounded = want as f32;
        let got = tanh(x);
        histogram[(rounded.to_bits().abs_diff(got.to_bits()) as usize).min(7)] += 1;
        let d = (got as f64 - want).abs() / ulp_size(rounded);
        if d > worst.0 {
            worst = (d, x);
        }
    };
    let mut x = -16.0f32;
    while x <= 16.0 {
        check(x);
        x += 1e-4;
    }
    let mut s = 0x2545F4914F6CDD1Du64;
    for _ in 0..2_000_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        check(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0);
    }
    for x in [
        0.0,
        -0.0,
        1e-8,
        -1e-8,
        3.9e-4,
        4.1e-4,
        1.0,
        -1.0,
        7.9,
        -7.9,
        8.0,
        20.0,
        -20.0,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        check(x);
    }
    assert_eq!(tanh(f32::INFINITY), 1.0);
    assert_eq!(tanh(f32::NEG_INFINITY), -1.0);
    assert!(tanh(f32::NAN).is_nan());
    report("sampled", worst, &histogram);
}

#[test]
#[ignore = "sweeps all 1.09 billion floats in (0, 10]; run with --release --ignored"]
fn the_crates_tanh_is_within_two_and_a_half_ulps_over_every_float() {
    // 10.0. Above it the input is clamped and the answer is exactly ±1.
    const TEN: u32 = 0x4120_0000;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()) as u32;
    let span = TEN / threads + 1;
    let mut worst = (0.0f64, 0.0f32);
    let mut histogram = [0usize; 8];
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut worst = (0.0f64, 0.0f32);
                    let mut histogram = [0usize; 8];
                    for b in (t * span)..((t + 1) * span).min(TEN + 1) {
                        let x = f32::from_bits(b);
                        let want = (x as f64).tanh();
                        let rounded = want as f32;
                        let got = tanh(x);
                        let apart = rounded.to_bits().abs_diff(got.to_bits()) as usize;
                        histogram[apart.min(7)] += 1;
                        let d = (got as f64 - want).abs() / ulp_size(rounded);
                        if d > worst.0 {
                            worst = (d, x);
                        }
                    }
                    (worst, histogram)
                })
            })
            .collect();
        for worker in workers {
            let (w, h) = worker.join().unwrap();
            if w.0 > worst.0 {
                worst = w;
            }
            for (total, part) in histogram.iter_mut().zip(h) {
                *total += part;
            }
        }
    });
    report("exhaustive", worst, &histogram);
}

/// The shared verdict: how far the crate's `tanh` is from the true value,
/// and how many floats lie between it and the correctly rounded one.
fn report(which: &str, worst: (f64, f32), histogram: &[usize; 8]) {
    let total: usize = histogram.iter().sum();
    eprintln!(
        "tanh, {which}: floats between the answer and the correctly rounded \
         one, 0..7+: {histogram:?} of {total}, {:.2}% of them the correctly \
         rounded float itself; worst {:.4} ULP from the true value at x = {}",
        100.0 * histogram[0] as f64 / total as f64,
        worst.0,
        worst.1
    );
    assert!(
        worst.0 <= 2.5,
        "tanh is {:.4} ULP from the true value at x = {}, which is more than \
         two floats from the correctly rounded one",
        worst.0,
        worst.1
    );
    let close: usize = histogram[..2].iter().sum();
    assert!(
        close * 100 >= total * 99,
        "fewer than 99% within one float of correctly rounded: {histogram:?}"
    );
}
