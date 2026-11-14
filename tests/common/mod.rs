//! Helpers shared by the integration tests. Each test binary uses a subset,
//! so the ones it does not use are not dead code.
#![allow(dead_code)]

use std::path::Path;
use valverig_nam::loader::Model;

/// A little-endian `f64` blob, as `tools/gen_vectors.cpp` writes them.
pub fn read_f64(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 8,
        0,
        "{} is not a whole number of f64",
        path.display()
    );
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

/// A deterministic test signal: a 64-bit LCG, its top 24 bits scaled to
/// `[-amp, amp)`, so every value is exactly representable in `f32`.
pub fn lcg_signal(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as u32 as f32 / 8_388_608.0 - 1.0) * amp
        })
        .collect()
}

/// Run a mono model over `signal` in blocks cycling through `schedule`,
/// returning its output.
pub fn run_schedule(model: &mut Model, signal: &[f32], schedule: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; signal.len()];
    let mut pos = 0;
    let mut i = 0;
    while pos < signal.len() {
        let n = schedule[i % schedule.len()].min(signal.len() - pos).max(1);
        i += 1;
        model.process_mono(&signal[pos..pos + n], &mut out[pos..pos + n]);
        pos += n;
    }
    out
}
