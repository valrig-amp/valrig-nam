//! What a capture costs, against the time it has: every bundled model, a
//! block at a time. Ignored by default, because it measures rather than
//! checks:
//!
//! ```text
//! cargo test --release --test benchmark -- --ignored --nocapture
//! ```
//!
//! A debug build measures the debug build and tells you nothing about what
//! ships; always run this with `--release`.

mod common;

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use common::lcg_signal;
use valrig_nam::loader::Model;

/// Steady-state throughput of each bundled capture at blocks of 64 and 512
/// frames: nanoseconds a frame, and multiples of real time at 48 kHz. The
/// two containers are measured at each size they offer. Loading and the
/// reset are outside the clock.
///
/// It measures rather than asserts, so it is not part of a normal run:
///
/// ```text
/// cargo test --release --test benchmark -- --ignored --nocapture
/// ```
#[test]
#[ignore = "measures rather than checks; run with --release --ignored"]
fn throughput() {
    if cfg!(debug_assertions) {
        eprintln!("warning: built without optimisations; these numbers mean nothing.");
    }
    let sample_rate = 48_000.0;
    let seconds = 2.0;
    let frames = (sample_rate * seconds) as usize;
    let input = lcg_signal(frames, 0xC0FFEE, 0.4);
    let mut output = vec![0.0f32; frames];

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "nam"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();

    println!(
        "{:<28} {:>5} {:>6} {:>10} {:>12}",
        "model", "size", "block", "ns/frame", "x real time"
    );
    for name in names {
        let path = dir.join(format!("{name}.nam"));
        let probe = Model::from_file(&path).expect(&name);
        if probe.in_channels() != 1 || probe.out_channels() != 1 {
            continue;
        }
        // A container's sizes: the smallest, each breakpoint, the largest.
        let breakpoints = probe.slimmable_size_breakpoints();
        let sizes: Vec<Option<f64>> = if breakpoints.is_empty() {
            vec![None]
        } else {
            let mut sizes = vec![Some(0.0)];
            sizes.extend(breakpoints.iter().map(|b| Some(*b)));
            sizes
        };
        for size in sizes {
            for block in [64usize, 512] {
                let mut model = Model::from_file(&path).expect(&name);
                if let Some(size) = size {
                    model.set_slimmable_size(size).expect(&name);
                }
                model.reset(sample_rate, block);
                let mut sweep = || {
                    for pos in (0..frames).step_by(block) {
                        let n = block.min(frames - pos);
                        model.process_mono(
                            black_box(&input[pos..pos + n]),
                            black_box(&mut output[pos..pos + n]),
                        );
                    }
                };
                sweep();
                let mut times = Vec::with_capacity(5);
                for _ in 0..5 {
                    let start = Instant::now();
                    sweep();
                    times.push(start.elapsed().as_secs_f64());
                }
                times.sort_by(f64::total_cmp);
                let median = times[times.len() / 2];
                println!(
                    "{name:<28} {:>5} {block:>6} {:>10.1} {:>12.1}",
                    size.map_or("-".to_string(), |s| format!("{s:.2}")),
                    median * 1e9 / frames as f64,
                    seconds / median
                );
            }
        }
    }
}
