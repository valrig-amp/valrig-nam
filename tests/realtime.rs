//! Real-time safety and streaming invariants.
//!
//! A model runs directly in a device callback, so its process path must not
//! allocate, free, lock or block. The reference enforces the same property
//! with its own allocation tracker (`tools/test/allocation_tracking.cpp`);
//! this is the Rust equivalent, over every bundled model, including the A2
//! fast path, the containers and the model with a nested conditioning model.

mod common;

use common::{lcg_signal, run_schedule};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;
use valverig_nam::loader::Model;

// Per-thread, because the test harness runs tests concurrently and a global
// counter would attribute one test's setup allocations to another's measured
// window. Both cells are const-initialised so that touching them from inside
// the allocator cannot itself allocate.
thread_local! {
    static WATCHING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Counts this thread's allocations while it is watching.
struct Counting;

impl Counting {
    fn note() {
        // `try_with` so that an allocation during thread teardown, after the
        // TLS block is destroyed, does not panic inside the allocator.
        let _ = WATCHING.try_with(|w| {
            if w.get() {
                let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
            }
        });
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `f`, returning how many times this thread allocated.
fn allocations_during<F: FnOnce()>(f: F) -> usize {
    ALLOCS.with(|a| a.set(0));
    WATCHING.with(|w| w.set(true));
    f();
    WATCHING.with(|w| w.set(false));
    ALLOCS.with(Cell::get)
}

fn model_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/models")
        .join(format!("{name}.nam"))
}

/// Every bundled model, by file stem.
fn models() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "nam"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names.len(), 12);
    names
}

#[test]
fn process_does_not_allocate() {
    for name in models() {
        let mut model = Model::from_file(model_path(&name)).expect(&name);
        model.reset(48000.0, 64);

        let input = vec![0.1f32; 64];
        let mut output = vec![0.0f32; 64];

        // One untracked call first: nothing in the process path should be
        // lazily initialised, but if something were, blaming the first call
        // would hide the steady-state result this test is about.
        model.process_mono(&input, &mut output);

        let n = allocations_during(|| {
            for _ in 0..200 {
                model.process_mono(&input, &mut output);
            }
        });
        assert_eq!(
            n, 0,
            "{name}: process() allocated {n} time(s) over 200 blocks"
        );
    }
}

#[test]
fn process_does_not_allocate_for_ragged_block_sizes() {
    // A host is free to hand over any block size up to the maximum, including
    // one frame. Each distinct size must still be allocation-free.
    let mut model = Model::from_file(model_path("wavenet_a1_standard")).unwrap();
    model.reset(48000.0, 128);
    let input = vec![0.05f32; 128];
    let mut output = vec![0.0f32; 128];

    for &n in &[1usize, 7, 128, 3, 64, 17] {
        model.process_mono(&input[..n], &mut output[..n]);
    }

    let n = allocations_during(|| {
        for round in 0..120 {
            let b = [1usize, 7, 128, 3, 64, 17][round % 6];
            model.process_mono(&input[..b], &mut output[..b]);
        }
    });
    assert_eq!(
        n, 0,
        "process() allocated {n} time(s) across ragged block sizes"
    );
}

/// Splitting a stream into blocks must not change a single bit of the output.
///
/// This is the property that makes a model safe to run in a host whose block
/// size varies. It holds only for a fixed maximum block size, because
/// `prewarm()` settles the model by pushing `max_buffer`-sized blocks of
/// silence through it, so a different maximum leaves a different (equally
/// valid) starting state. The reference behaves the same way.
#[test]
fn output_is_independent_of_how_frames_are_grouped() {
    const N: usize = 1500;
    const MAX_BLOCK: usize = 128;

    let signal = lcg_signal(N, 0x5EED, 0.4);

    for name in models() {
        let run = |schedule: &[usize]| -> Vec<f32> {
            let mut model = Model::from_file(model_path(&name)).unwrap();
            model.reset(48000.0, MAX_BLOCK);
            run_schedule(&mut model, &signal, schedule)
        };

        let reference = run(&[MAX_BLOCK]);
        for schedule in [
            vec![1usize],
            vec![7],
            vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128],
            vec![128, 1, 128, 2],
            vec![63, 65],
        ] {
            let got = run(&schedule);
            for (i, (a, b)) in reference.iter().zip(&got).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name}: schedule {schedule:?} differs at sample {i}"
                );
            }
        }
    }
}
