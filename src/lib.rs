//! Pure-Rust Neural Amp Modeler: load and run `.nam` captures, with no C++
//! and no linear-algebra dependency.
//!
//! Inference only. The crate loads a capture and runs it in real time;
//! making captures is not in scope. Every architecture the C++ reference
//! registers is supported except the two pre-WaveNet ones, `Linear` and
//! `ConvNet`, which are recognised and refused.
//!
//! # Running a capture
//!
//! ```
//! use valrig_nam::loader::Model;
//!
//! # let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/models/wavenet.nam");
//! let mut model = Model::from_file(path)?;
//!
//! // Allocates and settles the model on silence. Do this before the audio
//! // thread starts, never from inside it.
//! model.reset(48_000.0, 64);
//!
//! let dry = [0.1f32; 64];
//! let mut wet = [0.0f32; 64];
//! model.process_mono(&dry, &mut wet); // allocation-free from here on
//! # assert!(wet.iter().all(|x| x.is_finite()));
//! # Ok::<(), valrig_nam::error::Error>(())
//! ```
//!
//! A capture is only correct at the sample rate it was trained at; nothing
//! here resamples. Check [`loader::Model::expected_sample_rate`] and tell the
//! user when it disagrees with the stream.
//!
//! # What is public
//!
//! [`loader::Model`] is the whole interface a host needs. [`engine::Engine`]
//! is the trait behind it, for a host that would rather hold the trait
//! object; [`buffer::Buf`] is the column-major buffer that trait's nested
//! entry point takes. [`format::load_file`] parses a `.nam` document into
//! typed configuration without building anything, for tooling that wants to
//! say what a file holds; [`activations`] is the set of activation functions
//! that configuration names. Every error is [`error::Error`].
//!
//! # Untrusted files
//!
//! A malformed `.nam` returns an error; it never aborts the process. Every
//! count in a file is capped at [`format::MAX_COUNT`], every weight matrix
//! is sized against the file's own weight array before it is allocated, a
//! WaveNet's convolution histories are capped in total at
//! [`format::MAX_HISTORY_FLOATS`], and a stated sample rate must lie in
//! [`format::MIN_SAMPLE_RATE`] to [`format::MAX_SAMPLE_RATE`]. Real captures sit orders of
//! magnitude inside every one of those.
//!
//! # Agreement with the reference
//!
//! Output agrees with `NeuralAmpModelerCore` to a bound recorded per test
//! case in `assets/expectations.txt`, and that bound is inside the spread
//! the reference has against itself across its own supported build
//! configurations. `README.md` states the contract and says how
//! the vectors are produced.
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod activations;
pub mod buffer;
pub mod engine;
pub mod error;
pub mod format;
pub mod loader;

mod a2_fast;
mod container;
mod conv;
mod film;
mod fused;
mod gating;
mod history;
mod kernels;
mod lanes;
mod lstm;
#[cfg(test)]
mod testutil;
mod wavenet;
mod weights;

/// Block size a model is sized to when `prewarm()` runs before any size has
/// been set: the reference's `NAM_DEFAULT_MAX_BUFFER_SIZE`.
pub const DEFAULT_MAX_BUFFER_SIZE: usize = 4096;
