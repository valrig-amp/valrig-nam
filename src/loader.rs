//! [`Model`]: a loaded `.nam` capture, ready to run.
//!
//! [`crate::format`] owns everything about the file itself: the JSON schema,
//! the defaults, the version range, the validation. `engine::build` turns
//! the parsed result into an engine. This
//! module is the surface a host holds: the engine behind an [`Engine`]
//! trait object, plus the three file-level facts a host reads.

use crate::buffer::Buf;
use crate::engine::{self, Engine};
use crate::error::Result;
use crate::format::{self, NamFile};
use serde_json::Value;
use std::path::Path;

pub use crate::format::Metadata;

/// A loaded model, ready to run.
///
/// The architecture is behind an [`Engine`], the way the reference holds a
/// `std::unique_ptr<nam::DSP>`: which one it is stops mattering the moment
/// the file has been read. Every method that the trait has is forwarded
/// here, so a host never needs the trait in scope.
///
/// ```
/// use valverig_nam::loader::Model;
///
/// # let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/models/wavenet.nam");
/// let mut model = Model::from_file(path)?;
/// if let Some(rate) = model.expected_sample_rate {
///     assert_eq!(rate, 48_000.0);
/// }
///
/// // Allocates and settles the model. Before the audio thread starts.
/// model.reset(48_000.0, 64);
///
/// // Allocation-free from here on, for any block of up to 64 frames.
/// let dry = [0.1f32; 64];
/// let mut wet = [0.0f32; 64];
/// model.process_mono(&dry, &mut wet);
/// # assert!(wet.iter().all(|x| x.is_finite()));
/// # Ok::<(), valverig_nam::error::Error>(())
/// ```
#[derive(Debug)]
pub struct Model {
    inner: Box<dyn Engine>,
    /// The sample rate the model was trained at, in Hz, or `None` when the
    /// file, an old one, does not say.
    ///
    /// A capture is only correct at this rate and nothing here resamples;
    /// compare it against the stream before playing and tell the user.
    pub expected_sample_rate: Option<f64>,
    /// The file's `metadata` block.
    pub metadata: Metadata,
    /// The file's `version` string, verbatim.
    pub version: String,
}

impl Model {
    /// Load from a `.nam` file on disk.
    ///
    /// Fails with [`crate::error::Error::Io`] when the file cannot be read
    /// and otherwise as [`Model::build`].
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::build(format::load_file(path)?)
    }

    /// Load from `.nam` JSON text.
    pub fn from_json(text: &str) -> Result<Self> {
        Self::build(format::parse_json(text)?)
    }

    /// Load from an already-decoded JSON value.
    pub fn from_value(v: &Value) -> Result<Self> {
        Self::build(format::parse_value(v)?)
    }

    /// Build a runnable model from a parsed file.
    ///
    /// Fails with [`crate::error::Error::WeightCount`] when the weight array
    /// is not exactly as long as the architecture consumes,
    /// [`crate::error::Error::Config`] when shapes do not fit together, and
    /// [`crate::error::Error::UnsupportedArchitecture`] for a `Linear` or
    /// `ConvNet` file, which is recognised but not run.
    pub fn build(file: NamFile) -> Result<Self> {
        let NamFile {
            version,
            config,
            metadata,
            weights,
            sample_rate,
        } = file;
        Ok(Self {
            inner: engine::build(config, &weights, sample_rate)?,
            expected_sample_rate: sample_rate,
            metadata,
            version,
        })
    }

    /// Give up the wrapper and keep the engine, for a host that would rather
    /// hold the trait object. The file-level fields are dropped.
    pub fn into_engine(self) -> Box<dyn Engine> {
        self.inner
    }

    /// Audio channels the model reads. See [`Engine::in_channels`].
    pub fn in_channels(&self) -> usize {
        self.inner.in_channels()
    }

    /// Audio channels the model writes. See [`Engine::out_channels`].
    pub fn out_channels(&self) -> usize {
        self.inner.out_channels()
    }

    /// Frames of silence the model needs to settle. See
    /// [`Engine::prewarm_samples`].
    pub fn prewarm_samples(&self) -> usize {
        self.inner.prewarm_samples()
    }

    /// The largest block [`Model::process`] accepts. See
    /// [`Engine::max_buffer_size`].
    pub fn max_buffer_size(&self) -> usize {
        self.inner.max_buffer_size()
    }

    /// Whether [`Model::reset`] settles the model. True initially. See
    /// [`Engine::set_prewarm_on_reset`].
    pub fn set_prewarm_on_reset(&mut self, on: bool) {
        self.inner.set_prewarm_on_reset(on);
    }

    /// The flag [`Model::set_prewarm_on_reset`] last stored.
    pub fn prewarm_on_reset(&self) -> bool {
        self.inner.prewarm_on_reset()
    }

    /// Settle the model on silence without resizing anything.
    ///
    /// [`Model::reset`] does this for you. Reach for it directly only when
    /// the model is already sized and you want to re-settle it, after
    /// seeking, say. Allocating; never call it from the audio thread. A
    /// model that was never sized is sized to
    /// [`crate::DEFAULT_MAX_BUFFER_SIZE`] first.
    pub fn prewarm(&mut self) {
        self.inner.prewarm();
    }

    /// Size buffers for blocks of up to `max_buffer` frames, then settle the
    /// model on silence. See [`Engine::reset`].
    ///
    /// Allocating. Call it before the audio thread starts, not from it, and
    /// again whenever the host's block size changes. `sample_rate` is in Hz
    /// and only recorded; compare it against
    /// [`Model::expected_sample_rate`] yourself.
    pub fn reset(&mut self, sample_rate: f64, max_buffer: usize) {
        self.inner.reset(sample_rate, max_buffer);
    }

    /// Size the model's buffers without settling it. See
    /// [`Engine::set_max_buffer_size`]. Allocating.
    pub fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.inner.set_max_buffer_size(max_buffer);
    }

    /// Process `n` frames. `input[ch][frame]`, `output[ch][frame]`.
    ///
    /// Allocation-free and safe to call from an audio callback once
    /// [`Model::reset`] has run. `input` needs at least
    /// [`Model::in_channels`] slices and `output` at least
    /// [`Model::out_channels`], each at least `n` long, and `n` is at most
    /// [`Model::max_buffer_size`]; anything else panics, as
    /// [`Engine::process`] describes.
    pub fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]], n: usize) {
        self.inner.process(input, output, n);
    }

    /// Process `n` frames from one column-major buffer into another. See
    /// [`Engine::process_buf`].
    pub fn process_buf(&mut self, input: &Buf, output: &mut Buf, n: usize) {
        self.inner.process_buf(input, output, n);
    }

    /// Select the size of a size-switchable model, `0.0` smallest to `1.0`
    /// largest. See [`Engine::set_slimmable_size`].
    ///
    /// Only a `SlimmableContainer` answers it; every other architecture
    /// returns [`crate::error::Error::Config`]. Allocating: selecting a child
    /// resets it.
    pub fn set_slimmable_size(&mut self, val: f64) -> Result<()> {
        self.inner.set_slimmable_size(val)
    }

    /// The control values in `(0.0, 1.0)` at which a switchable model
    /// changes size. See [`Engine::slimmable_size_breakpoints`].
    pub fn slimmable_size_breakpoints(&self) -> Vec<f64> {
        self.inner.slimmable_size_breakpoints()
    }

    /// [`Model::process`] for a mono model: processes as many frames as the
    /// shorter of the two slices holds.
    pub fn process_mono(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(output.len());
        let ins = [input];
        let mut outs = [output];
        self.process(&ins, &mut outs, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    /// `prewarm` on a model that was never `reset` sizes it to the default,
    /// so the reported size has to say so rather than staying at 0.
    #[test]
    fn a_bare_prewarm_reports_the_size_it_actually_settled_at() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/wavenet.nam");
        let mut m = Model::from_file(&p).unwrap();
        assert_eq!(m.max_buffer_size(), 0, "nothing has sized it yet");
        m.prewarm();
        assert_eq!(m.max_buffer_size(), crate::DEFAULT_MAX_BUFFER_SIZE);

        // An explicit size is still reported verbatim.
        m.reset(48_000.0, 77);
        assert_eq!(m.max_buffer_size(), 77);
        m.prewarm();
        assert_eq!(
            m.max_buffer_size(),
            77,
            "prewarm must not resize a sized model"
        );
    }

    #[test]
    fn every_bundled_fixture_loads() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models");
        let mut seen = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("nam") {
                continue;
            }
            seen += 1;
            let m = Model::from_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(m.in_channels() > 0 && m.out_channels() > 0, "{path:?}");
        }
        assert_eq!(seen, 12, "expected the bundled fixtures, found {seen}");
    }

    #[test]
    fn a_dropped_architecture_is_refused_clearly_even_when_nested() {
        // `Sequential` and `SlimmableContainer` hold arbitrary submodels, so a
        // ConvNet or Linear can arrive one level down. That has to surface as
        // the same clear refusal, not as a panic or a silently wrong model.
        let inner = serde_json::json!({
            "version": "0.5.4", "architecture": "Linear",
            "config": { "receptive_field": 4, "bias": true },
            "weights": [0.1, 0.2, 0.3, 0.4, 0.5], "sample_rate": 48000.0,
        });
        for wrapper in [
            serde_json::json!({
                "version": "0.5.4", "architecture": "Sequential",
                "config": { "models": [inner.clone()] },
                "weights": [], "sample_rate": 48000.0,
            }),
            serde_json::json!({
                "version": "0.7.0", "architecture": "SlimmableContainer",
                "config": { "submodels": [{ "max_value": 1.0, "model": inner.clone() }] },
                "weights": [], "sample_rate": 48000.0,
            }),
        ] {
            match Model::from_value(&wrapper) {
                Err(Error::UnsupportedArchitecture(m)) => {
                    assert!(
                        m.contains("Linear"),
                        "message should name the architecture: {m}"
                    )
                }
                other => panic!("expected a clear refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_model_reports_the_sample_rate_it_was_trained_at() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/wavenet.nam");
        let m = Model::from_file(path).unwrap();
        assert_eq!(m.expected_sample_rate, Some(48_000.0));
        assert_eq!(m.in_channels(), 1);
        assert_eq!(m.out_channels(), 1);
    }

    #[test]
    fn an_old_file_without_a_sample_rate_reports_none() {
        // `sample_rate` arrived after 0.5.0; the reference stands in -1.0,
        // which this surfaces as `None` rather than a nonsense rate.
        let json = serde_json::json!({
            "version": "0.5.0",
            "architecture": "LSTM",
            "config": { "input_size": 1, "hidden_size": 1, "num_layers": 0 },
            "weights": [0.5, 0.25],
        });
        let m = Model::from_value(&json).unwrap();
        assert_eq!(m.expected_sample_rate, None);
    }

    #[test]
    fn a_model_is_send_and_sync() {
        // What an audio host does with a model: build it on the message
        // thread, hand it to the callback. A single non-`Send` field anywhere
        // in the graph would break every consumer, in their crate rather than
        // this one.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Model>();
        assert_send_sync::<Box<dyn Engine>>();
    }
}
