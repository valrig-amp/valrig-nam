//! The `.nam` file format: parsing, validation and typed configuration.
//!
//! A `.nam` file is a JSON object with four required keys (`version`,
//! `architecture`, `config`, `weights`) plus optional `metadata` and
//! `sample_rate`. This module turns one into a [`NamFile`]: a checked version
//! string, a fully typed [`ArchConfig`], the metadata the reference reads, and
//! the flat weight array that the model constructors consume in order.
//!
//! Nothing here runs any DSP. The reference reaches the same information
//! through `nam::validate_nam_file` (`NAM/nam_file.cpp`),
//! `nam::populate_dsp_data` (`NAM/get_dsp.cpp`) and the per-architecture
//! `parse_config_json` functions, and this module is a transliteration of that
//! path.
//!
//! ```
//! use valverig_nam::format::{self, ArchConfig};
//!
//! # let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/models/wavenet.nam");
//! let file = format::load_file(path)?;
//! assert_eq!(file.version, "0.5.4");
//! assert_eq!(file.sample_rate, Some(48_000.0));
//! let ArchConfig::WaveNet(cfg) = &file.config else { panic!("not a WaveNet") };
//! assert_eq!(cfg.layer_arrays.len(), 2);
//! # Ok::<(), valverig_nam::error::Error>(())
//! ```
//!
//! # Divergences from the reference
//!
//! * Where the reference indexes a *const* `nlohmann::json` with `operator[]`
//!   and a missing key, which is undefined behaviour once `NDEBUG` is set,
//!   this module returns an error naming the key.
//! * Sizes and counts are `usize`. The reference stores them in `int` and
//!   would carry a negative value into a `resize`; a negative here is an
//!   error, and so is anything above [`MAX_COUNT`].
//! * The reference defers `Sequential` and `SlimmableContainer` parsing to
//!   model-construction time. It happens during parsing here, so those errors
//!   surface earlier. The errors themselves are unchanged.
//! * A stated `sample_rate` must be a positive finite number no larger than
//!   [`MIN_SAMPLE_RATE`] to [`MAX_SAMPLE_RATE`]; the reference accepts any
//!   double.
//! * A config with more than one problem may not report the same one. The
//!   reference checks in a fixed order; a `config` block is deserialized here
//!   before any of it is checked, so a type error anywhere in it precedes
//!   every consistency check. Which files are accepted and which are refused
//!   is unaffected; only which message a doubly-broken one gets.

mod config;
mod de;
mod version;

pub use config::{
    ArchConfig, ContainerConfig, FILM_KEYS, FilmConfig, GatingMode, Head1x1Config, Layer1x1Config,
    LayerArrayConfig, LayerConfig, LstmConfig, PostStackHeadConfig, SequentialConfig, Submodel,
    WaveNetConfig, activation_from_json, film_site,
};
pub use version::{
    EARLIEST_SUPPORTED_NAM_FILE_VERSION, LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION, Supported,
    is_version_supported, parse_version, verify_version,
};

use crate::error::{Error, Result};
use serde_json::Value;
use std::path::Path;

/// Largest value any count in a file may take: channels, kernel sizes,
/// dilations, layer counts, hidden sizes, groups.
///
/// Every allocation the crate makes for a model is a product of such counts
/// with the host's block size, and a weight matrix is additionally checked
/// against the file's own weight array before it is allocated. Real captures
/// stay below a few thousand on every count; the cap exists so that a
/// malformed file is refused rather than sizing an allocation the process
/// cannot survive.
pub const MAX_COUNT: usize = 1 << 20;

/// Largest total size, in floats, of a WaveNet's convolution histories.
///
/// A convolution keeps `channels × (kernel_size - 1) × dilation` floats of
/// input history; this bounds the sum over a model. NAM's standard
/// architecture needs about 330 000; the A2 capture about a million.
pub const MAX_HISTORY_FLOATS: usize = 1 << 25;

/// Lowest `sample_rate` a file may state, in Hz.
///
/// The same bound the rest of ValveRig works to, so that a capture this
/// crate accepts is one the DSP stages beside it will also run at: a rate
/// they refuse would otherwise parse here and fail later, when a rack is
/// prepared rather than when the file is read.
pub const MIN_SAMPLE_RATE: f64 = 1_000.0;

/// Largest `sample_rate` a file may state, in Hz.
///
/// As [`MIN_SAMPLE_RATE`], and the LSTM's prewarm length is half a second of
/// samples, so the rate also bounds how long a `reset` can take.
pub const MAX_SAMPLE_RATE: f64 = 768_000.0;

/// The keys `validate_nam_file` requires of every `.nam` document, top-level
/// or nested.
pub(crate) const REQUIRED_KEYS: [&str; 4] = ["version", "architecture", "config", "weights"];

/// A parsed `.nam` file.
#[derive(Debug, Clone, PartialEq)]
pub struct NamFile {
    /// The `version` string, verbatim. Already checked against the supported
    /// range; see [`verify_version`].
    pub version: String,
    /// The typed `config` block, discriminated by the file's `architecture`.
    pub config: ArchConfig,
    /// The `metadata` block. Empty when the file has none.
    pub metadata: Metadata,
    /// The flat `weights` array, in file order.
    ///
    /// Model constructors walk this front to back; the order in which they do
    /// so is the architecture's real definition, so this stays a flat `Vec`
    /// rather than being split up here.
    pub weights: Vec<f32>,
    /// The `sample_rate` the model was trained at, in Hz, or `None` when the
    /// file does not say, because the key is absent or holds the reference's
    /// `NAM_UNKNOWN_EXPECTED_SAMPLE_RATE` sentinel, `-1`.
    pub sample_rate: Option<f64>,
}

/// The `metadata` block.
///
/// The reference reads only `loudness`, `input_level_dbu` and
/// `output_level_dbu`, and requires each to be a number when present
/// (`get_dsp_with_current_prewarm_default` in `NAM/get_dsp.cpp`). Those three
/// are strict here for the same reason. `gain` and `name` are read for the
/// benefit of hosts; since the reference never touches them, a value of an
/// unexpected type is left as `None` rather than being made an error.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    /// Measured loudness of the model's output, in dBFS.
    pub loudness: Option<f64>,
    /// Linear gain applied during training.
    pub gain: Option<f64>,
    /// Input level the model was calibrated at, in dBu.
    pub input_level_dbu: Option<f64>,
    /// Output level the model was calibrated at, in dBu.
    pub output_level_dbu: Option<f64>,
    /// Human-readable model name.
    pub name: Option<String>,
    /// The whole block, so nothing is lost: `date`, `training`, `gear_make`
    /// and any other key a trainer writes. [`Value::Null`] when the file has
    /// no `metadata`.
    pub raw: Value,
}

impl Metadata {
    /// Extract the fields the reference reads, and keep the rest.
    fn from_json(v: &Value) -> Result<Self> {
        if v.is_null() {
            return Ok(Metadata::default());
        }
        // The reference's `extract` lambda: present and non-null, then
        // `get<double>()`, which throws on anything that is not a number.
        let strict = |key: &str| -> Result<Option<f64>> {
            match v.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(x) => Ok(Some(de::as_f64(x, &format!("metadata.{key}"))?)),
            }
        };
        Ok(Metadata {
            loudness: strict("loudness")?,
            gain: v.get("gain").and_then(Value::as_f64),
            input_level_dbu: strict("input_level_dbu")?,
            output_level_dbu: strict("output_level_dbu")?,
            name: v.get("name").and_then(Value::as_str).map(str::to_owned),
            raw: v.clone(),
        })
    }
}

/// Read and parse a `.nam` file from disk.
///
/// `nam::get_dsp(path)` minus the DSP: `validate_nam_file` then
/// `populate_dsp_data`.
///
/// File-level failures ([`Error::Schema`], [`Error::Json`]) name the path, as
/// `NamFileValidationError` does. Architecture-level failures do not: the
/// reference raises those from deep inside a config parser that has never
/// seen a filename.
pub fn load_file<P: AsRef<Path>>(path: P) -> Result<NamFile> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(Error::Schema(format!(
            "Could not validate .nam file [{}]: file does not exist.",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path)?;
    parse_json(&text).map_err(|e| match e {
        Error::Schema(m) => Error::Schema(format!("Invalid .nam file [{}]: {m}", path.display())),
        Error::Json(m) => Error::Json(format!(
            "Could not parse .nam file [{}]: {m}",
            path.display()
        )),
        other => other,
    })
}

/// Parse a `.nam` file from a JSON string.
///
/// Fails with [`Error::Json`] when `s` is not JSON, and otherwise as
/// [`parse_value`].
pub fn parse_json(s: &str) -> Result<NamFile> {
    let v: Value = serde_json::from_str(s)?;
    parse_value(&v)
}

/// Parse an already-decoded `.nam` document.
///
/// Also the recursion point: `Sequential` children, `SlimmableContainer`
/// submodels and a WaveNet's `condition_dsp` are each a complete `.nam`
/// document, and the reference reaches all three through `get_dsp(json)`.
///
/// Fails with [`Error::Schema`] when the document is not an object with the
/// four required keys or `weights` is not an array of numbers,
/// [`Error::UnsupportedVersion`] outside the supported version range,
/// [`Error::UnsupportedArchitecture`] for a name the reference does not
/// register, and [`Error::Config`] for anything wrong inside `config`.
pub fn parse_value(v: &Value) -> Result<NamFile> {
    if !v.is_object() {
        return Err(Error::Schema("root JSON value must be an object.".into()));
    }
    for key in REQUIRED_KEYS {
        if v.get(key).is_none() {
            return Err(Error::Schema(format!("missing required key \"{key}\".")));
        }
    }

    let version = de::as_str(&v["version"], "version")?.to_string();
    version::verify_version(&version)?;

    let weights = parse_weights(&v["weights"])?;
    let architecture = de::as_str(&v["architecture"], "architecture")?;

    let sample_rate = match v.get("sample_rate") {
        Some(x) => parse_sample_rate(de::as_f64(x, "sample_rate")?)?,
        None => None,
    };
    let metadata = Metadata::from_json(v.get("metadata").unwrap_or(&Value::Null))?;

    let config = config::parse_arch_config(architecture, &v["config"], sample_rate, &weights)?;
    match &config {
        ArchConfig::SlimmableContainer(c) => c.check_sample_rates(sample_rate)?,
        ArchConfig::Sequential(s) => s.check_sample_rates(sample_rate)?,
        _ => {}
    }

    Ok(NamFile {
        version,
        config,
        metadata,
        weights,
        sample_rate,
    })
}

/// The stated `sample_rate`: the reference's `-1` sentinel is "unknown", and
/// anything else has to be a rate a host could run at.
fn parse_sample_rate(rate: f64) -> Result<Option<f64>> {
    if rate == -1.0 {
        return Ok(None);
    }
    if !(rate.is_finite() && (MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&rate)) {
        return Err(Error::Config(format!(
            "sample_rate must be -1, or between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz, found {rate}"
        )));
    }
    Ok(Some(rate))
}

/// Convert the `weights` array to the reference's `std::vector<float>`.
///
/// The reference parses each token into an `nlohmann::json` number, `int64_t`
/// for a token with no fraction or exponent and `double` otherwise, and then
/// narrows to `float` with a single `static_cast`. Matching that means
/// narrowing exactly once from the same intermediate, which is what the two
/// arms below do.
fn parse_weights(v: &Value) -> Result<Vec<f32>> {
    let arr = v
        .as_array()
        .ok_or_else(|| Error::Schema(format!("weights must be an array, found {}", de::kind(v))))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, e) in arr.iter().enumerate() {
        let Value::Number(n) = e else {
            return Err(Error::Schema(format!(
                "weights[{i}]: expected a number, found {}",
                de::kind(e)
            )));
        };
        out.push(if let Some(x) = n.as_i64() {
            x as f32
        } else if let Some(x) = n.as_u64() {
            x as f32
        } else {
            n.as_f64().unwrap_or(f64::NAN) as f32
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
