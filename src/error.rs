//! Error types for `valverig-nam`.
//!
//! This is a library, so the error stays an enum a caller can match on rather
//! than an opaque box. That distinction earns its keep: a host loading a
//! capture wants to tell the user *"this file has 13,801 weights, the
//! architecture needs 13,802"*, which means reading
//! [`Error::WeightCount`]'s fields, not printing a string. A caller that only
//! prints and exits is free to box this into `anyhow` one level up; nothing
//! here assumes it will.

use thiserror::Error;

/// Any failure that can occur loading or running a NAM model.
#[derive(Debug, Error)]
pub enum Error {
    /// The file could not be read.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The file is not valid JSON.
    #[error("invalid JSON: {0}")]
    Json(String),

    /// The JSON is valid but is not a well-formed `.nam` file.
    #[error("invalid .nam file: {0}")]
    Schema(String),

    /// The `.nam` file version is outside the supported range.
    #[error("unsupported .nam version {found}: supported range is {earliest}..={latest}")]
    UnsupportedVersion {
        /// The version string found in the file.
        found: String,
        /// The oldest version this crate accepts.
        earliest: &'static str,
        /// The newest version this crate fully supports.
        latest: &'static str,
    },

    /// The architecture string is not one this crate implements.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// A configuration value is out of range or inconsistent with another.
    #[error("invalid model config: {0}")]
    Config(String),

    /// The flat weight array did not contain the number of floats the
    /// architecture requires.
    #[error("weight count mismatch: architecture needs {expected} floats, file has {found}")]
    WeightCount {
        /// How many floats the architecture consumed (or would have consumed).
        expected: usize,
        /// How many floats the file actually contained.
        found: usize,
    },
}

impl From<serde_json::Error> for Error {
    // Not `#[from]`: serde_json's own message already names the line and
    // column, and wrapping it as a source would print it twice.
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e.to_string())
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
