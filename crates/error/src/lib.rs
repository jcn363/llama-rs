//! Unified error type for the llama-rs workspace.
//!
//! All crates in the workspace use the same [`Error`] enum, enabling
//! consistent error propagation with `?` via the [`Result`] type alias.

#![deny(missing_docs)]

use thiserror::Error as DeriveError;

/// Central error type for the project.
#[derive(Debug, DeriveError)]
pub enum Error {
    /// Wraps [`std::io::Error`] — file I/O, network streams, etc.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid or missing configuration.
    #[error("Configuration error: {0}")]
    Config(String),
    /// GGUF parsing or format error.
    #[error("GGUF parsing error: {0}")]
    Gguf(String),
    /// Network / HTTP request error.
    #[error("Network error: {0}")]
    Network(String),
    /// General parse / deserialization error.
    #[error("Parse error: {0}")]
    Parse(String),
    /// Chat template rendering error.
    #[error("Template error: {0}")]
    Template(String),
    /// GGUF metadata extraction error.
    #[error("GGUF metadata error: {0}")]
    GgufMeta(String),
    /// Catch-all error variant.
    #[error("Other error: {0}")]
    Other(String),
}

/// Alias for `Result<T, Error>` used throughout the workspace.
pub type Result<T> = std::result::Result<T, Error>;
