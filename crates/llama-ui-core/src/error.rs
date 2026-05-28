//! Consolidated error types for llama-ui crates.

use thiserror::Error;

/// Unified error type for all llama-ui operations.
#[derive(Debug, Error)]
pub enum Error {
    /// I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization failed.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// HTTP request failed.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Sandbox process management failed.
    #[error("Sandbox error: {0}")]
    Sandbox(String),

    /// Session export/import failed.
    #[error("Session error: {0}")]
    Session(String),

    /// Network-related error.
    #[error("Network error: {0}")]
    Network(String),

    /// Other unspecified error.
    #[error("Error: {0}")]
    Other(String),
}

/// Result type alias for llama-ui operations.
pub type Result<T> = std::result::Result<T, Error>;
