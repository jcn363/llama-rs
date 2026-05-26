#![deny(missing_docs)]

//! Unified error handling for the llama-rs workspace.
//!
//! Provides a single `Error` enum that aggregates errors from various
//! crates and a convenient `Result<T>` alias.

use thiserror::Error;

/// Central error type for the project.
#[derive(Debug, Error)]
pub enum Error {
    /// Wraps [`std::io::Error`] via `From` conversion.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration-related error with a descriptive message.
    #[error("Configuration error: {0}")]
    Config(String),

    /// GGUF parsing error with a descriptive message.
    #[error("GGUF parsing error: {0}")]
    Gguf(String),

    /// Network/download error.
    #[error("Network error: {0}")]
    Network(String),

    /// Parse error (JSON, TOML, etc.).
    #[error("Parse error: {0}")]
    Parse(String),

    /// Chat template rendering error.
    #[error("Template error: {0}")]
    Template(String),

    /// GGUF metadata extraction error.
    #[error("GGUF metadata error: {0}")]
    GgufMeta(String),

    /// Catch-all error with a descriptive message.
    #[error("Other error: {0}")]
    Other(String),
}

// Note: From impls for serde_json, toml, reqwest are intentionally omitted
// to keep the error crate dependency-free beyond thiserror.
// Convert at call sites via .map_err(|e| Error::Parse(e.to_string())).

/// Alias for `Result<T, Error>` used throughout the workspace.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_io_should_format_correctly() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = Error::Io(io_err);
        let display = format!("{err}");
        assert!(display.contains("IO error"));
        assert!(display.contains("file not found"));
    }

    #[test]
    fn error_config_should_format_correctly() {
        let err = Error::Config("missing key".into());
        assert_eq!(format!("{err}"), "Configuration error: missing key");
    }

    #[test]
    fn error_gguf_should_format_correctly() {
        let err = Error::Gguf("invalid magic".into());
        assert_eq!(format!("{err}"), "GGUF parsing error: invalid magic");
    }

    #[test]
    fn error_other_should_format_correctly() {
        let err = Error::Other("something went wrong".into());
        assert_eq!(format!("{err}"), "Other error: something went wrong");
    }

    #[test]
    fn error_should_impl_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }

    #[test]
    fn result_type_alias_should_work() {
        fn returns_result() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(returns_result().unwrap(), 42);
    }

    #[test]
    fn error_should_be_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn error_network_should_format_correctly() {
        let err = Error::Network("connection refused".into());
        assert_eq!(format!("{err}"), "Network error: connection refused");
    }

    #[test]
    fn error_parse_should_format_correctly() {
        let err = Error::Parse("invalid toml".into());
        assert_eq!(format!("{err}"), "Parse error: invalid toml");
    }

    #[test]
    fn error_template_should_format_correctly() {
        let err = Error::Template("missing variable".into());
        assert_eq!(format!("{err}"), "Template error: missing variable");
    }

    #[test]
    fn error_gguf_meta_should_format_correctly() {
        let err = Error::GgufMeta("unknown architecture".into());
        assert_eq!(format!("{err}"), "GGUF metadata error: unknown architecture");
    }
}
