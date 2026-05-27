use thiserror::Error;

/// Central error type for the project.
#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("GGUF parsing error: {0}")]
    Gguf(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Template error: {0}")]
    Template(String),
    #[error("GGUF metadata error: {0}")]
    GgufMeta(String),
    #[error("Other error: {0}")]
    Other(String),
}

/// Alias for `Result<T, Error>` used throughout the workspace.
pub type Result<T> = std::result::Result<T, Error>;