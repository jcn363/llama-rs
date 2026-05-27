use thiserror::Error;

/// Errors specific to GGUF parsing and handling.
#[derive(Debug, Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number")]
    InvalidMagic,
    #[error("Unsupported version {0}")]
    UnsupportedVersion(u32),
    #[error("Unexpected EOF")]
    UnexpectedEof,
    #[error("Decode error: {0}")]
    DecodeError(String),
    #[error("KV index out of range {0}/{1}")]
    KvIndexOutOfRange(usize, usize),
    #[error("Tensor index out of range {0}/{1}")]
    TensorIndexOutOfRange(usize, usize),
}

/// Alias for `Result<T, GgufError>` used throughout the gguf crate.
pub type GgufResult<T> = std::result::Result<T, GgufError>;
