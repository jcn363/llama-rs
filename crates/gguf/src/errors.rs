use thiserror::Error;

/// Errors specific to GGUF parsing and handling.
#[derive(Debug, Error)]
pub enum GgufError {
    /// An I/O error occurred while reading the GGUF file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not start with the GGUF magic number `GGUF`.
    #[error("Invalid magic number")]
    InvalidMagic,

    /// The GGUF file declares a version not supported by this reader.
    #[error("Unsupported version {0}")]
    UnsupportedVersion(u32),

    /// Reached end-of-file before the expected data was fully read.
    #[error("Unexpected EOF")]
    UnexpectedEof,

    /// A binary decode step failed (e.g. invalid enum discriminant).
    #[error("Decode error: {0}")]
    DecodeError(String),

    /// A key-value metadata index is beyond the declared count.
    #[error("KV index out of range {0}/{1}")]
    KvIndexOutOfRange(usize, usize),

    /// A tensor metadata index is beyond the declared count.
    #[error("Tensor index out of range {0}/{1}")]
    TensorIndexOutOfRange(usize, usize),
}

/// Alias for `Result<T, GgufError>` used throughout the gguf crate.
pub type GgufResult<T> = std::result::Result<T, GgufError>;
