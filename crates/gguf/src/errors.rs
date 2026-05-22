// ─── Errors ──────────────────────────────────────────────────────────────────

use std::io;
use thiserror::Error;

/// Errors that can occur when reading or writing GGUF files.
#[derive(Debug, Error)]
pub enum GgufError {
    /// The file could not be opened or read.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// The file does not have a valid GGUF magic number.
    #[error("invalid GGUF magic number")]
    InvalidMagic,

    /// The GGUF version is not supported.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// A value could not be decoded.
    #[error("decode error: {0}")]
    DecodeError(String),

    /// The file is truncated or incomplete.
    #[error("unexpected end of file")]
    UnexpectedEof,

    /// The tensor index is out of range.
    #[error("tensor index {0} out of range (max {1})")]
    TensorIndexOutOfRange(usize, usize),

    /// The KV key index is out of range.
    #[error("KV key index {0} out of range (max {1})")]
    KvIndexOutOfRange(usize, usize),
}

/// Result type alias for GGUF operations.
pub type GgufResult<T> = Result<T, GgufError>;
