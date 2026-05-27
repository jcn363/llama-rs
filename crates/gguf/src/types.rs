//! GGUF value types and GGML tensor types.
//!
//! This module contains the type enumerations that were previously
//! defined in `lib.rs`.  Extracting them keeps the main library file
//! focused on the top‑level reader API.

use crate::GgufError;

// ─── GGUF Value Types ────────────────────────────────────────────────────────

/// GGUF metadata value types (13 total).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgufType {
    /// Unsigned 8-bit integer.
    Uint8 = 0,
    /// Signed 8-bit integer.
    Int8 = 1,
    /// Unsigned 16-bit integer.
    Uint16 = 2,
    /// Signed 16-bit integer.
    Int16 = 3,
    /// Unsigned 32-bit integer.
    Uint32 = 4,
    /// Signed 32-bit integer.
    Int32 = 5,
    /// 32-bit IEEE 754 float.
    Float32 = 6,
    /// Boolean (stored as int8).
    Bool = 7,
    /// UTF-8 string.
    String = 8,
    /// Array of homogeneous values.
    Array = 9,
    /// Unsigned 64-bit integer.
    Uint64 = 10,
    /// Signed 64-bit integer.
    Int64 = 11,
    /// 64-bit IEEE 754 float.
    Float64 = 12,
}

impl GgufType {
    /// Returns the size in bytes of a single value of this type.
    /// Returns 0 for variable-size types (String, Array).
    #[must_use]
    pub fn size_of(self) -> usize {
        match self {
            GgufType::Uint8 | GgufType::Int8 | GgufType::Bool => 1,
            GgufType::Uint16 | GgufType::Int16 => 2,
            GgufType::Uint32 | GgufType::Int32 | GgufType::Float32 => 4,
            GgufType::Uint64 | GgufType::Int64 | GgufType::Float64 => 8,
            GgufType::String | GgufType::Array => 0,
        }
    }

    /// Returns the human-readable name of this type.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GgufType::Uint8 => "u8",
            GgufType::Int8 => "i8",
            GgufType::Uint16 => "u16",
            GgufType::Int16 => "i16",
            GgufType::Uint32 => "u32",
            GgufType::Int32 => "i32",
            GgufType::Float32 => "f32",
            GgufType::Bool => "bool",
            GgufType::String => "str",
            GgufType::Array => "arr",
            GgufType::Uint64 => "u64",
            GgufType::Int64 => "i64",
            GgufType::Float64 => "f64",
        }
    }

    /// Try to convert from a raw i32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not a valid GGUF type.
    pub fn from_i32(v: i32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(GgufType::Uint8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::Uint16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::Uint32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::Uint64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            _ => Err(GgufError::DecodeError(format!("unknown gguf_type: {v}"))),
        }
    }
}

// ─── GGML Tensor Types (re-exported from llama_core) ──────────────────────────

pub use llama_core::GgmlType;
