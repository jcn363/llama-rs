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

// ─── GGML Tensor Types ───────────────────────────────────────────────────────

/// GGML tensor data types (stored as i32 in GGUF files).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[expect(non_camel_case_types)]
pub enum GgmlType {
    /// 32-bit float.
    F32 = 0,
    /// 16-bit float.
    F16 = 1,
    /// 4-bit quantized (variant 0).
    Q4_0 = 2,
    /// 4-bit quantized (variant 1).
    Q4_1 = 3,
    /// 5-bit quantized (variant 0).
    Q5_0 = 6,
    /// 5-bit quantized (variant 1).
    Q5_1 = 7,
    /// 8-bit quantized (variant 0).
    Q8_0 = 8,
    /// 8-bit quantized (variant 1).
    Q8_1 = 9,
    /// 2-bit K-quants.
    Q2_K = 10,
    /// 3-bit K-quants.
    Q3_K = 11,
    /// 4-bit K-quants.
    Q4_K = 12,
    /// 5-bit K-quants.
    Q5_K = 13,
    /// 6-bit K-quants.
    Q6_K = 14,
    /// 8-bit K-quants.
    Q8_K = 15,
    /// IQ2 XXS.
    Iq2Xxs = 16,
    /// IQ2 XS.
    Iq2Xs = 17,
    /// IQ3 XXS.
    Iq3Xxs = 18,
    /// IQ1 S.
    Iq1S = 19,
    /// IQ4 NL.
    Iq4Nl = 20,
    /// IQ3 S.
    Iq3S = 21,
    /// IQ2 S.
    Iq2S = 22,
    /// IQ4 XS.
    Iq4Xs = 23,
    /// 8-bit integer.
    I8 = 24,
    /// 16-bit integer.
    I16 = 25,
    /// 32-bit integer.
    I32 = 26,
    /// 64-bit integer.
    I64 = 27,
    /// 64-bit float.
    F64 = 28,
    /// IQ1 M.
    Iq1M = 29,
    /// Brain float 16.
    Bf16 = 30,
    /// Ternary quantized 1.0.
    Tq1_0 = 34,
    /// Ternary quantized 2.0.
    Tq2_0 = 35,
    /// MXFP4.
    Mxfp4 = 39,
    /// NVFP4.
    Nvfp4 = 40,
    /// 1-bit quantized.
    Q1_0 = 41,
}

impl GgmlType {
    /// Try to convert from a raw i32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not a valid GGML type.
    pub fn from_i32(v: i32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2_K),
            11 => Ok(GgmlType::Q3_K),
            12 => Ok(GgmlType::Q4_K),
            13 => Ok(GgmlType::Q5_K),
            14 => Ok(GgmlType::Q6_K),
            15 => Ok(GgmlType::Q8_K),
            16 => Ok(GgmlType::Iq2Xxs),
            17 => Ok(GgmlType::Iq2Xs),
            18 => Ok(GgmlType::Iq3Xxs),
            19 => Ok(GgmlType::Iq1S),
            20 => Ok(GgmlType::Iq4Nl),
            21 => Ok(GgmlType::Iq3S),
            22 => Ok(GgmlType::Iq2S),
            23 => Ok(GgmlType::Iq4Xs),
            24 => Ok(GgmlType::I8),
            25 => Ok(GgmlType::I16),
            26 => Ok(GgmlType::I32),
            27 => Ok(GgmlType::I64),
            28 => Ok(GgmlType::F64),
            29 => Ok(GgmlType::Iq1M),
            30 => Ok(GgmlType::Bf16),
            34 => Ok(GgmlType::Tq1_0),
            35 => Ok(GgmlType::Tq2_0),
            39 => Ok(GgmlType::Mxfp4),
            40 => Ok(GgmlType::Nvfp4),
            41 => Ok(GgmlType::Q1_0),
            _ => Err(GgufError::DecodeError(format!("unknown ggml_type: {v}"))),
        }
    }

    /// Returns the human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::Q4_0 => "q4_0",
            GgmlType::Q4_1 => "q4_1",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q5_1 => "q5_1",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Q8_1 => "q8_1",
            GgmlType::Q2_K => "q2_k",
            GgmlType::Q3_K => "q3_k",
            GgmlType::Q4_K => "q4_k",
            GgmlType::Q5_K => "q5_k",
            GgmlType::Q6_K => "q6_k",
            GgmlType::Q8_K => "q8_k",
            GgmlType::Iq2Xxs => "iq2_xxs",
            GgmlType::Iq2Xs => "iq2_xs",
            GgmlType::Iq3Xxs => "iq3_xxs",
            GgmlType::Iq1S => "iq1_s",
            GgmlType::Iq4Nl => "iq4_nl",
            GgmlType::Iq3S => "iq3_s",
            GgmlType::Iq2S => "iq2_s",
            GgmlType::Iq4Xs => "iq4_xs",
            GgmlType::I8 => "i8",
            GgmlType::I16 => "i16",
            GgmlType::I32 => "i32",
            GgmlType::I64 => "i64",
            GgmlType::F64 => "f64",
            GgmlType::Iq1M => "iq1_m",
            GgmlType::Bf16 => "bf16",
            GgmlType::Tq1_0 => "tq1_0",
            GgmlType::Tq2_0 => "tq2_0",
            GgmlType::Mxfp4 => "mxfp4",
            GgmlType::Nvfp4 => "nvfp4",
            GgmlType::Q1_0 => "q1_0",
        }
    }
}
