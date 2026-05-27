//! GGML tensor data types.
//!
//! These are the tensor data type identifiers used in GGML and GGUF formats.
//! This module was extracted from `gguf` so that shared types in `llama-core`
//! can reference `GgmlType` without creating circular dependencies.

/// GGML tensor data types (stored as i32 in GGUF files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    /// IQ1 M.
    Iq1M = 24,
    /// IQ3 M.
    Iq3M = 25,
    /// IQ3 XS.
    Iq3Xs = 26,
    /// 8-bit integer.
    I8 = 27,
    /// 16-bit integer.
    I16 = 28,
    /// 32-bit integer.
    I32 = 29,
    /// 64-bit integer.
    I64 = 30,
    /// 64-bit float.
    F64 = 31,
    /// Brain float 16.
    Bf16 = 32,
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
    /// Returns an error string if the value is not a valid GGML type.
    pub fn from_i32(v: i32) -> Result<Self, String> {
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
            24 => Ok(GgmlType::Iq1M),
            25 => Ok(GgmlType::Iq3M),
            26 => Ok(GgmlType::Iq3Xs),
            27 => Ok(GgmlType::I8),
            28 => Ok(GgmlType::I16),
            29 => Ok(GgmlType::I32),
            30 => Ok(GgmlType::I64),
            31 => Ok(GgmlType::F64),
            32 => Ok(GgmlType::Bf16),
            34 => Ok(GgmlType::Tq1_0),
            35 => Ok(GgmlType::Tq2_0),
            39 => Ok(GgmlType::Mxfp4),
            40 => Ok(GgmlType::Nvfp4),
            41 => Ok(GgmlType::Q1_0),
            _ => Err(format!("unknown ggml_type: {v}")),
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
            GgmlType::Iq1M => "iq1_m",
            GgmlType::Iq3M => "iq3_m",
            GgmlType::Iq3Xs => "iq3_xs",
            GgmlType::I8 => "i8",
            GgmlType::I16 => "i16",
            GgmlType::I32 => "i32",
            GgmlType::I64 => "i64",
            GgmlType::F64 => "f64",
            GgmlType::Bf16 => "bf16",
            GgmlType::Tq1_0 => "tq1_0",
            GgmlType::Tq2_0 => "tq2_0",
            GgmlType::Mxfp4 => "mxfp4",
            GgmlType::Nvfp4 => "nvfp4",
            GgmlType::Q1_0 => "q1_0",
        }
    }
}
