// ─── GGUF Value Types ────────────────────────────────────────────────────────

use crate::GgufType;

/// A single GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// String.
    Str(String),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
    /// Array of values.
    Array {
        /// Element type.
        elem_type: GgufType,
        /// Elements.
        data: Vec<GgufValue>,
    },
}

impl GgufValue {
    /// Returns the GGUF type of this value.
    #[must_use]
    pub fn gguf_type(&self) -> GgufType {
        match self {
            GgufValue::U8(_) => GgufType::Uint8,
            GgufValue::I8(_) => GgufType::Int8,
            GgufValue::U16(_) => GgufType::Uint16,
            GgufValue::I16(_) => GgufType::Int16,
            GgufValue::U32(_) => GgufType::Uint32,
            GgufValue::I32(_) => GgufType::Int32,
            GgufValue::F32(_) => GgufType::Float32,
            GgufValue::Bool(_) => GgufType::Bool,
            GgufValue::Str(_) => GgufType::String,
            GgufValue::U64(_) => GgufType::Uint64,
            GgufValue::I64(_) => GgufType::Int64,
            GgufValue::F64(_) => GgufType::Float64,
            GgufValue::Array { .. } => GgufType::Array,
        }
    }
}
