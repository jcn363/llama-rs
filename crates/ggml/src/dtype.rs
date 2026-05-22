//! Data type definitions used by ggml tensors.

/// Primitive data types supported by ggml tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32‑bit floating point.
    F32,
    /// 16‑bit floating point (half precision).
    F16,
    /// 8‑bit integer.
    I8,
    /// 8‑bit unsigned integer.
    U8,
    /// 32‑bit signed integer.
    I32,
    /// 64‑bit signed integer.
    I64,
}

impl DType {
    /// Size in bytes of a single element of this data type.
    #[must_use]
    pub const fn size_of(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
            DType::U8 => 1,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}
