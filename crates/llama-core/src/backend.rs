//! Hardware backend types shared across backend implementations.

/// Supported quantized tensor types for use with
/// `Backend::mat_vec_quant` (the `ggml::backend::Backend` trait).
///
/// Each variant corresponds to a GGML quantization format with known
/// block structure (block size in elements, block size in bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 4-bit quantized, block = 32 elems, 18 bytes.
    Q4_0,
    /// 4-bit quantized with min, block = 32 elems, 20 bytes.
    Q4_1,
    /// 8-bit quantized, block = 32 elems, 34 bytes.
    Q8_0,
}

impl QuantType {
    /// Number of f32 elements per quantized block.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
        }
    }

    /// Size of a quantized block in bytes.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q8_0 => 34,
        }
    }
}

/// Information about a hardware backend's capabilities.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Human-readable name (e.g. `"CPU"`, `"CUDA"`).
    pub name: &'static str,
    /// Whether the backend is available and ready for use.
    pub is_available: bool,
    /// Total device memory in bytes (0 if not applicable, e.g. CPU).
    pub total_memory: usize,
    /// Free device memory in bytes.
    pub free_memory: usize,
    /// Degree of parallelism (threads, SM count, CUDA cores, etc.).
    pub parallelism: usize,
}
