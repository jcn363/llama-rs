//! Hardware backend trait for tensor operations.
//!
//! Defines the [`Backend`] trait that all hardware backends (CPU, CUDA, etc.)
//! must implement, and the [`BackendInfo`] struct for reporting capabilities.
//!
//! This is the plugin interface: adding a new hardware backend means creating a
//! new crate that implements [`Backend`] and registering it with the registry.
//!
//! # Quantized types
//!
//! The [`QuantType`] enum and [`Backend::mat_vec_quant`] method support
//! quantized weight formats (Q4_0, Q8_0, Q4_1).  Backends can override
//! `mat_vec_quant` with format-specific kernels for 2–4× throughput
//! improvement over dequantize-then-compute.

/// Supported quantized tensor types for use with [`Backend::mat_vec_quant`].
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

/// A hardware backend capable of executing tensor operations.
///
/// This is **the** plugin interface for supporting different hardware.
/// Each backend implements the core math operations needed by the
/// inference engine. The trait is object-safe so backends can be
/// used polymorphically via `Arc<dyn Backend>`.
///
/// # Extending
///
/// To add a new hardware backend:
///
/// 1. Create a struct for your backend (e.g. `VulkanBackend`).
/// 2. Implement `Backend` for it.
/// 3. Register it with `BackendRegistry` so it participates in auto-selection.
///
/// # Notes on object-safety
///
/// The trait avoids generic parameters and uses only `&[f32]` / `Vec<f32>`
/// signatures so it remains object-safe. Parallelism is handled internally
/// by each backend.
pub trait Backend: Send + Sync {
    /// Returns information about this backend.
    fn info(&self) -> BackendInfo;

    /// Matrix-vector product: `y = weight @ input`
    ///
    /// `weight` has shape `(rows, cols)` in row-major order.
    /// `input` has length `cols`.
    /// Returns a vector of length `rows`.
    fn mat_vec(&self, weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32>;

    /// Quantized matrix-vector product: `y = quantized_weight @ input`
    ///
    /// `weight` is the quantized weight data in `quant_type` format with shape
    /// `(rows, cols)` in row-major order.  `input` has length `cols`.
    /// Returns a vector of length `rows`.
    ///
    /// The default implementation dequantizes each row block-by-block and
    /// falls back to dot-product logic.  Backends with format-specific
    /// kernels (SIMD, GPU) should override this.
    fn mat_vec_quant(
        &self,
        weight: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        default_mat_vec_quant(weight, quant_type, rows, cols, input)
    }

    /// Root Mean Square Normalization: `y = (x / RMS(x)) * weight`
    /// where `RMS(x) = sqrt(mean(x^2) + eps)`.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        default_rms_norm(x, weight, eps)
    }

    /// Sigmoid Linear Unit (SiLU) activation: `y = x * sigmoid(x)`
    fn silu(&self, x: &[f32]) -> Vec<f32> {
        default_silu(x)
    }

    /// Gaussian Error Linear Unit (GELU) activation:
    /// `y = x * Φ(x)` where Φ is the standard Gaussian CDF.
    fn gelu(&self, x: &[f32]) -> Vec<f32> {
        default_gelu(x)
    }

    /// Element-wise addition: `c[i] = a[i] + b[i]`
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_add(a, b)
    }

    /// Element-wise multiplication: `c[i] = a[i] * b[i]`
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_mul(a, b)
    }
}

// ─── Default (CPU fallback) implementations ──────────────────────────────────

/// Dequantize one block of a quantized type into f32 values.
/// The logical element count is [`QuantType::block_size`].
///
/// # Panics
///
/// Panics if `block_bytes` is shorter than the type's `block_bytes()`.
fn dequantize_block(block: &[u8], quant_type: QuantType, out: &mut [f32]) {
    let block_size = quant_type.block_size();
    assert_eq!(out.len(), block_size, "output slice must match block size");
    match quant_type {
        QuantType::Q4_0 => {
            assert!(block.len() >= 18);
            let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for i in 0..16 {
                let byte = block[2 + i];
                let lo = ((byte & 0x0F) as i8).wrapping_sub(8);
                let hi = ((byte >> 4) as i8).wrapping_sub(8);
                out[i * 2] = f32::from(lo) * scale;
                out[i * 2 + 1] = f32::from(hi) * scale;
            }
        }
        QuantType::Q4_1 => {
            assert!(block.len() >= 20);
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
            for i in 0..16 {
                let byte = block[4 + i];
                let lo = (byte & 0x0F) as i8;
                let hi = (byte >> 4) as i8;
                out[i * 2] = f32::from(lo) * d + m;
                out[i * 2 + 1] = f32::from(hi) * d + m;
            }
        }
        QuantType::Q8_0 => {
            assert!(block.len() >= 34);
            let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for i in 0..32 {
                out[i] = f32::from(block[2 + i] as i8) * scale;
            }
        }
    }
}

/// Default quantized matrix-vector product.
///
/// Dequantizes each row block-by-block and computes dot products.
/// This is a portable fallback — backends with format-specific kernels
/// should override [`Backend::mat_vec_quant`].
pub fn default_mat_vec_quant(
    weight: &[u8],
    quant_type: QuantType,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Vec<f32> {
    let block_size = quant_type.block_size();
    let block_bytes = quant_type.block_bytes();
    let n_blocks_per_row = cols.div_ceil(block_size);
    let mut dequant_buf = vec![0.0f32; block_size];
    (0..rows)
        .map(|r| {
            let mut sum = 0.0f32;
            for b in 0..n_blocks_per_row {
                let block_off = (r * n_blocks_per_row + b) * block_bytes;
                // Ensure we don't read past the buffer
                let block_end = (block_off + block_bytes).min(weight.len());
                dequantize_block(&weight[block_off..block_end], quant_type, &mut dequant_buf);
                let col_start = b * block_size;
                let n = cols.saturating_sub(col_start).min(block_size);
                for i in 0..n {
                    sum += dequant_buf[i] * input[col_start + i];
                }
            }
            sum
        })
        .collect()
}

/// Default matrix-vector product implementation using sequential iteration.
///
/// This is a portable fallback used by backends that don't accelerate
/// matrix-vector multiplication (or for small matrices where dispatch
/// overhead isn't worth it).
pub fn default_mat_vec(weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let start = r * cols;
            let row = &weight[start..start + cols];
            row.iter().zip(input.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Default element-wise addition.
pub fn default_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Default element-wise multiplication.
pub fn default_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Default RMSNorm implementation.
pub fn default_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    // Delegate to SIMD implementation for performance, fallback to scalar if unavailable.
    // The improvements module provides a DRY implementation.
    crate::improvements::rms_norm_simd(x, weight, eps)
}

/// Default SiLU (Swish) activation: x * sigmoid(x)
pub fn default_silu(x: &[f32]) -> Vec<f32> {
    // Use SIMD implementation from improvements module.
    crate::improvements::silu_simd(x)
}

/// Default GELU activation: x * Φ(x) where Φ is the standard Gaussian CDF approximation.
pub fn default_gelu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| {
            // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            let sqrt_2_over_pi = 0.797_884_6;
            let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            let tanh = (inner.exp() - (-inner).exp()) / (inner.exp() + (-inner).exp());
            0.5 * v * (1.0 + tanh)
        })
        .collect()
}
