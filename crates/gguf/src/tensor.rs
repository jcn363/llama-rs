//! Tensor info and memory-mapped tensor reference.
//!
//! Contains [`TensorInfo`] (metadata about a tensor in a GGUF file) and
//! [`MmapTensor`] (a lazy memory-mapped reference to tensor data).
//! Both were previously defined in `lib.rs`.

use std::sync::Arc;

use crate::dequant::*;
use crate::{GgmlType, GgufError};

// ─── Tensor Info ─────────────────────────────────────────────────────────────

/// Information about a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape (dimensions in reverse order from file, matching ggml convention).
    pub shape: Vec<i64>,
    /// Data type.
    pub dtype: GgmlType,
    /// Offset into the tensor data blob.
    pub offset: u64,
}

impl TensorInfo {
    /// De‑quantize the raw tensor bytes into a `Vec<f32>`.
    /// Supports F32, F16, and common quantization types (``Q4_0``, ``Q4_1``,
    /// ``Q5_0``, ``Q5_1``, ``Q8_0``, ``Q2_K``-``Q6_K``).
    ///
    /// # Panics
    ///
    /// This function will panic if the underlying byte slices cannot be
    /// converted to the expected array sizes via `try_into`. The panic is
    /// considered acceptable because the size checks are performed just
    /// before the conversion, and a mismatch would indicate a corrupted
    /// GGUF file.
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if the tensor size is not a multiple
    /// of the element size, or if the dtype is unsupported.
    pub fn dequantize(&self, raw: &[u8]) -> Result<Vec<f32>, GgufError> {
        match self.dtype {
            GgmlType::F32 => {
                if raw.len() % 4 != 0 {
                    return Err(GgufError::DecodeError(
                        "F32 tensor size not multiple of 4".into(),
                    ));
                }
                let num_elements = raw.len() / 4;

                if num_elements > 65536 {
                    use rayon::prelude::*;
                    let mut out = vec![0.0f32; num_elements];
                    out.par_chunks_mut(1024)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let start = chunk_idx * 1024 * 4;
                            for (i, out_val) in chunk.iter_mut().enumerate() {
                                let byte_idx = start + i * 4;
                                // SAFETY: chunks_exact(4) guarantees 4 bytes available
                                *out_val = f32::from_le_bytes(
                                    raw[byte_idx..byte_idx + 4].try_into().expect("chunks_exact(4) verified"),
                                );
                            }
                        });
                    Ok(out)
                } else {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in raw.chunks_exact(4) {
                        // SAFETY: chunks_exact(4) guarantees chunk.len() == 4
                        let v = f32::from_le_bytes(chunk.try_into().expect("chunks_exact(4) verified"));
                        out.push(v);
                    }
                    Ok(out)
                }
            }
            GgmlType::F16 => {
                if raw.len() % 2 != 0 {
                    return Err(GgufError::DecodeError(
                        "F16 tensor size not multiple of 2".into(),
                    ));
                }
                let num_elements = raw.len() / 2;

                if num_elements > 65536 {
                    use rayon::prelude::*;
                    let mut out = vec![0.0f32; num_elements];
                    out.par_chunks_mut(1024)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let start = chunk_idx * 1024 * 2;
                            for (i, out_val) in chunk.iter_mut().enumerate() {
                                let byte_idx = start + i * 2;
                                // SAFETY: chunks_exact(2) guarantees 2 bytes available
                                let bits = u16::from_le_bytes(
                                    raw[byte_idx..byte_idx + 2].try_into().expect("chunks_exact(2) verified"),
                                );
                                *out_val = half::f16::from_bits(bits).to_f32();
                            }
                        });
                    Ok(out)
                } else {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in raw.chunks_exact(2) {
                        // SAFETY: chunks_exact(2) guarantees chunk.len() == 2
                        let bits = u16::from_le_bytes(chunk.try_into().expect("chunks_exact(2) verified"));
                        let f: f32 = half::f16::from_bits(bits).to_f32();
                        out.push(f);
                    }
                    Ok(out)
                }
            }
            GgmlType::Q4_0 => dequantize_q4_0(raw),
            GgmlType::Q4_1 => dequantize_q4_1(raw),
            GgmlType::Q5_0 => dequantize_q5_0(raw),
            GgmlType::Q5_1 => dequantize_q5_1(raw),
            GgmlType::Q8_0 => dequantize_q8_0(raw),
            GgmlType::Q2_K => dequantize_q2_k(raw),
            GgmlType::Q3_K => dequantize_q3_k(raw),
            GgmlType::Q4_K => dequantize_q4_k(raw),
            GgmlType::Q5_K => dequantize_q5_k(raw),
            GgmlType::Q6_K => dequantize_q6_k(raw),
            _ => Err(GgufError::DecodeError(format!(
                "unsupported dtype for dequantize: {:?}",
                self.dtype
            ))),
        }
    }
}

// ─── Mmap Tensor ─────────────────────────────────────────────────────────────

/// Memory-mapped tensor reference — holds a shared mmap plus offset/size.
/// Enables lazy loading: tensor data is only accessed from the mmap on demand,
/// letting the OS page in only the needed regions.
#[derive(Debug, Clone)]
pub struct MmapTensor {
    /// Shared reference to the memory-mapped file.
    pub mmap: Arc<memmap2::Mmap>,
    /// Byte offset within the mmap where this tensor's data starts.
    pub offset: usize,
    /// Size of the tensor data in bytes.
    pub size: usize,
}

impl MmapTensor {
    /// Create a new memory-mapped tensor reference.
    #[must_use]
    pub fn new(mmap: Arc<memmap2::Mmap>, offset: usize, size: usize) -> Self {
        Self { mmap, offset, size }
    }

    /// Get a slice of the raw tensor data from the mmap.
    pub fn as_slice(&self) -> Result<&[u8], GgufError> {
        let end = self.offset + self.size;
        if end > self.mmap.len() {
            return Err(GgufError::DecodeError(format!(
                "tensor data extends beyond mmap (need {end}, have {})",
                self.mmap.len()
            )));
        }
        Ok(&self.mmap[self.offset..end])
    }

    /// Dequantize the tensor data directly from the mmap.
    pub fn dequantize(&self, info: &TensorInfo) -> Result<Vec<f32>, GgufError> {
        info.dequantize(self.as_slice()?)
    }
}
