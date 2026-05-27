//! Tensor info and memory-mapped tensor reference.
//!
//! This module re‑exports [`TensorInfo`] and [`MmapTensor`] from `llama_core`
//! and provides GGUF‑specific de‑quantization functionality via an extension
//! trait and free functions.

pub use llama_core::{MmapTensor, TensorInfo};

use crate::dequant::*;
use crate::{GgmlType, GgufError};

/// Extension trait adding GGUF-specific dequantization methods to [`TensorInfo`].
///
/// Defined here instead of on `TensorInfo` directly to avoid Rust's orphan rule
/// (`TensorInfo` is defined in `llama_core` while our error types are in `gguf`).
pub trait TensorDequantExt {
    /// De-quantize the raw tensor bytes into a `Vec<f32>`.
    ///
    /// Supports F32, F16, and common quantization types (Q4_0 through IQ4_XS).
    ///
    /// # Errors
    ///
    /// Returns a [`GgufError::DecodeError`] if the data is malformed or the
    /// dtype is unsupported.
    fn dequantize(&self, raw: &[u8]) -> Result<Vec<f32>, GgufError>;
}

impl TensorDequantExt for TensorInfo {
    fn dequantize(&self, raw: &[u8]) -> Result<Vec<f32>, GgufError> {
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
                                *out_val = f32::from_le_bytes(
                                    raw[byte_idx..byte_idx + 4]
                                        .try_into()
                                        .expect("chunks_exact(4) verified"),
                                );
                            }
                        });
                    Ok(out)
                } else {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in raw.chunks_exact(4) {
                        let v =
                            f32::from_le_bytes(chunk.try_into().expect("chunks_exact(4) verified"));
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
                                let bits = u16::from_le_bytes(
                                    raw[byte_idx..byte_idx + 2]
                                        .try_into()
                                        .expect("chunks_exact(2) verified"),
                                );
                                *out_val = half::f16::from_bits(bits).to_f32();
                            }
                        });
                    Ok(out)
                } else {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in raw.chunks_exact(2) {
                        let bits =
                            u16::from_le_bytes(chunk.try_into().expect("chunks_exact(2) verified"));
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
            GgmlType::Q8_1 => dequantize_q8_1(raw),
            GgmlType::Q2_K => dequantize_q2_k(raw),
            GgmlType::Q3_K => dequantize_q3_k(raw),
            GgmlType::Q4_K => dequantize_q4_k(raw),
            GgmlType::Q5_K => dequantize_q5_k(raw),
            GgmlType::Q6_K => dequantize_q6_k(raw),
            GgmlType::Q8_K => dequantize_q8_k(raw),
            GgmlType::Q1_0 => dequantize_q1_0(raw),
            GgmlType::Tq1_0 => dequantize_tq1_0(raw),
            GgmlType::Tq2_0 => dequantize_tq2_0(raw),
            GgmlType::Mxfp4 => dequantize_mxfp4(raw),
            GgmlType::Nvfp4 => dequantize_nvfp4(raw),
            GgmlType::Iq1S => dequantize_iq1_s(raw),
            GgmlType::Iq1M => dequantize_iq1_m(raw),
            GgmlType::Iq2S => dequantize_iq2_s(raw),
            GgmlType::Iq2Xxs => dequantize_iq2_xxs(raw),
            GgmlType::Iq2Xs => dequantize_iq2_xs(raw),
            GgmlType::Iq3S => dequantize_iq3_s(raw),
            GgmlType::Iq3Xxs => dequantize_iq3_xxs(raw),
            GgmlType::Iq3Xs => dequantize_iq3_xs(raw),
            GgmlType::Iq3M => dequantize_iq3_m(raw),
            GgmlType::Iq4Nl => dequantize_iq4_nl(raw),
            GgmlType::Iq4Xs => dequantize_iq4_xs(raw),
            _ => Err(GgufError::DecodeError(format!(
                "unsupported dtype for dequantize: {:?}",
                self.dtype
            ))),
        }
    }
}

/// De-quantize a memory-mapped tensor using its `TensorInfo`.
///
/// This is a convenience function that calls `TensorDequantExt::dequantize` on
/// the mmap tensor's raw data slice.
///
/// # Errors
///
/// Returns [`GgufError`] if the tensor data cannot be read or de-quantized.
pub fn mmap_tensor_dequantize(
    mmap: &MmapTensor,
    info: &TensorInfo,
) -> Result<Vec<f32>, GgufError> {
    use TensorDequantExt as _;
    info.dequantize(mmap.as_slice().map_err(|e| {
        GgufError::DecodeError(format!("mmap bounds exceeded: {e}"))
    })?)
}
