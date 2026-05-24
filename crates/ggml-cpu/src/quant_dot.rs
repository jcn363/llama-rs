//! Quantized dot product kernels.
//!
//! Each quantized format has a dot product that computes the inner product
//! of a quantized weight vector and an f32 input vector WITHOUT
//! dequantizing the entire weight vector first.
//!
//! This gives 2-4x throughput improvement vs dequantize-then-compute.

/// Trait for quantized dot product implementations.
/// `block_size` is the number of f32 values per quantized block.
/// E.g., `Q4_0` has `block_size=32` (each block stores 32 4-bit weights + scale).
pub trait QuantDot: Send + Sync {
    /// Dot product of one quantized block with the corresponding f32 input.
    /// `quantized` is the block's raw bytes (length = `block_bytes()`),
    /// `input` is `&[f32; block_size()]`.
    fn dot_block(&self, quantized: &[u8], input: &[f32]) -> f32;

    /// Number of f32 elements per quantized block.
    fn block_size(&self) -> usize;

    /// Size of the quantized block in bytes.
    fn block_bytes(&self) -> usize;
}

/// Compute dot product of a quantized weight row with an f32 input vector.
/// Iterates over blocks, calls the kernel for each, sums results.
/// Handles the tail (partial last block) by padding the input with zeros.
pub fn quant_dot_row<T: QuantDot>(
    kernel: &T,
    quantized_row: &[u8],
    input: &[f32],
    cols: usize,
) -> f32 {
    let block_size = kernel.block_size();
    let block_bytes = kernel.block_bytes();
    let n_blocks = cols.div_ceil(block_size);
    let mut sum = 0.0f32;
    for b in 0..n_blocks {
        let q_start = b * block_bytes;
        let i_start = b * block_size;
        let q_block = &quantized_row[q_start..q_start + block_bytes];
        // For the tail block, pad input with zeros
        let i_end = cols.min(i_start + block_size);
        let mut padded = [0.0f32; 32]; // max block_size across all formats
        let actual = &input[i_start..i_end];
        padded[..actual.len()].copy_from_slice(actual);
        sum += kernel.dot_block(q_block, &padded[..block_size]);
    }
    sum
}

pub mod q4_0;
pub mod q4_1;
pub mod q8_0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_dot::q4_0::Q4_0Dot;
    use crate::simd;
    use half::f16;

    fn build_q4_0_block(scale: f32, values: &[i8]) -> Vec<u8> {
        assert_eq!(values.len(), 32);
        let scale_bytes = f16::from_f32(scale).to_le_bytes();
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&scale_bytes);
        for i in 0..16 {
            let lo = (values[i * 2].wrapping_add(8) & 0x0F) as u8;
            let hi = ((values[i * 2 + 1].wrapping_add(8) & 0x0F) as u8) << 4;
            block.push(lo | hi);
        }
        block
    }

    #[test]
    fn test_quant_dot_row_matches_dequant_dot() {
        let kernel = Q4_0Dot;
        let n_blocks = 4;
        let cols = n_blocks * 32;

        // Build a quantized weight row: 4 Q4_0 blocks
        // Values must be in Q4_0's [-8, 7] range
        let mut quantized = Vec::new();
        let mut dequant = Vec::with_capacity(cols);
        for b in 0..n_blocks {
            let scale = 1.0 + b as f32 * 0.5;
            let values: Vec<i8> = (0..32).map(|i| ((i as i8) % 16) - 8).collect();
            quantized.extend_from_slice(&build_q4_0_block(scale, &values));
            // Dequantize for reference
            for v in &values {
                dequant.push(*v as f32 * scale);
            }
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();

        let quant_result = quant_dot_row(&kernel, &quantized, &input, cols);
        let dequant_result = simd::dot_f32(&dequant, &input);

        assert!(
            (quant_result - dequant_result).abs() < 1e-3,
            "quant={quant_result} vs dequant={dequant_result}"
        );
    }

    #[test]
    fn test_quant_dot_row_tail_block() {
        // Non-multiple-of-block-size cols (e.g., 40 f32 values = 1 full block + 8 tail)
        // Need 2 Q4_0 blocks (36 bytes) to cover 40 values
        let kernel = Q4_0Dot;
        let cols = 40;
        let mut quantized = Vec::new();
        quantized.extend_from_slice(&build_q4_0_block(1.0, &[1i8; 32]));
        quantized.extend_from_slice(&build_q4_0_block(1.0, &[2i8; 32]));

        let input: Vec<f32> = (0..cols).map(|i| i as f32 * 0.1).collect();

        let result = quant_dot_row(&kernel, &quantized, &input, cols);
        // Block 0: 32 values of 1.0 (nibble 0x9 decodes to 1), Block 1: 32 values of 2.0
        // but only input[32..40] (8 values) contribute to block 1, tail zeros for rest
        let block0: f32 = (0..32).map(|i| 1.0f32 * (i as f32 * 0.1)).sum();
        let block1: f32 = (0..8).map(|i| 2.0f32 * ((32 + i) as f32 * 0.1)).sum();
        let expected = block0 + block1;
        assert!(
            (result - expected).abs() < 1e-3,
            "got {result} expected {expected}"
        );
    }
}
