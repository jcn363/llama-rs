//! `Q8_0` quantized dot product.
//!
//! Block layout: [f16 scale (2 bytes)] + [32 × i8 weights (32 bytes)] = 34 bytes for 32 f32 values.
//! Each weight byte is the actual signed integer value.
//! Dot product: `scale * Σ(weight[i] * input[i])`

use crate::quant_dot::QuantDot;
use half::f16;

/// `Q8_0` quantized dot product kernel.
pub struct Q8_0Dot;

impl QuantDot for Q8_0Dot {
    #[inline]
    fn block_size(&self) -> usize {
        32
    }

    #[inline]
    fn block_bytes(&self) -> usize {
        34
    }

    #[inline]
    fn dot_block(&self, quantized: &[u8], input: &[f32]) -> f32 {
        let scale = f16::from_le_bytes([quantized[0], quantized[1]]).to_f32();
        let mut sum = 0.0f32;
        for i in 0..32 {
            // Nibble values are -128..127, always safe for i8
            #[expect(clippy::cast_possible_wrap)]
            let w = quantized[2 + i] as i8;
            sum += f32::from(w) * input[i];
        }
        sum * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_dot::QuantDot;

    /// Build a Q8_0 block from a scale and 32 signed values.
    #[cfg(feature = "simd")]
    fn build_block(scale: f32, values: &[i8]) -> [u8; 34] {
        // Buffer reuse with fixed-size array; SIMD could be applied here in the future.
        debug_assert_eq!(values.len(), 32);
        let scale_bytes = f16::from_f32(scale).to_le_bytes();
        let mut block = [0u8; 34];
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        for (i, v) in values.iter().enumerate() {
            block[2 + i] = *v as u8;
        }
        block
    }

    #[cfg(not(feature = "simd"))]
    fn build_block(scale: f32, values: &[i8]) -> Vec<u8> {
        debug_assert_eq!(values.len(), 32);
        let scale_bytes = f16::from_f32(scale).to_le_bytes();
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale_bytes);
        for v in values {
            block.push(*v as u8);
        }
        block
    }

    #[test]
    fn test_q8_0_dot_block_simple() {
        let kernel = Q8_0Dot;
        let block = build_block(1.0, &[3i8; 32]);
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 32 * (3 * 1.0) = 96
        assert!((result - 96.0).abs() < 1e-3, "expected 96, got {result}");
    }

    #[test]
    fn test_q8_0_dot_block_negative() {
        let kernel = Q8_0Dot;
        let block = build_block(2.0, &[-5i8; 32]);
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 32 * (-5 * 2.0) = -320
        assert!(
            (result - (-320.0)).abs() < 1e-3,
            "expected -320, got {result}"
        );
    }

    #[test]
    fn test_q8_0_dot_block_zero_input() {
        let kernel = Q8_0Dot;
        let block = build_block(1.0, &[7i8; 32]);
        let input = [0.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        assert!((result - 0.0).abs() < 1e-6, "expected 0, got {result}");
    }

    #[test]
    fn test_q8_0_dot_block_varied() {
        let kernel = Q8_0Dot;
        let values: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
        let block = build_block(1.5, &values);
        let input: [f32; 32] = std::array::from_fn(|i| i as f32 * 0.5);

        let expected: f32 = values
            .iter()
            .zip(input.iter())
            .map(|(v, inp)| f32::from(*v) * 1.5 * inp)
            .sum();
        let result = kernel.dot_block(&block, &input);
        assert!(
            (result - expected).abs() < 1e-3,
            "expected {expected}, got {result}"
        );
    }
}
