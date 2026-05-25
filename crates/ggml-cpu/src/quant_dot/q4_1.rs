//! `Q4_1` quantized dot product.
//!
//! Block layout: [f16 scale (2 bytes)] + [f16 min (2 bytes)] + [16 bytes of 4-bit nibbles] = 20 bytes for 32 f32 values.
//! Each nibble is an unsigned 4-bit value (0..15).
//! Dequant: value * d + m  (where d = scale, m = min).
//! Dot product: `d * Σ(nibble[i] * input[i]) + m * Σ(input[i])`

use crate::quant_dot::QuantDot;
use half::f16;

/// `Q4_1` quantized dot product kernel.
pub struct Q4_1Dot;

impl QuantDot for Q4_1Dot {
    #[inline]
    fn block_size(&self) -> usize {
        32
    }

    #[inline]
    fn block_bytes(&self) -> usize {
        20
    }

    #[inline]
    fn dot_block(&self, quantized: &[u8], input: &[f32]) -> f32 {
        let d = f16::from_le_bytes([quantized[0], quantized[1]]).to_f32();
        let m = f16::from_le_bytes([quantized[2], quantized[3]]).to_f32();
        let mut sum_nibble_input = 0.0f32;
        let mut sum_input = 0.0f32;
        for i in 0..16 {
            let byte = quantized[4 + i];
            // Nibble values are 0..15, always safe for i8
            #[expect(clippy::cast_possible_wrap)]
            let lo = (byte & 0x0F) as i8;
            #[expect(clippy::cast_possible_wrap)]
            let hi = (byte >> 4) as i8;
            sum_nibble_input += f32::from(lo) * input[i * 2];
            sum_nibble_input += f32::from(hi) * input[i * 2 + 1];
            sum_input += input[i * 2] + input[i * 2 + 1];
        }
        d * sum_nibble_input + m * sum_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_dot::QuantDot;

    /// Build a Q4_1 block from scale d, min m, and 32 unsigned nibble values (0..15).
    fn build_block(d: f32, m: f32, values: &[i8]) -> Vec<u8> {
        assert_eq!(values.len(), 32);
        let d_bytes = f16::from_f32(d).to_le_bytes();
        let m_bytes = f16::from_f32(m).to_le_bytes();
        let mut block = Vec::with_capacity(20);
        block.extend_from_slice(&d_bytes);
        block.extend_from_slice(&m_bytes);
        for i in 0..16 {
            let lo = (values[i * 2] & 0x0F) as u8;
            let hi = ((values[i * 2 + 1] & 0x0F) as u8) << 4;
            block.push(lo | hi);
        }
        block
    }

    #[test]
    fn test_q4_1_dot_block_simple() {
        let kernel = Q4_1Dot;
        // d=1.0, m=0.0, all nibbles = 1
        let block = build_block(1.0, 0.0, &[1i8; 32]);
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 1.0 * 32 * 1 + 0.0 * 32 = 32
        assert!((result - 32.0).abs() < 1e-3, "expected 32, got {result}");
    }

    #[test]
    fn test_q4_1_dot_block_with_min() {
        let kernel = Q4_1Dot;
        // d=2.0, m=1.0, all nibbles = 3
        let block = build_block(2.0, 1.0, &[3i8; 32]);
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 2.0 * (32 * 3) + 1.0 * 32 = 192 + 32 = 224
        assert!((result - 224.0).abs() < 1e-3, "expected 224, got {result}");
    }

    #[test]
    fn test_q4_1_dot_block_zero_input() {
        let kernel = Q4_1Dot;
        let block = build_block(1.0, 0.5, &[7i8; 32]);
        let input = [0.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 1.0 * 0 + 0.5 * 0 = 0
        assert!((result - 0.0).abs() < 1e-6, "expected 0, got {result}");
    }

    #[test]
    fn test_q4_1_dot_block_varied() {
        let kernel = Q4_1Dot;
        let values: Vec<i8> = (0..32).map(|i| (i as i8) % 16).collect();
        let block = build_block(1.5, -0.5, &values);
        let input: [f32; 32] = std::array::from_fn(|i| i as f32 * 0.5);

        // Reference: sum((nibble * d + m) * input[i])
        let expected: f32 = values
            .iter()
            .zip(input.iter())
            .map(|(v, inp)| (f32::from(*v) * 1.5 + (-0.5)) * inp)
            .sum();
        let result = kernel.dot_block(&block, &input);
        assert!(
            (result - expected).abs() < 1e-3,
            "expected {expected}, got {result}"
        );
    }
}
