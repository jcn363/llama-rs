//! `Q4_0` quantized dot product.
//!
//! Block layout: [f16 scale (2 bytes)] + [16 bytes of 4-bit nibbles] = 18 bytes for 32 f32 values.
//! Each nibble encodes a signed 4-bit value: 0 → -8, 1 → -7, ..., 15 → 7.
//! Byte `i` stores nibble for `value[2*i]` in low 4 bits, `value[2*i+1]` in high 4 bits.

use crate::quant_dot::{unpack_nibbles, QuantDot};
use half::f16;

/// `Q4_0` quantized dot product kernel.
pub struct Q4_0Dot;

impl QuantDot for Q4_0Dot {
    #[inline]
    fn block_size(&self) -> usize {
        32
    }

    #[inline]
    fn block_bytes(&self) -> usize {
        18
    }

    #[inline]
    fn dot_block(&self, quantized: &[u8], input: &[f32]) -> f32 {
        let scale = f16::from_le_bytes([quantized[0], quantized[1]]).to_f32();
        let mut sum = 0.0f32;
        for i in 0..16 {
            let (lo, hi) = unpack_nibbles(quantized[2 + i]);
            // Q4_0: signed 4-bit, values stored as unsigned 0..15, meaning -8..7
            // Subtract 8 to get the actual value: 0→-8, 1→-7, ..., 15→7
            sum += f32::from(lo.wrapping_sub(8)) * input[i * 2];
            sum += f32::from(hi.wrapping_sub(8)) * input[i * 2 + 1];
        }
        sum * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_dot::test_utils::*;
    use crate::quant_dot::QuantDot;

    /// Build a Q4_0 block from a scale and 32 signed values.
    #[cfg(feature = "simd")]
    fn build_block(scale: f32, values: &[i8]) -> [u8; 18] {
        debug_assert_eq!(values.len(), 32);
        let mut block = [0u8; 18];
        write_f16_scale(&mut block[..2], scale);
        fill_nibble_block(&mut block, 2, values, true);
        block
    }

    #[cfg(not(feature = "simd"))]
    fn build_block(scale: f32, values: &[i8]) -> Vec<u8> {
        debug_assert_eq!(values.len(), 32);
        let mut block = vec![0u8; 18];
        write_f16_scale(&mut block[..2], scale);
        fill_nibble_block(&mut block, 2, values, true);
        block
    }

    #[test]
    fn test_q4_0_dot_block_positive_values() {
        let kernel = Q4_0Dot;
        let block = build_block(2.0, &[7i8; 32]);
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 32 * (7 * 2.0 * 1.0) = 448
        assert_close(result, 448.0, 1e-3);
    }

    #[test]
    fn test_q4_0_dot_block_zero_input() {
        let kernel = Q4_0Dot;
        let block = build_block(1.0, &[1i8; 32]);
        let input = [0.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        assert_close(result, 0.0, 1e-6);
    }

    #[test]
    fn test_q4_0_dot_block_varied_values() {
        let kernel = Q4_0Dot;
        let values: Vec<i8> = (0..32).map(|i| ((i as i8) % 16).wrapping_sub(8)).collect();
        let block = build_block(1.5, &values);
        let input: [f32; 32] = std::array::from_fn(|i| i as f32 * 0.5);

        let expected: f32 = values
            .iter()
            .zip(input.iter())
            .map(|(v, inp)| (*v as f32) * 1.5 * inp)
            .sum();
        let result = kernel.dot_block(&block, &input);
        assert_close(result, expected, 1e-3);
    }
}
