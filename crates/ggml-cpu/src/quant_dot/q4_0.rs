//! `Q4_0` quantized dot product.
//!
//! Block layout: [f16 scale (2 bytes)] + [16 bytes of 4-bit nibbles] = 18 bytes for 32 f32 values.
//! Each nibble encodes a signed 4-bit value: 0 → -8, 1 → -7, ..., 15 → 7.
//! Byte `i` stores nibble for `value[2*i]` in low 4 bits, `value[2*i+1]` in high 4 bits.

use crate::quant_dot::QuantDot;
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
            let byte = quantized[2 + i];
            // Nibble values are 0..15, always safe for i8
            #[expect(clippy::cast_possible_wrap)]
            let lo = (byte & 0x0F) as i8;
            #[expect(clippy::cast_possible_wrap)]
            let hi = (byte >> 4) as i8;
            // Q4_0: signed 4-bit, values are -8..7, but stored as unsigned 0..15
            // So we subtract 8 to get the actual value: 0→-8, 1→-7, ..., 15→7
            // Safety: nibble values 0..15 fit in i8
            let lo_val = f32::from(lo.wrapping_sub(8));
            let hi_val = f32::from(hi.wrapping_sub(8));
            sum += lo_val * input[i * 2];
            sum += hi_val * input[i * 2 + 1];
        }
        sum * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_dot::QuantDot;

    /// Build a Q4_0 block from a scale and 32 signed values.
    /// Values are clamped to [-8, 7].
    fn build_block(scale: f32, values: &[i8]) -> Vec<u8> {
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
    fn test_q4_0_dot_block_simple() {
        let kernel = Q4_0Dot;
        // scale=1.0, all nibbles = 0x1 → value = 1-8 = -7
        let block = build_block(1.0, &[-7i8; 32]);
        assert_eq!(block.len(), 18);
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 32 * (-7 * 1.0) = -224
        assert!(
            (result - (-224.0)).abs() < 1e-3,
            "expected -224, got {result}"
        );
    }

    #[test]
    fn test_q4_0_dot_block_positive_values() {
        let kernel = Q4_0Dot;
        // scale=2.0, all values = 7 (max positive for 4-bit signed)
        let block = build_block(2.0, &[7i8; 32]);
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        // sum = 32 * (7 * 2.0 * 1.0) = 448
        assert!((result - 448.0).abs() < 1e-3, "expected 448, got {result}");
    }

    #[test]
    fn test_q4_0_dot_block_zero_input() {
        let kernel = Q4_0Dot;
        let block = build_block(1.0, &[1i8; 32]);
        let input = [0.0f32; 32];
        let result = kernel.dot_block(&block, &input);
        assert!((result - 0.0).abs() < 1e-6, "expected 0, got {result}");
    }

    #[test]
    fn test_q4_0_dot_block_varied_values() {
        let kernel = Q4_0Dot;
        let values: Vec<i8> = (0..32).map(|i| ((i as i8) % 16).wrapping_sub(8)).collect();
        let block = build_block(1.5, &values);
        let input: [f32; 32] = std::array::from_fn(|i| i as f32 * 0.5);

        // Reference: sum(values[i] * scale * input[i])
        let expected: f32 = values
            .iter()
            .zip(input.iter())
            .map(|(v, inp)| (*v as f32) * 1.5 * inp)
            .sum();
        let result = kernel.dot_block(&block, &input);
        assert!(
            (result - expected).abs() < 1e-3,
            "expected {expected}, got {result}"
        );
    }
}
