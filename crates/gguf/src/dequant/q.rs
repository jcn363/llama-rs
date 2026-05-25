//! Standard quantized type dequantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1).

use rayon::prelude::*;

use crate::GgufError;

use super::u16_from_le;

/// ``Q4_0``: 4-bit quantization, variant 0.
/// Block size: 32 elements.
/// Layout: [d: f16][qs: 16 bytes]
/// Each byte in qs contains 2 4-bit values: `qs[i] & 0xF`, `qs[i] >> 4`
pub fn dequantize_q4_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK4_0: usize = 32;
    const BLOCK_SIZE: usize = 2 + 16; // d (2 bytes) + qs (16 bytes)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q4_0 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;

    // Parallel for large tensors (> 64K elements = 2048 blocks)
    if num_blocks > 2048 {
        let mut out = vec![0.0f32; num_blocks * QK4_0];

        out.par_chunks_mut(1024)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_block = (chunk_idx * 1024) / QK4_0;
                let end_block = ((chunk_idx + 1) * 1024) / QK4_0;

                for block_idx in start_block..end_block.min(num_blocks) {
                    let block_start = block_idx * BLOCK_SIZE;
                    let block = &raw[block_start..block_start + BLOCK_SIZE];
                    let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
                    let qs = &block[2..18];

                    let out_start = block_idx * QK4_0 - start_block * QK4_0;
                    for i in 0..16 {
                        let v0 = (qs[i] & 0x0F) as i8 - 8;
                        let v1 = (qs[i] >> 4) as i8 - 8;
                        if out_start + i * 2 < chunk.len() {
                            chunk[out_start + i * 2] = v0 as f32 * d;
                        }
                        if out_start + i * 2 + 1 < chunk.len() {
                            chunk[out_start + i * 2 + 1] = v1 as f32 * d;
                        }
                    }
                }
            });

        Ok(out)
    } else {
        let mut out = Vec::with_capacity(num_blocks * QK4_0);

        for block in raw.chunks_exact(BLOCK_SIZE) {
            let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
            let qs = &block[2..18];

            for &q in qs.iter().take(16) {
                let v0 = (q & 0x0F) as i8 - 8;
                let v1 = (q >> 4) as i8 - 8;
                out.push(v0 as f32 * d);
                out.push(v1 as f32 * d);
            }
        }

        Ok(out)
    }
}

/// ``Q4_1``: 4-bit quantization, variant 1.
/// Block size: 32 elements.
/// Layout: [d: f16][m: f16][qs: 16 bytes]
/// value = d * qs + m
pub fn dequantize_q4_1(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK4_1: usize = 32;
    const BLOCK_SIZE: usize = 2 + 2 + 16; // d (2) + m (2) + qs (16)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q4_1 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK4_1);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let m = half::f16::from_bits(u16_from_le(&block[2..4])).to_f32();
        let qs = &block[4..20];

        for &q in qs.iter().take(16) {
            let v0 = (q & 0x0F) as f32;
            let v1 = (q >> 4) as f32;
            out.push(v0 * d + m);
            out.push(v1 * d + m);
        }
    }

    Ok(out)
}

/// ``Q5_0``: 5-bit quantization, variant 0.
/// Block size: 32 elements.
/// Layout: [d: f16][qh: 4 bytes][qs: 16 bytes]
pub fn dequantize_q5_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK5_0: usize = 32;
    const BLOCK_SIZE: usize = 2 + 4 + 16; // d (2) + qh (4) + qs (16)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q5_0 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK5_0);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qh = &block[2..6];
        let qs = &block[6..22];

        for i in 0..16 {
            let h0 = ((qh[i / 4] >> ((i % 4) * 2)) & 0x01) as i8;
            let h1 = ((qh[i / 4] >> ((i % 4) * 2 + 1)) & 0x01) as i8;

            let v0 = ((qs[i] & 0x0F) | ((h0 as u8) << 4)) as i8 - 16;
            let v1 = ((qs[i] >> 4) | ((h1 as u8) << 4)) as i8 - 16;
            out.push(v0 as f32 * d);
            out.push(v1 as f32 * d);
        }
    }

    Ok(out)
}

/// ``Q5_1``: 5-bit quantization, variant 1.
/// Block size: 32 elements.
/// Layout: [d: f16][m: f16][qh: 4 bytes][qs: 16 bytes]
pub fn dequantize_q5_1(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK5_1: usize = 32;
    const BLOCK_SIZE: usize = 2 + 2 + 4 + 16; // d (2) + m (2) + qh (4) + qs (16)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q5_1 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK5_1);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let m = half::f16::from_bits(u16_from_le(&block[2..4])).to_f32();
        let qh = &block[4..8];
        let qs = &block[8..24];

        for i in 0..16 {
            let h0 = (qh[i / 4] >> ((i % 4) * 2)) & 0x01;
            let h1 = (qh[i / 4] >> ((i % 4) * 2 + 1)) & 0x01;

            let v0 = ((qs[i] & 0x0F) | (h0 << 4)) as f32;
            let v1 = ((qs[i] >> 4) | (h1 << 4)) as f32;
            out.push(v0 * d + m);
            out.push(v1 * d + m);
        }
    }

    Ok(out)
}

/// ``Q8_0``: 8-bit quantization, variant 0.
/// Block size: 32 elements.
/// Layout: [d: f16][qs: 32 bytes]
pub fn dequantize_q8_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK8_0: usize = 32;
    const BLOCK_SIZE: usize = 2 + 32; // d (2) + qs (32)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q8_0 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK8_0);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[2..34];

        for &q in qs {
            out.push(q as i8 as f32 * d);
        }
    }

    Ok(out)
}

/// ``Q8_1``: 8-bit quantization, variant 1.
/// Block size: 32 elements.
/// Layout: [d: f16][s: f16][qs: 32 bytes] = 36 bytes
pub fn dequantize_q8_1(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK8_1: usize = 32;
    const BLOCK_SIZE: usize = 2 + 2 + 32; // d (2) + s (2) + qs (32)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q8_1 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK8_1);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[4..36];

        for &q in qs {
            out.push(q as i8 as f32 * d);
        }
    }

    Ok(out)
}
