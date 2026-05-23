//! Dequantization helper functions for GGUF tensors.
//!
//! This module contains the implementations of the various quantized
//! tensor dequantization algorithms that were previously embedded in
//! `lib.rs`.  Keeping them in a dedicated file improves readability
//! and reduces the size of the main library file.

use super::GgufError;
use rayon::prelude::*;

/// Infallibly read a `u16` from a two-byte slice (CPU‑local data, known size).
#[inline(always)]
fn u16_from_le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

// ──────────────────────────────────────────────────────────────────────────────
// Dequantization Functions
// ──────────────────────────────────────────────────────────────────────────────

/// ``Q4_0``: 4-bit quantization, variant 0.
/// Block size: 32 elements.
/// Layout: [d: f16][qs: 16 bytes]
/// Each byte in qs contains 2 4-bit values: qs[i] & 0xF, qs[i] >> 4
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

/// ``Q8_K``: 8-bit K-quant.
/// Block size: 256 elements.
/// Layout: [scales: 16][qs: 128][ql: 64][qh: 64][d: f16] = 288 bytes
pub fn dequantize_q8_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 16 + 128 + 64 + 64 + 2; // scales + qs + ql + qh + d
    
    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q8_K tensor size not multiple of block size".into(),
        ));
    }
    
    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);
    
    for block in raw.chunks_exact(BLOCK_SIZE) {
        let scales = &block[0..16];
        let qs = &block[16..144]; // 128 bytes
        let _ql = &block[144..208]; // 64 bytes
        let _qh = &block[208..272]; // 64 bytes
        let d = half::f16::from_bits(u16_from_le(&block[272..274])).to_f32();
        
        // Process 8-bit values (qs contains the 8-bit quantized values)
        for i in 0..QK_K {
            let q = qs[i] as i8 as f32;
            let scale_idx = i / 16;
            let scale = scales[scale_idx] as f32 / 32.0; // Scale factor for Q8_K
            out.push(d * q * scale);
        }
    }
    
    Ok(out)
}

/// ``Q1_0``: 1-bit quantization.
/// Block size: 256 elements.
/// Layout: [scales: 32][qs: 32] = 64 bytes
pub fn dequantize_q1_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK1_0: usize = 256;
    const BLOCK_SIZE: usize = 32 + 32; // scales + qs
    
    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q1_0 tensor size not multiple of block size".into(),
        ));
    }
    
    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK1_0);
    
    for block in raw.chunks_exact(BLOCK_SIZE) {
        let scales = &block[0..32];
        let qs = &block[32..64];
        let d = 1.0f32; // No explicit d factor in Q1_0 layout, assume 1.0
        
        for i in 0..QK1_0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let q = ((qs[byte_idx] >> bit_idx) & 1) as f32 * 2.0 - 1.0; // Convert 0/1 to -1/+1
            let scale = scales[byte_idx] as f32 / 32.0;
            out.push(d * q * scale);
        }
    }
    
    Ok(out)
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
/// Layout: [d: f16][qs: 32 bytes] (same as Q8_0 but different interpretation)
pub fn dequantize_q8_1(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK8_1: usize = 32;
    const BLOCK_SIZE: usize = 2 + 32; // d (2) + qs (32)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q8_1 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK8_1);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[2..34];

        for &q in qs {
            // Q8_1 uses unsigned 8-bit values (0-255) mapped to [-1, 1] range
            let val = q as f32 / 255.0 * 2.0 - 1.0;
            out.push(val * d);
        }
    }

    Ok(out)
}



/// `Q2_K`: 2-bit K-quant.
/// Block size: 256 elements.
/// Layout: [scales: 16][qs: 64][d: f16][dmin: f16] = 84 bytes
pub fn dequantize_q2_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 16 + 64 + 2 + 2; // scales + qs + d + dmin

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q2_K tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = half::f16::from_bits(u16_from_le(&block[80..82])).to_f32();
        let min = half::f16::from_bits(u16_from_le(&block[82..84])).to_f32();

        let mut is = 0;
        let mut q_offset = 0;
        for _n in 0..2 {
            let mut shift = 0;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let q = ((qs[q_offset + l] >> shift) & 3) as f32;
                    out.push(dl * q - ml);
                }

                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let q = ((qs[q_offset + l + 16] >> shift) & 3) as f32;
                    out.push(dl * q - ml);
                }

                shift += 2;
            }
            q_offset += 32;
        }
    }

    Ok(out)
}

/// `Q3_K`: 3-bit K-quant.
/// Block size: 256 elements.
/// Layout: [hmask: 32][qs: 64][scales: 12][d: f16] = 110 bytes
pub fn dequantize_q3_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 32 + 64 + 12 + 2; // hmask + qs + scales + d

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q3_K tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d_all = half::f16::from_bits(u16_from_le(&block[108..110])).to_f32();

        // Unpack scales from 12 bytes into 16 signed scale values
        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;

        let mut aux: [u32; 4] = [
            u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]),
            u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]),
            u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]),
            0,
        ];
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        let scales: [i8; 16] = [
            aux[0] as i8,
            (aux[0] >> 8) as i8,
            (aux[0] >> 16) as i8,
            (aux[0] >> 24) as i8,
            aux[1] as i8,
            (aux[1] >> 8) as i8,
            (aux[1] >> 16) as i8,
            (aux[1] >> 24) as i8,
            aux[2] as i8,
            (aux[2] >> 8) as i8,
            (aux[2] >> 16) as i8,
            (aux[2] >> 24) as i8,
            aux[3] as i8,
            (aux[3] >> 8) as i8,
            (aux[3] >> 16) as i8,
            (aux[3] >> 24) as i8,
        ];

        let mut is = 0;
        let mut q_offset = 0;
        let mut m: u8 = 1;
        for _n in 0..2 {
            let mut shift = 0;
            for _j in 0..4 {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = ((qs[q_offset + l] >> shift) & 3) as i8;
                    let hm = if hmask[q_offset + l] & m != 0 {
                        0i8
                    } else {
                        4i8
                    };
                    out.push(dl * (q - hm) as f32);
                }

                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = ((qs[q_offset + l + 16] >> shift) & 3) as i8;
                    let hm = if hmask[q_offset + l + 16] & m != 0 {
                        0i8
                    } else {
                        4i8
                    };
                    out.push(dl * (q - hm) as f32);
                }

                shift += 2;
                m <<= 1;
            }
            q_offset += 32;
        }
    }

    Ok(out)
}

/// Extract scale and min from K-quant scales array.
pub fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// `Q4_K`: 4-bit K-quant.
/// Block size: 256 elements.
/// Layout: [d: f16][dmin: f16][scales: 12 bytes][qs: 128 bytes] = 144 bytes
pub fn dequantize_q4_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 2 + 12 + 128;

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q4_K tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let dmin = half::f16::from_bits(u16_from_le(&block[2..4])).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut is = 0;
        for _j in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for &q in qs.iter().take(32) {
                out.push(d1 * (q & 0xF) as f32 - m1);
            }
            for &q in qs.iter().take(32) {
                out.push(d2 * (q >> 4) as f32 - m2);
            }
        }
    }

    Ok(out)
}

/// `Q5_K`: 5-bit K-quant.
/// Block size: 256 elements.
/// Layout: [d: f16][dmin: f16][scales: 12][qh: 32][qs: 128] = 176 bytes
pub fn dequantize_q5_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 2 + 12 + 32 + 128;

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q5_K tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let dmin = half::f16::from_bits(u16_from_le(&block[2..4])).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for l in 0..32 {
                let high = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d1 * ((qs[l] & 0xF) as f32 + high) - m1);
            }
            for l in 0..32 {
                let high = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out.push(d2 * ((qs[l] >> 4) as f32 + high) - m2);
            }

            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(out)
}

/// `Q6_K`: 6-bit K-quant.
/// Block size: 256 elements.
/// Layout: [ql: 128][qh: 64][scales: 16][d: f16] = 210 bytes
pub fn dequantize_q6_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 128 + 64 + 16 + 2;

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Q6_K tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales_raw = &block[192..208];
        let d = half::f16::from_bits(u16_from_le(&block[208..210])).to_f32();

        let mut ql_offset = 0;
        let mut qh_offset = 0;
        let mut sc_offset = 0;
        for _n in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_offset + l] & 0xF) | ((qh[qh_offset + l] & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[ql_offset + l + 32] & 0xF) | (((qh[qh_offset + l] >> 2) & 0x03) << 4))
                    as i8
                    - 32;
                let q3 = ((ql[ql_offset + l] >> 4) | (((qh[qh_offset + l] >> 4) & 0x03) << 4))
                    as i8
                    - 32;
                let q4 = ((ql[ql_offset + l + 32] >> 4) | (((qh[qh_offset + l] >> 6) & 0x03) << 4))
                    as i8
                    - 32;

                let s0 = scales_raw[sc_offset + is * 2] as i8;
                let s2 = scales_raw[sc_offset + is * 2 + 2] as i8;
                let s4 = scales_raw[sc_offset + is * 2 + 4] as i8;
                let s6 = scales_raw[sc_offset + is * 2 + 6] as i8;

                out.push(d * s0 as f32 * q1 as f32);
                out.push(d * s2 as f32 * q2 as f32);
                out.push(d * s4 as f32 * q3 as f32);
                out.push(d * s6 as f32 * q4 as f32);
            }
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }
    }

    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// IQ (Importance Matrix) Dequantization Functions
// Follows GGML block layouts from ggml-common.h (llama.cpp master).
// QK_K = 256 for all super-block types, except IQ4_NL (QK4_NL = 32).
// ──────────────────────────────────────────────────────────────────────────────

/// ``IQ1_S``: 1.5625 bpw, 256 elems/block, 50 bytes/block.
/// Layout: [d: f16][qs: u8×32][qh: u16×8]
/// qs: 8 1-bit values per byte. qh: sign bits, 1 per 32-element group.
pub fn dequantize_iq1_s(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 32 + 16; // d + qs(u8[32]) + qh(u16[8])

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ1_S tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[2..34];    // 32 bytes, 8 1-bit values per byte
        let qh = &block[34..50];   // 8 u16 values

        for i in 0..32 {
            let qh_word = u16_from_le(&qh[i * 2..i * 2 + 2]);
            // Each qh word has sign bits for 8 qs entries
            for j in 0..8 {
                let qs_byte = qs[i * 8 + j];
                // Each qs byte has 1-bit occupancy for 8 elements
                for k in 0..8 {
                    let occupied = (qs_byte >> k) & 1;
                    let sign = (qh_word >> (j * 8 + k)) & 1;
                    let val = if occupied == 0 {
                        0.0
                    } else if sign == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    out.push(d * val);
                }
            }
        }
    }

    Ok(out)
}

/// ``IQ1_M``: 1.75 bpw, 256 elems/block, 56 bytes/block.
/// Layout: [qs: u8×32][qh: u8×16][scales: u8×8]
/// NOTE: No ``d`` field in the struct. Scale is stored differently (global per tensor).
/// For dequantization we synthesize a d=1.0 since this is a reconstruction helper.
pub fn dequantize_iq1_m(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 32 + 16 + 8; // qs + qh + scales (NO d field)

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ1_M tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        // No d field — use default scale
        let d = 1.0;
        let qs = &block[0..32];     // 32 bytes, 8 1-bit values per byte
        let qh = &block[32..48];    // 16 bytes, sign and grid shift bits
        let scales = &block[48..56]; // 8 bytes, 2 × 4-bit scales per byte

        for i in 0..32 {
            let qs_byte = qs[i];
            // QH byte for this group
            let qh_byte = qh[i / 2];
            // Scale: per 32-element group, 4-bit stored in scales
            let sc_byte = scales[i / 4];
            let sc = if (i / 4) % 2 == 0 {
                (sc_byte & 0xF) as f32
            } else {
                (sc_byte >> 4) as f32
            };

            for k in 0..8 {
                let occupied = (qs_byte >> k) & 1;
                let sign = if k < 4 {
                    (qh_byte >> k) & 1
                } else {
                    (qh_byte >> (4 + k - 4)) & 1
                };
                let val = if occupied == 0 {
                    0.0
                } else if sign == 0 {
                    1.0
                } else {
                    -1.0
                };
                out.push(d * sc * val);
            }
        }
    }

    Ok(out)
}

/// ``IQ2_S``: 2.5625 bpw, 256 elems/block, 82 bytes/block.
/// Layout: [d: f16][qs: u8×64][qh: u8×8][scales: u8×8]
/// qs: 4 × 2-bit values per byte. qh: 1 extra bit per 32-element group.
/// scales: 2 × 4-bit per byte, giving 16 scale factors for 32-element groups.
pub fn dequantize_iq2_s(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 64 + 8 + 8; // d + qs + qh + scales

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ2_S tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[2..66];     // 64 bytes, 4 × 2-bit per byte
        let qh = &block[66..74];    // 8 bytes, 1 extra bit per 32 elements
        let scales = &block[74..82]; // 8 bytes, 2 × 4-bit per byte

        for i in 0..64 {
            let byte_val = qs[i];
            let v0 = ((byte_val >> 0) & 3) as f32;
            let v1 = ((byte_val >> 2) & 3) as f32;
            let v2 = ((byte_val >> 4) & 3) as f32;
            let v3 = ((byte_val >> 6) & 3) as f32;

            // Extra bit from qh for this value (each qh byte has 8 bits for 32 elements)
            // Index the qh bit: each value has position (i / 8) within group
            let qh_extra0 = ((qh[i / 8] >> (i % 8)) & 1) as f32;
            let _qh_extra1 = ((qh[i / 8] >> ((i % 8))) & 1) as f32; // same bit for all in group

            // Scale: each group of 32 elements has a 4-bit scale (2 per byte)
            let sc_idx = i / 16; // 4 scale groups
            let sc_byte = scales[sc_idx / 2];
            let sc_val = if sc_idx % 2 == 0 {
                (sc_byte & 0xF) as f32
            } else {
                (sc_byte >> 4) as f32
            };

            out.push(d * sc_val * (v0 + qh_extra0 - 1.5));
            out.push(d * sc_val * (v1 + qh_extra0 - 1.5));
            out.push(d * sc_val * (v2 + qh_extra0 - 1.5));
            out.push(d * sc_val * (v3 + qh_extra0 - 1.5));
        }
    }

    Ok(out)
}

/// ``IQ2_XXS``: 2.0625 bpw, 256 elems/block, 66 bytes/block.
/// Layout: [d: f16][qs: u16×32]
/// qs: each u16 holds 8 × 2-bit values.
pub fn dequantize_iq2_xxs(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 64; // d + qs(u16[32])

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ2_XXS tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs_bytes = &block[2..66]; // 32 u16 values

        for i in 0..32 {
            let qs_word = u16_from_le(&qs_bytes[i * 2..i * 2 + 2]);
            let vals = [
                ((qs_word >> 0) & 3) as f32,
                ((qs_word >> 2) & 3) as f32,
                ((qs_word >> 4) & 3) as f32,
                ((qs_word >> 6) & 3) as f32,
                ((qs_word >> 8) & 3) as f32,
                ((qs_word >> 10) & 3) as f32,
                ((qs_word >> 12) & 3) as f32,
                ((qs_word >> 14) & 3) as f32,
            ];

            for &v in &vals {
                out.push(d * (v - 1.5));
            }
        }
    }

    Ok(out)
}

/// ``IQ2_XS``: 2.3125 bpw, 256 elems/block, 74 bytes/block.
/// Layout: [d: f16][qs: u16×32][scales: u8×8]
/// qs: each u16 holds 8 × 2-bit values. scales: 2 × 4-bit per byte.
pub fn dequantize_iq2_xs(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 64 + 8; // d + qs(u16[32]) + scales

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ2_XS tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs_bytes = &block[2..66]; // 32 u16 values
        let scales = &block[66..74];  // 8 bytes, 2 × 4-bit per byte

        for i in 0..32 {
            let qs_word = u16_from_le(&qs_bytes[i * 2..i * 2 + 2]);
            let vals = [
                ((qs_word >> 0) & 3) as f32,
                ((qs_word >> 2) & 3) as f32,
                ((qs_word >> 4) & 3) as f32,
                ((qs_word >> 6) & 3) as f32,
                ((qs_word >> 8) & 3) as f32,
                ((qs_word >> 10) & 3) as f32,
                ((qs_word >> 12) & 3) as f32,
                ((qs_word >> 14) & 3) as f32,
            ];

            // Scale per 32 elements (4 qs words), 4-bit
            let sc_idx = i / 4;
            let sc_byte = scales[sc_idx / 2];
            let sc_val = if sc_idx % 2 == 0 {
                (sc_byte & 0xF) as f32
            } else {
                (sc_byte >> 4) as f32
            };

            for &v in &vals {
                out.push(d * sc_val * (v - 1.5));
            }
        }
    }

    Ok(out)
}

/// ``IQ3_S``: 3.4375 bpw, 256 elems/block, 110 bytes/block.
/// Layout: [d: f16][qs: u8×64][qh: u8×8][signs: u8×32][scales: u8×4]
/// qs: 4 × 2-bit low bits per byte. qh: 8 extra bits (1 per 32-element group).
/// signs: 8 sign bits per byte. scales: 4 × 6-bit values.
pub fn dequantize_iq3_s(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 64 + 8 + 32 + 4; // d + qs + qh + signs + scales

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ3_S tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[2..66];      // 64 bytes
        let qh = &block[66..74];     // 8 bytes
        let signs = &block[74..106]; // 32 bytes
        let scales_raw = &block[106..110]; // 4 bytes

        for i in 0..64 {
            let byte_val = qs[i];
            // 4 × 2-bit per byte
            let vals = [
                (byte_val & 0x3) as i32,
                ((byte_val >> 2) & 0x3) as i32,
                ((byte_val >> 4) & 0x3) as i32,
                ((byte_val >> 6) & 0x3) as i32,
            ];

            // Extra bit per 4-byte group from qh
            let qh_byte = qh[i / 8];
            let extra = ((qh_byte >> (i % 8)) & 1) as i32;
            let sign_byte = signs[i];
            let sc_val = scales_raw[i / 16] as f32;

            for k in 0..4 {
                let q = vals[k] | (extra << 2);
                let s = if (sign_byte >> k) & 1 != 0 { -1.0 } else { 1.0 };
                out.push(d * sc_val * s * q as f32);
            }
        }
    }

    Ok(out)
}

/// ``IQ3_XXS``: 3.0625 bpw, 256 elems/block, 98 bytes/block.
/// Layout: [d: f16][qs: u8×96]
/// qs: 256 × 3-bit values packed in 96 bytes (3 bytes per 8 values).
pub fn dequantize_iq3_xxs(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 96; // d + qs(u8[96])

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ3_XXS tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[2..98]; // 96 bytes

        // 3 bytes per 8 values: bytes [3i..3i+3] hold 8 × 3-bit values
        for i in 0..(QK_K / 8) {
            let b0 = qs[i * 3] as u32;
            let b1 = qs[i * 3 + 1] as u32;
            let b2 = qs[i * 3 + 2] as u32;
            let triple = b0 | (b1 << 8) | (b2 << 16); // 24 bits total

            // 8 × 3-bit values packed in 24 bits
            for j in 0..8 {
                let val = ((triple >> (j * 3)) & 7) as f32;
                out.push(d * (val - 3.0));
            }
        }
    }

    Ok(out)
}

/// ``IQ3_XS``: 3.0625 bpw, 256 elems/block, 98 bytes/block.
/// Shares the same struct layout as IQ3_XXS (block_iq3_xxs).
pub fn dequantize_iq3_xs(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    dequantize_iq3_xxs(raw)
}

/// ``IQ3_M``: 3-bit medium, 256 elems/block, 112 bytes/block.
/// Layout: [d: f16][dmin: f16][qs: u8×64][qh: u8×8][signs: u8×32][scales: u8×4]
pub fn dequantize_iq3_m(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 2 + 64 + 8 + 32 + 4; // d + dmin + qs + qh + signs + scales

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ3_M tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let dmin = half::f16::from_bits(u16_from_le(&block[2..4])).to_f32();
        let qs = &block[4..68];       // 64 bytes
        let qh = &block[68..76];      // 8 bytes
        let signs = &block[76..108];  // 32 bytes
        let scales_raw = &block[108..112]; // 4 bytes

        for i in 0..64 {
            let byte_val = qs[i];
            // 4 × 2-bit low bits
            let v0 = (byte_val & 0x3) as i32;
            let v1 = ((byte_val >> 2) & 0x3) as i32;
            let v2 = ((byte_val >> 4) & 0x3) as i32;
            let v3 = ((byte_val >> 6) & 0x3) as i32;

            // Extra from qh
            let qh_byte = qh[i / 8];
            let extra = ((qh_byte >> (i % 8)) & 1) as i32;

            // Sign byte
            let sign_byte = signs[i];
            let sc_val = scales_raw[i / 16] as f32;

            for k in 0..4 {
                let q = match k {
                    0 => v0,
                    1 => v1,
                    2 => v2,
                    _ => v3,
                };
                let s = if (sign_byte >> k) & 1 != 0 { -1.0 } else { 1.0 };
                out.push(d * sc_val * s * (q | extra) as f32 + dmin);
            }
        }
    }

    Ok(out)
}

/// ``IQ4_NL``: 4-bit non-linear, 32 elems/block, 18 bytes/block.
/// Layout: [d: f16][qs: u8×16] — each byte has two 4-bit values.
/// Already correct — no changes needed.
pub fn dequantize_iq4_nl(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK: usize = 32;
    const BLOCK_SIZE: usize = 2 + 16; // d + qs

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ4_NL tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let qs = &block[2..18];

        for &byte_val in qs {
            let v0 = (byte_val & 0xF) as f32;
            let v1 = (byte_val >> 4) as f32;
            out.push(d * (v0 - 8.0));
            out.push(d * (v1 - 8.0));
        }
    }

    Ok(out)
}

/// ``IQ4_XS``: 4-bit, extra small, 256 elems/block, 136 bytes/block.
/// Layout: [d: f16][scales_h: u16][scales_l: u8×4][qs: u8×128]
/// scales_h is a single uint16 (scale bits), scales_l[4] gives per 64-elem group scale lows.
/// qs: 2 × 4-bit values per byte.
pub fn dequantize_iq4_xs(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 2 + 4 + 128; // d + scales_h(u16) + scales_l(u8[4]) + qs

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "IQ4_XS tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16_from_le(&block[0..2])).to_f32();
        let scales_h = u16_from_le(&block[2..4]); // single u16
        let scales_l = &block[4..8];               // 4 bytes
        let qs = &block[8..136];                   // 128 bytes

        for i in 0..128 {
            let byte_val = qs[i];
            let v0 = (byte_val & 0xF) as f32;
            let v1 = (byte_val >> 4) as f32;

            // Scale group: 8 groups of 32 elements
            let sc_idx = i / 16; // scale group (0..7)
            // scales_h gives 1 bit per group (high), scales_l gives 4 bits per pair (low)
            let sc_h_bit = ((scales_h >> sc_idx) & 1) as f32;
            let sc_l_byte = scales_l[sc_idx / 2];
            let sc_l = if sc_idx % 2 == 0 {
                (sc_l_byte & 0xF) as f32
            } else {
                (sc_l_byte >> 4) as f32
            };
            let combined_scale = sc_l + sc_h_bit * 16.0 + 8.0;

            out.push(d * combined_scale * (v0 - 8.0));
            out.push(d * combined_scale * (v1 - 8.0));
        }
    }

    Ok(out)
}
