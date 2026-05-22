//! Dequantization helper functions for GGUF tensors.
//!
//! This module contains the implementations of the various quantized
//! tensor dequantization algorithms that were previously embedded in
//! `lib.rs`.  Keeping them in a dedicated file improves readability
//! and reduces the size of the main library file.

use super::GgufError;
use rayon::prelude::*;
use std::convert::TryInto;

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
                    let d =
                        half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                            .to_f32();
                    let qs = &block[2..18];

                    let out_start = block_idx * QK4_0 - start_block * QK4_0;
                    for i in 0..16 {
                        let v0 = (qs[i] & 0x0F) as i8 - 8;
                        let v1 = (qs[i] >> 4) as i8 - 8;
                        if out_start + i * 2 < chunk.len() {
                            chunk[out_start + i * 2] = v0 as f32 * d;
                            if out_start + i * 2 + 1 < chunk.len() {
                                chunk[out_start + i * 2 + 1] = v1 as f32 * d;
                            }
                        }
                    }
                }
            });

        Ok(out)
    } else {
        let mut out = Vec::with_capacity(num_blocks * QK4_0);

        for block in raw.chunks_exact(BLOCK_SIZE) {
            let d =
                half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
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
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
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
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
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
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
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
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let qs = &block[2..34];

        for &q in qs {
            out.push(q as i8 as f32 * d);
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
        let d =
            half::f16::from_bits(u16::from_le_bytes(block[80..82].try_into().unwrap())).to_f32();
        let min =
            half::f16::from_bits(u16::from_le_bytes(block[82..84].try_into().unwrap())).to_f32();

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
        let d_all =
            half::f16::from_bits(u16::from_le_bytes(block[108..110].try_into().unwrap())).to_f32();

        // Unpack scales from 12 bytes into 16 signed scale values
        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;

        let mut aux: [u32; 4] = [
            u32::from_le_bytes(scales_raw[0..4].try_into().unwrap()),
            u32::from_le_bytes(scales_raw[4..8].try_into().unwrap()),
            u32::from_le_bytes(scales_raw[8..12].try_into().unwrap()),
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
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
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
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
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
        let d =
            half::f16::from_bits(u16::from_le_bytes(block[208..210].try_into().unwrap())).to_f32();

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
