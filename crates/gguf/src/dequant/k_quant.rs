//! K-quantized type dequantization (Q2_K through Q6_K, Q8_K, Q1_0).

use crate::GgufError;

use super::u16_from_le;

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

/// ``Q2_K``: 2-bit K-quant.
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

/// ``Q3_K``: 3-bit K-quant.
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

/// ``Q4_K``: 4-bit K-quant.
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

/// ``Q5_K``: 5-bit K-quant.
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

/// ``Q6_K``: 6-bit K-quant.
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
        for (i, &q_byte) in qs.iter().enumerate().take(QK_K) {
            let q = q_byte as i8 as f32;
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
