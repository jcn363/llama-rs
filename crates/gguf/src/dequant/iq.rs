//! IQ (Importance Matrix) quantized type dequantization.
//! Block layouts mirrored from `ggml-common.h` (llama.cpp master).
//! `QK_K = 256` for all super-block types, except IQ4_NL (`QK4_NL = 32`).

use crate::GgufError;

use super::u16_from_le;

// ──────────────────────────────────────────────────────────────────────────────
// IQ (Importance Matrix) Dequantization Functions
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
        let qs = &block[2..34]; // 32 bytes, 8 1-bit values per byte
        let qh = &block[34..50]; // 8 u16 values

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
        let qs = &block[0..32]; // 32 bytes, 8 1-bit values per byte
        let qh = &block[32..48]; // 16 bytes, sign and grid shift bits
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
        let qs = &block[2..66]; // 64 bytes, 4 × 2-bit per byte
        let qh = &block[66..74]; // 8 bytes, 1 extra bit per 32 elements
        let scales = &block[74..82]; // 8 bytes, 2 × 4-bit per byte

        for i in 0..64 {
            let byte_val = qs[i];
            let v0 = (byte_val & 3) as f32;
            let v1 = ((byte_val >> 2) & 3) as f32;
            let v2 = ((byte_val >> 4) & 3) as f32;
            let v3 = ((byte_val >> 6) & 3) as f32;

            // Extra bit from qh for this value (each qh byte has 8 bits for 32 elements)
            let qh_extra0 = ((qh[i / 8] >> (i % 8)) & 1) as f32;

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
                (qs_word & 3) as f32,
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
        let scales = &block[66..74]; // 8 bytes, 2 × 4-bit per byte

        for i in 0..32 {
            let qs_word = u16_from_le(&qs_bytes[i * 2..i * 2 + 2]);
            let vals = [
                (qs_word & 3) as f32,
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
        let qs = &block[2..66]; // 64 bytes
        let qh = &block[66..74]; // 8 bytes
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

            for (k, &v) in vals.iter().enumerate() {
                let q = v | (extra << 2);
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
        let qs = &block[4..68]; // 64 bytes
        let qh = &block[68..76]; // 8 bytes
        let signs = &block[76..108]; // 32 bytes
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

            for k in 0..4 {
                let q = match k {
                    0 => v0,
                    1 => v1,
                    2 => v2,
                    _ => v3,
                };
                let sc_val = scales_raw[i / 16] as f32;
                let s = if (sign_byte >> k) & 1 != 0 { -1.0 } else { 1.0 };
                out.push(d * sc_val * s * (q | extra) as f32 + dmin);
            }
        }
    }

    Ok(out)
}

/// ``IQ4_NL``: 4-bit non-linear, 32 elems/block, 18 bytes/block.
/// Layout: [d: f16][qs: u8×16] — each byte has two 4-bit values.
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
        let scales_l = &block[4..8]; // 4 bytes
        let qs = &block[8..136]; // 128 bytes

        for (i, &byte_val) in qs.iter().enumerate() {
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
