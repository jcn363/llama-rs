//! MXFP4 / NVFP4 micro-exponent floating-point dequantization.
//!
//! These types use a hybrid format: E2M1 values (2 exponent bits, 1 mantissa bit,
//! representing -12..12 via a lookup table) scaled by a per-sub-block exponent
//! stored as E8M0 (MXFP4) or UE4M3 (NVFP4).

use crate::GgufError;

/// MXFP4 k-values lookup table — E2M1 nibbles decoded to absolute float values.
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Convert an E8M0 byte to an f32 multiplier.
///
/// E8M0 is an 8-bit exponent-only IEEE-like format with bias = 127.
/// `0` is treated as zero; otherwise the value is `2^(e - 127)`.
#[inline(always)]
fn e8m0_to_f32(e: u8) -> f32 {
    if e == 0 {
        0.0
    } else {
        f32::from_bits((e as u32) << 23)
    }
}

/// Convert a UE4M3 byte to an f32 multiplier.
///
/// UE4M3 is an unsigned 7-bit OCP MX format:
/// - Bit [6:3]: exponent (4 bits, bias = 7)
/// - Bit [2:0]: mantissa (3 bits, implicit leading 1)
/// - High bit (bit 7) is padding/unused.
///
/// For exponent == 0 and mantissa != 0: subnormal `2^(1-7) * mantissa/8`.
/// For exponent == 0 and mantissa == 0: zero.
/// Otherwise: `2^(e - 7) * (1 + mantissa/8)`.
#[inline(always)]
fn ue4m3_to_f32(ue: u8) -> f32 {
    let exp = (ue >> 3) & 0x0F; // 4 exponent bits
    let man = (ue & 0x07) as u32; // 3 mantissa bits

    if exp == 0 {
        if man == 0 {
            return 0.0;
        }
        // Subnormal
        2.0f32.powi(1 - 7) * (man as f32 / 8.0)
    } else {
        // Normal: (-1)^0 * 2^(e-7) * (1 + m/8)
        f32::from_bits(((exp as u32 + 127 - 7) << 23) | (man << 20))
    }
}

/// Decode a single E2M1 nibble (4 bits) from an MXFP4/NVFP4 block to a float,
/// scaled by the per-sub-block multiplier `d`.
#[inline(always)]
fn decode_e2m1_value(nibble: u8, d: f32) -> f32 {
    KVALUES_MXFP4[(nibble & 0x0F) as usize] as f32 * d
}

/// ``Mxfp4``: MXFP4 micro-exponent floating-point (block size: 32).
/// Layout: [e: u8][qs: 16 bytes] = 17 bytes
///
/// `e` is an E8M0 (8-bit exponent-only) scale shared by all 32 E2M1 nibbles
/// in the block. Each byte in `qs` packs 2 E2M1 nibbles (low/high).
pub fn dequantize_mxfp4(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK: usize = 32;
    const BLOCK_SIZE: usize = 1 + 16; // e + qs

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Mxfp4 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let e = block[0];
        let qs = &block[1..17];
        let d = e8m0_to_f32(e);

        for &q in qs {
            out.push(decode_e2m1_value(q & 0x0F, d));
            out.push(decode_e2m1_value(q >> 4, d));
        }
    }

    Ok(out)
}

/// ``Nvfp4``: NVFP4 micro-exponent floating-point (block size: 64 elements).
/// Layout: [d: 4 × u8][qs: 32 bytes] = 36 bytes
///
/// 4 sub-blocks of 16 elements each. Each sub-block has a 7-bit UE4M3 scale
/// stored in `d[s]`. The 32 `qs` bytes pack 64 E2M1 nibbles (low nibble first).
pub fn dequantize_nvfp4(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK: usize = 64;
    const QK_SUB: usize = 16;
    const N_SUB: usize = 4;
    const BLOCK_SIZE: usize = N_SUB + 32; // d[4] + qs[32]

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Nvfp4 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let scales = &block[0..N_SUB];
        let qs = &block[N_SUB..];

        for (s_idx, &scale_byte) in scales.iter().enumerate().take(N_SUB) {
            let d = ue4m3_to_f32(scale_byte);
            let sub_start = s_idx * QK_SUB;

            for j in 0..(QK_SUB / 2) {
                let q_byte = qs[sub_start / 2 + j];
                out.push(decode_e2m1_value(q_byte & 0x0F, d));
                out.push(decode_e2m1_value(q_byte >> 4, d));
            }
        }
    }

    Ok(out)
}
