//! Ternary quantized type dequantization (Tq1_0, Tq2_0).

use crate::GgufError;

use super::u16_from_le;

/// Lookup table for base-5/3 digit extraction.
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

/// ``Tq1_0``: Ternary quantized 1.0 (∼1.69 bpw).
/// Block size: 256 elements.
/// Layout: [qs: 48][qh: 4][d: f16] = 54 bytes
///
/// Each of the 48 `qs` bytes encodes 5 ternary values {-1, 0, +1} via
/// base-3 packing. The 4 `qh` bytes each encode 4 extra ternary values.
/// Values are decoded using a fixed-point extraction: `((qs * pow3[n] * 3) >> 8) - 1`.
pub fn dequantize_tq1_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const QS_BYTES: usize = 48;
    const QH_BYTES: usize = 4;
    const BLOCK_SIZE: usize = QS_BYTES + QH_BYTES + 2; // qs + qh + d

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Tq1_0 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let qs = &block[0..QS_BYTES];
        let qh = &block[QS_BYTES..QS_BYTES + QH_BYTES];
        let d = half::f16::from_bits(u16_from_le(
            &block[QS_BYTES + QH_BYTES..QS_BYTES + QH_BYTES + 2],
        ))
        .to_f32();

        // Process qs in 32-byte chunks (first chunk)
        let mut j = 0;
        while j + 32 <= QS_BYTES {
            for &p in POW3.iter().take(5) {
                for m in 0..32 {
                    let q = (qs[j + m] as u16).wrapping_mul(p as u16);
                    let xi = ((q * 3) >> 8) as i16;
                    out.push((xi - 1) as f32 * d);
                }
            }
            j += 32;
        }

        // Process remaining qs bytes (< 32)
        while j + 16 <= QS_BYTES {
            for &p in POW3.iter().take(5) {
                for m in 0..16 {
                    let q = (qs[j + m] as u16).wrapping_mul(p as u16);
                    let xi = ((q * 3) >> 8) as i16;
                    out.push((xi - 1) as f32 * d);
                }
            }
            j += 16;
        }

        // Process qh bytes — each encodes 4 ternary values using pow3[0..4]
        for &p in POW3.iter().take(4) {
            for &qh_byte in qh {
                let q = (qh_byte as u16).wrapping_mul(p as u16);
                let xi = ((q * 3) >> 8) as i16;
                out.push((xi - 1) as f32 * d);
            }
        }
    }

    Ok(out)
}

/// ``Tq2_0``: Ternary quantized 2.0 (∼2.06 bpw).
/// Block size: 256 elements.
/// Layout: [qs: 64][d: f16] = 66 bytes
///
/// Each byte in `qs` encodes 4 × 2-bit values decoded as `(value - 1) * d`,
/// producing values {-d, 0, d, 2d}.
pub fn dequantize_tq2_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const QS_BYTES: usize = 64;
    const BLOCK_SIZE: usize = QS_BYTES + 2; // qs + d

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError(
            "Tq2_0 tensor size not multiple of block size".into(),
        ));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let qs = &block[0..QS_BYTES];
        let d = half::f16::from_bits(u16_from_le(&block[QS_BYTES..QS_BYTES + 2])).to_f32();

        // Process in 32-byte chunks: each byte encodes 4 × 2-bit values
        let mut j = 0;
        while j + 32 <= QS_BYTES {
            for l in 0..4 {
                for m in 0..32 {
                    let q = (qs[j + m] >> (l * 2)) & 3;
                    out.push((q as i8 - 1) as f32 * d);
                }
            }
            j += 32;
        }
    }

    Ok(out)
}
