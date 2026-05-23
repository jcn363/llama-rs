//! Dequantization helpers — split into ``q`` (standard) and ``iq`` (imatrix) modules.

/// Infallibly read a `u16` from a two-byte slice (CPU‑local data, known size).
#[inline(always)]
pub(crate) fn u16_from_le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

mod iq;
mod q;

pub use iq::*;
pub use q::*;
