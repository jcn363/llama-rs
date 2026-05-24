//! Dequantization helpers.
//!
//! Split into per-family modules:
//! - [`q`] — standard quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
//! - [`k_quant`] — K-quants (Q2_K–Q6_K, Q8_K, Q1_0)
//! - [`iq`] — importance-matrix quants (Iq1S–Iq4Xs)
//! - [`ternary`] — ternary quants (Tq1_0, Tq2_0)
//! - [`mxfp`] — MXFP4/NVFP4 micro-exponent quants

/// Infallibly read a `u16` from a two-byte slice (CPU‑local data, known size).
#[inline(always)]
pub(crate) fn u16_from_le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

mod iq;
mod k_quant;
mod mxfp;
mod q;
mod ternary;

pub use iq::*;
pub use k_quant::*;
pub use mxfp::*;
pub use q::*;
pub use ternary::*;
