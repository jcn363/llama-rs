//! SIMD-optimized dot product for f32 vectors.
//!
//! Uses AVX (8-wide) → SSE4.2 (4-wide) → scalar fallback.
//! No FMA instructions (bdver1 doesn't support them) — uses mul + add.

/// Number of floats per AVX register (256 bits / 32 bits).
#[cfg(target_arch = "x86_64")]
const AVX_F32_EPR: usize = 8;

/// Number of floats per SSE register (128 bits / 32 bits).
#[cfg(target_arch = "x86_64")]
const SSE_F32_EPR: usize = 4;

/// Number of accumulators for SIMD dot product (unroll factor).
const DOT_ARR: usize = 4;

/// AVX step size: 8 floats × 4 accumulators = 32 floats per iteration.
#[cfg(target_arch = "x86_64")]
const AVX_F32_STEP: usize = AVX_F32_EPR * DOT_ARR;

/// SSE step size: 4 floats × 4 accumulators = 16 floats per iteration.
#[cfg(target_arch = "x86_64")]
const SSE_F32_STEP: usize = SSE_F32_EPR * DOT_ARR;

/// Compute dot product of two f32 vectors using SIMD when available.
///
/// Uses AVX (8-wide) → SSE4.2 (4-wide) → scalar fallback.
/// No FMA instructions (bdver1 doesn't support them) — uses mul + add.
#[must_use]
#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Try AVX first
        if crate::cpu_features::has_avx() {
            return dot_f32_avx(&x[..n], &y[..n]);
        }
        // Fallback to SSE4.2
        if crate::cpu_features::has_sse4_2() {
            return dot_f32_sse(&x[..n], &y[..n]);
        }
    }

    // Scalar fallback
    dot_f32_scalar(&x[..n], &y[..n])
}

/// Scalar dot product fallback.
#[inline]
pub(super) fn dot_f32_scalar(x: &[f32], y: &[f32]) -> f32 {
    let mut sum: f64 = 0.0;
    for i in 0..x.len() {
        sum += f64::from(x[i]) * f64::from(y[i]);
    }
    sum as f32
}

/// AVX-optimized dot product (8-wide, 4 accumulators = 32 floats/iteration).
/// Uses mul + add (no FMA) since bdver1 doesn't support FMA.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_f32_avx(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let np = n & !(AVX_F32_STEP - 1);

    unsafe {
        let mut sum: [__m256; DOT_ARR] = [_mm256_setzero_ps(); DOT_ARR];
        let mut ax: [__m256; DOT_ARR] = [_mm256_setzero_ps(); DOT_ARR];
        let mut ay: [__m256; DOT_ARR] = [_mm256_setzero_ps(); DOT_ARR];

        // Main loop: process AVX_F32_STEP (32) floats per iteration
        for i in (0..np).step_by(AVX_F32_STEP) {
            for j in 0..DOT_ARR {
                let idx = i + j * AVX_F32_EPR;
                ax[j] = _mm256_loadu_ps(x.as_ptr().add(idx));
                ay[j] = _mm256_loadu_ps(y.as_ptr().add(idx));
                // mul + add (no FMA on bdver1)
                sum[j] = _mm256_add_ps(_mm256_mul_ps(ax[j], ay[j]), sum[j]);
            }
        }

        // Horizontal reduction: sum[0..3] → sum[0]
        for j in 1..DOT_ARR {
            sum[0] = _mm256_add_ps(sum[0], sum[j]);
        }

        // Extract 256-bit sum to scalar
        let sum128 = _mm_add_ps(
            _mm256_extractf128_ps(sum[0], 1),
            _mm256_castps256_ps128(sum[0]),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
        let mut result = _mm_cvtss_f32(sum32);

        // Leftover elements (scalar)
        for i in np..n {
            result += x[i] * y[i];
        }

        result
    }
}

/// SSE4.2-optimized dot product (4-wide, 4 accumulators = 16 floats/iteration).
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_f32_sse(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let np = n & !(SSE_F32_STEP - 1);

    unsafe {
        let mut sum: [__m128; DOT_ARR] = [_mm_setzero_ps(); DOT_ARR];
        let mut ax: [__m128; DOT_ARR] = [_mm_setzero_ps(); DOT_ARR];
        let mut ay: [__m128; DOT_ARR] = [_mm_setzero_ps(); DOT_ARR];

        // Main loop: process SSE_F32_STEP (16) floats per iteration
        for i in (0..np).step_by(SSE_F32_STEP) {
            for j in 0..DOT_ARR {
                let idx = i + j * SSE_F32_EPR;
                ax[j] = _mm_loadu_ps(x.as_ptr().add(idx));
                ay[j] = _mm_loadu_ps(y.as_ptr().add(idx));
                sum[j] = _mm_add_ps(_mm_mul_ps(ax[j], ay[j]), sum[j]);
            }
        }

        // Horizontal reduction
        for j in 1..DOT_ARR {
            sum[0] = _mm_add_ps(sum[0], sum[j]);
        }

        let sum64 = _mm_add_ps(sum[0], _mm_movehl_ps(sum[0], sum[0]));
        let sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
        let mut result = _mm_cvtss_f32(sum32);

        // Leftover
        for i in np..n {
            result += x[i] * y[i];
        }

        result
    }
}
