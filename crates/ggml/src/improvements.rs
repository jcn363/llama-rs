// Stub implementations for improvements module functions used in defaults.

/// Simple RMS norm implementation without SIMD.
pub fn rms_norm_simd(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ssq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ssq / n as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter().cycle())
        .map(|(&xi, &wi)| wi * (xi / rms))
        .collect()
}

// Add other stub functions as needed.
