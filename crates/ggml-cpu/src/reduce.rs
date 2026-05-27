//! Reduction and normalization operations for the CPU backend.
//!
//! Provides common tensor reduction operations (`sum`, `mean`, `argmax`),
//! normalization techniques (`rms_norm`, `norm`, `group_norm`, `l2_norm`),
//! activation-related functions (`soft_max`, `soft_max_back`),
//! and loss functions (`cross_entropy_loss`, `cross_entropy_loss_back`).
//!
//! All functions operate on `f32` slices and return owned results.

// ─── Reductions ──────────────────────────────────────────────────────────────

/// Sum of all elements.
#[must_use]
pub fn sum(x: &[f32]) -> Vec<f32> {
    vec![x.iter().sum()]
}

/// Mean of all elements.
#[must_use]
pub fn mean(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![0.0];
    }
    vec![x.iter().sum::<f32>() / x.len() as f32]
}

/// Index of the maximum value.
///
/// # Panics
/// Panics if `x` is empty.
#[must_use]
pub fn argmax(x: &[f32]) -> usize {
    assert!(!x.is_empty(), "argmax requires non-empty input");
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("argmax failed on non-empty slice")
}

/// Count equal elements between two arrays.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
///
/// This intentionally uses exact float comparison. For approximate comparison,
/// use a tolerance-based check instead.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn count_equal(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "count_equal requires equal-length slices");
    a.iter().zip(b.iter()).filter(|(x, y)| x == y).count()
}

/// Cumulative sum along axis 0 (flattened).
#[must_use]
pub fn cumsum(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let mut running = 0.0f32;
    for &val in x {
        running += val;
        out.push(running);
    }
    out
}

/// Sum rows: for 2D input `[rows, cols]`, sum each row.
///
/// # Panics
/// Panics if `x.len()` is not divisible by `cols`.
#[must_use]
pub fn sum_rows(x: &[f32], cols: usize) -> Vec<f32> {
    assert_eq!(
        x.len() % cols,
        0,
        "sum_rows: x.len() must be divisible by cols"
    );
    let rows = x.len() / cols;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        out.push(x[start..start + cols].iter().sum());
    }
    out
}

// ─── Softmax & Loss ──────────────────────────────────────────────────────────

/// Softmax: `exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))`.
#[must_use]
pub fn soft_max(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    if sum_exp == 0.0 {
        return vec![0.0; x.len()];
    }
    exps.iter().map(|&e| e / sum_exp).collect()
}

/// Softmax backward: used for training.
///
/// `output` is the softmax output, `grad` is the upstream gradient.
///
/// # Panics
/// Panics if `output` and `grad` have different lengths.
#[must_use]
pub fn soft_max_back(output: &[f32], grad: &[f32]) -> Vec<f32> {
    assert_eq!(
        output.len(),
        grad.len(),
        "soft_max_back requires equal-length slices"
    );
    let n = output.len();
    let dot: f32 = output.iter().zip(grad.iter()).map(|(s, g)| s * g).sum();
    (0..n).map(|i| output[i] * (grad[i] - dot)).collect()
}

/// Cross entropy loss: `-sum(target * log(prediction))`.
///
/// # Panics
/// Panics if `prediction` and `target` have different lengths.
#[must_use]
pub fn cross_entropy_loss(prediction: &[f32], target: &[f32]) -> Vec<f32> {
    assert_eq!(
        prediction.len(),
        target.len(),
        "cross_entropy_loss requires equal-length slices"
    );
    let loss: f32 = prediction
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            if p <= 0.0 {
                -t * f32::EPSILON.ln()
            } else {
                -t * p.ln()
            }
        })
        .sum();
    vec![loss]
}

/// Cross entropy loss backward.
///
/// # Panics
/// Panics if `prediction` and `target` have different lengths.
#[must_use]
pub fn cross_entropy_loss_back(prediction: &[f32], target: &[f32]) -> Vec<f32> {
    assert_eq!(
        prediction.len(),
        target.len(),
        "cross_entropy_loss_back requires equal-length slices"
    );
    prediction
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| if p <= 0.0 { -t / f32::EPSILON } else { -t / p })
        .collect()
}

// ─── Normalization ───────────────────────────────────────────────────────────

/// Root Mean Square normalization: `y = (x / RMS(x)) * weight`
/// where `RMS(x) = sqrt(mean(x^2) + eps)`.
///
/// # Panics
/// Panics if `x` and `weight` have different lengths.
#[must_use]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "rms_norm requires equal-length x and weight"
    );
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v / rms * w)
        .collect()
}

/// RMS Norm backward.
///
/// # Panics
/// Panics if input slices have mismatched lengths.
#[must_use]
pub fn rms_norm_back(x: &[f32], weight: &[f32], grad: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(
        n,
        weight.len(),
        "rms_norm_back: x and weight must have same length"
    );
    assert_eq!(
        n,
        grad.len(),
        "rms_norm_back: x and grad must have same length"
    );
    if n == 0 {
        return Vec::new();
    }
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_n = 1.0 / n as f32;
    let sum_sq_grad: f32 = x
        .iter()
        .zip(grad.iter())
        .zip(weight.iter())
        .map(|((&v, &g), &w)| v * g * w)
        .sum();
    let factor = -inv_rms.powi(3) * inv_n * sum_sq_grad;
    x.iter()
        .zip(weight.iter())
        .zip(grad.iter())
        .map(|((&v, &w), &g)| g * w * inv_rms + v * factor * w)
        .collect()
}

/// L2 normalization: `y[i] = x[i] / sqrt(sum(x[j]^2) + eps)`.
#[must_use]
pub fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt() + eps;
    x.iter().map(|&v| v / norm).collect()
}

/// Group normalization.
///
/// Normalizes each group independently:
/// `y = (x - mean(group)) / sqrt(var(group) + eps) * weight + bias`
///
/// # Panics
/// Panics if `x.len()` is not divisible by `n_groups`, or if weight/bias
/// lengths don't match the group size.
#[must_use]
pub fn group_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, n_groups: usize) -> Vec<f32> {
    assert!(
        x.len() % n_groups == 0,
        "group_norm: x.len() must be divisible by n_groups"
    );
    let n = x.len();
    let group_size = n / n_groups;
    assert_eq!(
        weight.len(),
        group_size,
        "group_norm: weight.len() must equal group_size"
    );
    assert_eq!(
        bias.len(),
        group_size,
        "group_norm: bias.len() must equal group_size"
    );
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for g in 0..n_groups {
        let start = g * group_size;
        let group = &x[start..start + group_size];
        let mean: f32 = group.iter().sum::<f32>() / group_size as f32;
        let var: f32 = group.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / group_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, &v) in group.iter().enumerate() {
            out.push((v - mean) * inv_std * weight[i] + bias[i]);
        }
    }
    out
}

/// Layer normalization (standard):
/// `y = (x - mean) / sqrt(var + eps) * weight + bias`.
///
/// If `bias` is `None`, only the weight scaling is applied.
///
/// # Panics
/// Panics if `x` and `weight` have different lengths, or if `bias` is
/// `Some` with a length different from `x`.
#[must_use]
pub fn norm(x: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "norm: x and weight must have same length"
    );
    if let Some(b) = bias {
        assert_eq!(x.len(), b.len(), "norm: x and bias must have same length");
    }
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    match bias {
        Some(b) => x
            .iter()
            .zip(weight.iter())
            .zip(b.iter())
            .map(|((&v, &w), &b)| (v - mean) * inv_std * w + b)
            .collect(),
        None => x
            .iter()
            .zip(weight.iter())
            .map(|(&v, &w)| (v - mean) * inv_std * w)
            .collect(),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sum ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sum() {
        let result = sum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((result[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_empty() {
        let result = sum(&[]);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    // ── mean ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mean() {
        let result = mean(&[2.0, 4.0, 6.0]);
        assert!((result[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_empty() {
        let result = mean(&[]);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    // ── argmax ───────────────────────────────────────────────────────────

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn test_argmax_first() {
        assert_eq!(argmax(&[9.0, 3.0, 1.0]), 0);
    }

    #[test]
    #[should_panic(expected = "argmax requires non-empty input")]
    fn test_argmax_empty() {
        argmax(&[]);
    }

    // ── count_equal ──────────────────────────────────────────────────────

    #[test]
    fn test_count_equal() {
        assert_eq!(count_equal(&[1.0, 2.0, 3.0], &[1.0, 0.0, 3.0]), 2);
    }

    #[test]
    #[should_panic]
    fn test_count_equal_mismatched() {
        count_equal(&[1.0], &[1.0, 2.0]);
    }

    // ── cumsum ───────────────────────────────────────────────────────────

    #[test]
    fn test_cumsum() {
        let result = cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
        assert!((result[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_cumsum_empty() {
        let result = cumsum(&[]);
        assert!(result.is_empty());
    }

    // ── sum_rows ─────────────────────────────────────────────────────────

    #[test]
    fn test_sum_rows() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = sum_rows(&x, 2);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 7.0).abs() < 1e-6);
        assert!((result[2] - 11.0).abs() < 1e-6);
    }

    // ── soft_max ─────────────────────────────────────────────────────────

    #[test]
    fn test_soft_max() {
        let x = vec![1.0, 2.0, 3.0];
        let result = soft_max(&x);
        assert_eq!(result.len(), 3);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_soft_max_empty() {
        let result = soft_max(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_soft_max_uniform() {
        let x = vec![0.0; 5];
        let result = soft_max(&x);
        for &v in &result {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }

    // ── soft_max_back ────────────────────────────────────────────────────

    #[test]
    fn test_soft_max_back() {
        let output = vec![0.2, 0.5, 0.3];
        let grad = vec![1.0, 0.0, 0.0];
        let result = soft_max_back(&output, &grad);
        assert_eq!(result.len(), 3);
    }

    // ── cross_entropy_loss ───────────────────────────────────────────────

    #[test]
    fn test_cross_entropy_loss() {
        let pred = vec![0.7, 0.2, 0.1];
        let target = vec![1.0, 0.0, 0.0];
        let result = cross_entropy_loss(&pred, &target);
        assert!(result[0] > 0.0);
    }

    // ── cross_entropy_loss_back ──────────────────────────────────────────

    #[test]
    fn test_cross_entropy_loss_back() {
        let pred = vec![0.7, 0.2, 0.1];
        let target = vec![1.0, 0.0, 0.0];
        let result = cross_entropy_loss_back(&pred, &target);
        assert_eq!(result.len(), 3);
    }

    // ── rms_norm ─────────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let eps = 1e-6;
        let result = rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        let rms = (30.0_f32 / 4.0 + eps).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 1e-5);
        assert!((result[3] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_weighted() {
        let x = vec![1.0, 2.0];
        let weight = vec![2.0, 0.5];
        let eps = 1e-6;
        let result = rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 2);
        let rms = ((1.0 + 4.0) / 2.0 + eps).sqrt();
        assert!((result[0] - 1.0 / rms * 2.0).abs() < 1e-5);
        assert!((result[1] - 2.0 / rms * 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_empty() {
        let result = rms_norm(&[], &[], 1e-6);
        assert!(result.is_empty());
    }

    // ── rms_norm_back ────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_back() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0; 3];
        let g = vec![1.0; 3];
        let result = rms_norm_back(&x, &w, &g, 1e-6);
        assert_eq!(result.len(), 3);
    }

    // ── l2_norm ──────────────────────────────────────────────────────────

    #[test]
    fn test_l2_norm() {
        let x = vec![3.0, 4.0];
        let result = l2_norm(&x, 1e-6);
        let norm = (9.0 + 16.0_f32).sqrt() + 1e-6;
        assert!((result[0] - 3.0 / norm).abs() < 1e-5);
        assert!((result[1] - 4.0 / norm).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm_empty() {
        let result = l2_norm(&[], 1e-6);
        assert!(result.is_empty());
    }

    // ── group_norm ───────────────────────────────────────────────────────

    #[test]
    fn test_group_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0; 3];
        let bias = vec![0.0; 3];
        let eps = 1e-6;
        let result = group_norm(&x, &weight, &bias, eps, 2);
        assert_eq!(result.len(), 6);
        let inv_std = 1.0 / (2.0_f32 / 3.0 + eps).sqrt();
        assert!((result[0] - (1.0 - 2.0) * inv_std).abs() < 1e-5);
        assert!((result[3] - (4.0 - 5.0) * inv_std).abs() < 1e-5);
    }

    #[test]
    fn test_group_norm_single_group() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let eps = 1e-6;
        let result = group_norm(&x, &weight, &bias, eps, 1);
        assert_eq!(result.len(), 4);
        let mean = 2.5;
        let var = ((1.0f32 - mean).powi(2)
            + (2.0f32 - mean).powi(2)
            + (3.0f32 - mean).powi(2)
            + (4.0f32 - mean).powi(2))
            / 4.0f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        assert!((result[0] - (1.0 - mean) * inv_std).abs() < 1e-5);
    }

    // ── norm (layer norm) ────────────────────────────────────────────────

    #[test]
    fn test_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let eps = 1e-6;
        let result = norm(&x, &weight, None, eps);
        assert_eq!(result.len(), 4);
        let mean = 2.5;
        let var = ((1.0f32 - mean).powi(2)
            + (2.0f32 - mean).powi(2)
            + (3.0f32 - mean).powi(2)
            + (4.0f32 - mean).powi(2))
            / 4.0f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        assert!((result[0] - (1.0 - mean) * inv_std).abs() < 1e-5);
    }

    #[test]
    fn test_norm_with_bias() {
        let x = vec![1.0, 2.0];
        let weight = vec![1.0; 2];
        let bias = vec![0.5, -0.5];
        let eps = 1e-6;
        let result = norm(&x, &weight, Some(&bias), eps);
        assert_eq!(result.len(), 2);
        let mean = 1.5;
        let var = ((1.0f32 - mean).powi(2) + (2.0f32 - mean).powi(2)) / 2.0f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        assert!((result[0] - ((1.0 - mean) * inv_std + 0.5)).abs() < 1e-5);
        assert!((result[1] - ((2.0 - mean) * inv_std - 0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_norm_empty() {
        let result = norm(&[], &[], None, 1e-6);
        assert!(result.is_empty());
    }
}
