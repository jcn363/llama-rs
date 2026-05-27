//! Element-wise arithmetic operations for f32 vectors.
//!
//! Provides scalar (non-SIMD) implementations of common element-wise
//! operations: `add`, `sub`, `mul`, `div`, `add1`, `clamp`, `scale`,
//! `fill`, and `arange`.
//!
//! All public functions are `#[must_use]` — ignoring the result is
//! almost certainly a bug.

/// Element-wise addition: `c[i] = a[i] + b[i]`.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "add: inputs must have the same length ({} != {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise subtraction: `c[i] = a[i] - b[i]`.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "sub: inputs must have the same length ({} != {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Element-wise multiplication: `c[i] = a[i] * b[i]`.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "mul: inputs must have the same length ({} != {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Element-wise division: `c[i] = a[i] / b[i]`.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn div(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "div: inputs must have the same length ({} != {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

/// Element-wise addition with scalar broadcast: `c[i] = a[i] + b`.
#[must_use]
pub fn add1(a: &[f32], b: f32) -> Vec<f32> {
    a.iter().map(|x| x + b).collect()
}

/// Element-wise clamp: `c[i] = clamp(a[i], min, max)`.
///
/// Each element is bounded to `[min, max]`.  `min` must be ≤ `max`.
///
/// # Panics
///
/// Panics if `min > max`.
#[must_use]
pub fn clamp(a: &[f32], min: f32, max: f32) -> Vec<f32> {
    assert!(min <= max, "clamp: min ({min}) must be <= max ({max})");
    a.iter()
        .map(|x| {
            if *x < min {
                min
            } else if *x > max {
                max
            } else {
                *x
            }
        })
        .collect()
}

/// Scale: `c[i] = a[i] * scale`.
#[must_use]
pub fn scale(a: &[f32], scale: f32) -> Vec<f32> {
    a.iter().map(|x| x * scale).collect()
}

/// Fill: returns a vector of length `n` filled with value `v`.
#[must_use]
pub fn fill(n: usize, v: f32) -> Vec<f32> {
    vec![v; n]
}

/// Arange: returns `[start, start + step, start + 2 * step, ...]`.
///
/// Produces exactly `n` elements.  When `step` is negative successive
/// values decrease; no check is performed for overflow.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn arange(n: usize, start: f32, step: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(start + (i as f32) * step);
    }
    out
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- add ----------------------------------------------------------------

    #[test]
    fn add_should_compute_correct_result() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let c = add(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn add_should_handle_empty() {
        assert!(add(&[], &[]).is_empty());
    }

    #[test]
    #[should_panic(expected = "must have the same length")]
    fn add_should_panic_on_length_mismatch() {
        add(&[1.0], &[2.0, 3.0]);
    }

    // -- sub ----------------------------------------------------------------

    #[test]
    fn sub_should_compute_correct_result() {
        let a = [5.0, 7.0, 9.0];
        let b = [1.0, 2.0, 3.0];
        let c = sub(&a, &b);
        assert_eq!(c, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn sub_should_handle_empty() {
        assert!(sub(&[], &[]).is_empty());
    }

    #[test]
    #[should_panic(expected = "must have the same length")]
    fn sub_should_panic_on_length_mismatch() {
        sub(&[1.0], &[2.0, 3.0]);
    }

    // -- mul ----------------------------------------------------------------

    #[test]
    fn mul_should_compute_correct_result() {
        let a = [2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0];
        let c = mul(&a, &b);
        assert_eq!(c, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn mul_should_handle_empty() {
        assert!(mul(&[], &[]).is_empty());
    }

    #[test]
    #[should_panic(expected = "must have the same length")]
    fn mul_should_panic_on_length_mismatch() {
        mul(&[1.0], &[2.0, 3.0]);
    }

    // -- div ----------------------------------------------------------------

    #[test]
    fn div_should_compute_correct_result() {
        let a = [10.0, 20.0, 30.0];
        let b = [2.0, 4.0, 5.0];
        let c = div(&a, &b);
        assert_eq!(c, vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn div_should_handle_empty() {
        assert!(div(&[], &[]).is_empty());
    }

    #[test]
    #[should_panic(expected = "must have the same length")]
    fn div_should_panic_on_length_mismatch() {
        div(&[1.0], &[2.0, 3.0]);
    }

    // -- add1 ---------------------------------------------------------------

    #[test]
    fn add1_should_broadcast_scalar() {
        let a = [1.0, 2.0, 3.0];
        let c = add1(&a, 10.0);
        assert_eq!(c, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn add1_should_handle_empty() {
        assert!(add1(&[], 5.0).is_empty());
    }

    // -- clamp --------------------------------------------------------------

    #[test]
    fn clamp_should_bound_values() {
        let a = [-5.0, 0.5, 2.0, 8.0, 15.0];
        let c = clamp(&a, 0.0, 10.0);
        assert_eq!(c, vec![0.0, 0.5, 2.0, 8.0, 10.0]);
    }

    #[test]
    fn clamp_should_handle_empty() {
        assert!(clamp(&[], 0.0, 1.0).is_empty());
    }

    #[test]
    fn clamp_should_allow_equal_bounds() {
        let a = [1.0, 2.0, 3.0];
        let c = clamp(&a, 2.0, 2.0);
        assert_eq!(c, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "must be <= max")]
    fn clamp_should_panic_on_reversed_bounds() {
        clamp(&[1.0], 5.0, 0.0);
    }

    // -- scale --------------------------------------------------------------

    #[test]
    fn scale_should_multiply_each_element() {
        let a = [1.0, 2.0, 3.0];
        let c = scale(&a, 2.5);
        assert_eq!(c, vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn scale_should_handle_zero() {
        let a = [1.0, 2.0, 3.0];
        let c = scale(&a, 0.0);
        assert_eq!(c, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn scale_should_handle_empty() {
        assert!(scale(&[], 3.0).is_empty());
    }

    // -- fill ---------------------------------------------------------------

    #[test]
    fn fill_should_create_correct_length() {
        let c = fill(5, 3.14);
        assert_eq!(c.len(), 5);
        assert!(c.iter().all(|x| (*x - 3.14).abs() < f32::EPSILON));
    }

    #[test]
    fn fill_should_handle_zero() {
        assert!(fill(0, 1.0).is_empty());
    }

    // -- arange -------------------------------------------------------------

    #[test]
    fn arange_should_produce_ascending_sequence() {
        let c = arange(5, 0.0, 1.0);
        assert_eq!(c, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn arange_should_handle_negative_step() {
        let c = arange(4, 3.0, -1.0);
        assert_eq!(c, vec![3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn arange_should_handle_fractional_steps() {
        let c = arange(4, 1.0, 0.5);
        assert_eq!(c, vec![1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn arange_should_handle_zero() {
        assert!(arange(0, 0.0, 1.0).is_empty());
    }

    // -- identity / round-trip ----------------------------------------------

    #[test]
    fn add_then_sub_should_return_original() {
        let a: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let b: Vec<f32> = (10..20).map(|i| i as f32).collect();
        let c = add(&a, &b);
        let back = sub(&c, &b);
        for (orig, result) in a.iter().zip(back.iter()) {
            assert!((orig - result).abs() < 0.001);
        }
    }

    #[test]
    fn scale_then_scale_by_reciprocal_should_return_original() {
        let a: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let s = 4.0;
        let scaled = scale(&a, s);
        let back = scale(&scaled, 1.0 / s);
        for (orig, result) in a.iter().zip(back.iter()) {
            assert!((orig - result).abs() < 0.001);
        }
    }

    #[test]
    fn fill_then_scale_should_maintain_uniformity() {
        let a = fill(100, 7.0);
        let b = scale(&a, 2.0);
        assert!(b.iter().all(|x| (*x - 14.0).abs() < f32::EPSILON));
    }

    #[test]
    fn arange_should_not_panic_on_large_n() {
        let c = arange(1_000_000, 0.0, 0.001);
        assert_eq!(c.len(), 1_000_000);
    }
}
