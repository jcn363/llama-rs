// Complete multi-head attention implementation with RoPE and KV cache.
// Supports MHA, GQA (Grouped Query Attention), and MQA (Multi-Query Attention).
// Includes flash attention for memory-efficient prefill.

use crate::kv_cache::KvCache;
use crate::{RoPEConfig, RopeScaleType, dot_product};

/// Flash attention: compute softmax(Q @ K^T) @ V in a single pass without materializing the full attention matrix.
/// Uses the online softmax trick: track running max and sum to compute softmax incrementally.
///
/// Memory complexity: O(N * head_dim) instead of O(N²)
/// where N = seq_len (context length)
///
/// Supports sliding window attention: when `window_size` is Some(n), only the
/// last `n` KV positions are attended to (Mistral/Mixtral architectures).
///
/// # Arguments
/// * `q` - Query vector of shape (1, head_dim) for current token
/// * `keys` - Flat key cache, shape (seq_len, n_head_kv, head_dim) row-major
/// * `values` - Flat value cache, shape (seq_len, n_head_kv, head_dim) row-major
/// * `seq_len` - Current sequence length (number of cached tokens)
/// * `head_dim` - Dimension of each head
/// * `n_head_kv` - Number of KV heads
/// * `head` - KV head index
/// * `window_size` - Optional sliding window size (None = full attention)
///
/// # Returns
/// Output vector of shape (1, head_dim)
#[expect(clippy::too_many_arguments)]
fn flash_attention_head(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_head_kv: usize,
    head: usize,
    window_size: Option<usize>,
) -> Vec<f32> {
    assert_eq!(q.len(), head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Sliding window: only attend to the last `window_size` positions
    let start = match window_size {
        Some(w) => seq_len.saturating_sub(w),
        None => 0,
    };

    // Online softmax: track running max and sum
    let mut max_val = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;
    let mut output = vec![0.0f32; head_dim];

    // Single pass: compute scores, update max/sum, accumulate weighted V
    for j in start..seq_len {
        let base = (j * n_head_kv + head) * head_dim;
        let k_row = &keys[base..base + head_dim];
        let score = dot_product(q, k_row) * scale;

        // Update running max and sum
        let prev_max = max_val;
        if score > max_val {
            max_val = score;
        }
        let exp_val = (score - max_val).exp();

        // Rescale previous output and sum
        let rescale = (prev_max - max_val).exp();
        sum_exp = sum_exp * rescale + exp_val;
        for o in &mut output[..head_dim] {
            *o *= rescale;
        }

        // Add weighted V
        let v_row = &values[base..base + head_dim];
        for d in 0..head_dim {
            output[d] += exp_val * v_row[d];
        }
    }

    // Normalize by sum
    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for o in &mut output[..head_dim] {
            *o *= inv_sum;
        }
    }

    output
}

/// Apply Rotary Position Embedding (RoPE) with optional scaling.
///
/// Supports Linear, NTK-aware, and Dynamic NTK scaling strategies, plus
/// partial rotation (used by Phi-3).
///
/// # Arguments
/// * `x` - Input vector of shape (seq_len, head_dim), flattened
/// * `seq_len` - Sequence length
/// * `head_dim` - Dimension of each head
/// * `position_offset` - Starting position (for KV cache continuity)
/// * `config` - RoPE configuration (theta, scaling type, etc.)
/// * `actual_seq_len` - Current context length for Dynamic NTK
pub fn apply_rope_with_config(
    x: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    position_offset: usize,
    config: &RoPEConfig,
    actual_seq_len: Option<usize>,
) {
    let rot_dim = config.partial_dim.unwrap_or(head_dim);
    let half_dim = rot_dim / 2;
    let rope_theta = config.theta;

    // Effective theta based on scaling type
    let effective_theta = match config.scale_type {
        RopeScaleType::NtkAware => {
            let scale = config.scale_factor.max(1.0);
            let d = head_dim as f32;
            rope_theta * scale.powf(d / (d - 2.0))
        }
        RopeScaleType::DynamicNtk => {
            let max_s = config.original_max_seq_len as f32;
            let actual = actual_seq_len.unwrap_or(seq_len + position_offset) as f32;
            let scale = (config.scale_factor * actual / max_s).max(1.0);
            rope_theta * scale.powf(head_dim as f32 / (head_dim as f32 - 2.0))
        }
        _ => rope_theta,
    };

    for pos in 0..seq_len {
        let actual_pos = position_offset + pos;

        // Compute effective position based on scaling type
        let eff_pos = match config.scale_type {
            RopeScaleType::Linear => actual_pos as f32 / config.scale_factor.max(1.0),
            RopeScaleType::DynamicNtk => {
                let max_s = config.original_max_seq_len as f32;
                let cur = (actual_pos + 1) as f32;
                let s = (config.scale_factor * cur / max_s).max(1.0);
                actual_pos as f32 / s
            }
            _ => actual_pos as f32,
        };

        let row_start = pos * head_dim;

        for i in 0..half_dim {
            let freq = 1.0 / effective_theta.powf(i as f32 / half_dim as f32);
            let theta = eff_pos * freq;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let idx1 = row_start + i;
            let idx2 = row_start + i + half_dim;

            let x1 = x[idx1];
            let x2 = x[idx2];

            // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
            x[idx1] = x1 * cos_theta - x2 * sin_theta;
            x[idx2] = x1 * sin_theta + x2 * cos_theta;
        }
    }
}

/// Apply Rotary Position Embedding (RoPE) with default (no-scaling) config.
///
/// Backward-compatible wrapper around [`apply_rope_with_config`].
#[cfg_attr(not(test), expect(dead_code))]
pub fn apply_rope(
    x: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    position_offset: usize,
    rope_theta: f32,
) {
    let config = RoPEConfig::new(rope_theta);
    apply_rope_with_config(x, seq_len, head_dim, position_offset, &config, None);
}

/// Compute scaled dot-product attention for a single head with causal masking.
/// (Legacy implementation, kept for reference. Flash attention is preferred.)
#[expect(dead_code)]
fn attention_head_with_cache(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    head_dim: usize,
    scores: &mut [f32],
) -> Vec<f32> {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k_cache.len(), seq_len * head_dim);
    assert_eq!(v_cache.len(), seq_len * head_dim);
    assert_eq!(scores.len(), seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // 1. Compute Q @ K^T for all cached positions
    let mut max_val = f32::NEG_INFINITY;
    for j in 0..seq_len {
        let k_row = &k_cache[j * head_dim..(j + 1) * head_dim];
        let val = dot_product(q, k_row) * scale;
        scores[j] = val;
        if val > max_val {
            max_val = val;
        }
    }

    // 2. Softmax with numerical stability
    let mut sum = 0.0f32;
    for s in &mut scores[..seq_len] {
        let exp_val = (*s - max_val).exp();
        *s = exp_val;
        sum += exp_val;
    }
    for s in &mut scores[..seq_len] {
        *s /= sum;
    }

    // 3. Weighted sum of V
    let mut out = vec![0.0f32; head_dim];
    for j in 0..seq_len {
        let weight = scores[j];
        let v_row = &v_cache[j * head_dim..(j + 1) * head_dim];
        for d in 0..head_dim {
            out[d] += weight * v_row[d];
        }
    }

    out
}

/// Multi-head attention with KV cache support.
///
/// # Arguments
/// * `n_head` - Number of query heads
/// * `n_head_kv` - Number of key/value heads (for GQA/MQA)
/// * `head_dim` - Dimension of each head
/// * `seq_len` - Current sequence length
/// * `position_offset` - Starting position in KV cache
/// * `q` - Query projections, shape (seq_len, n_head * head_dim)
/// * `k` - Key projections, shape (seq_len, n_head_kv * head_dim)
/// * `v` - Value projections, shape (seq_len, n_head_kv * head_dim)
/// * `kv_cache` - KV cache to store/retrieve keys and values
/// * `rope_theta` - RoPE base frequency
/// * `window_size` - Optional sliding window size (None = full attention)
///
/// # Returns
/// Attention output of shape (seq_len, n_head * head_dim)
#[expect(clippy::too_many_arguments)]
pub fn multi_head_attention_with_cache(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    seq_len: usize,
    position_offset: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    kv_cache: &mut KvCache,
    rope_config: &RoPEConfig,
    window_size: Option<usize>,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * n_head * head_dim);
    assert_eq!(k.len(), seq_len * n_head_kv * head_dim);
    assert_eq!(v.len(), seq_len * n_head_kv * head_dim);

    // Apply RoPE to Q and K
    apply_rope_with_config(q, seq_len, head_dim, position_offset, rope_config, None);
    apply_rope_with_config(k, seq_len, head_dim, position_offset, rope_config, None);

    // Store K and V in cache
    for pos in 0..seq_len {
        let k_offset = pos * n_head_kv * head_dim;
        let v_offset = pos * n_head_kv * head_dim;
        kv_cache.push(
            &k[k_offset..k_offset + n_head_kv * head_dim],
            &v[v_offset..v_offset + n_head_kv * head_dim],
        );
    }

    let total_seq_len = position_offset + seq_len;
    let n_rep = n_head / n_head_kv; // Number of query heads per KV head (for GQA)

    // Compute attention for each query head
    let mut output = vec![0.0f32; seq_len * n_head * head_dim];

    for h in 0..n_head {
        // For GQA, multiple query heads share the same KV head
        let kv_head = h / n_rep;

        for pos in 0..seq_len {
            // Get query for this position and head
            let q_offset = pos * n_head * head_dim + h * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];

            // Use flash attention directly on flat cache arrays (no Vec allocation)
            let out = flash_attention_head(
                q_vec,
                &kv_cache.keys,
                &kv_cache.values,
                total_seq_len,
                head_dim,
                n_head_kv,
                kv_head,
                window_size,
            );

            // Store output
            let out_offset = pos * n_head * head_dim + h * head_dim;
            output[out_offset..out_offset + head_dim].copy_from_slice(&out);
        }
    }

    output
}

/// Flash attention for prefill phase (multi-token processing).
///
/// Uses online softmax to compute attention in a single pass without
/// materializing the full seq_len × seq_len attention matrix.
/// Memory complexity: O(seq_len * head_dim) instead of O(seq_len²).
///
/// # Arguments
/// * `n_head` - Number of query heads
/// * `n_head_kv` - Number of key/value heads (for GQA/MQA)
/// * `head_dim` - Dimension of each head
/// * `seq_len` - Sequence length
/// * `q` - Query projections, shape (seq_len, n_head * head_dim)
/// * `k` - Key projections, shape (seq_len, n_head_kv * head_dim)
/// * `v` - Value projections, shape (seq_len, n_head_kv * head_dim)
/// * `rope_config` - RoPE configuration
/// * `window_size` - Optional sliding window size (None = full attention)
///
/// # Returns
/// Attention output of shape (seq_len, n_head * head_dim)
#[expect(clippy::too_many_arguments)]
// Ready for production prefill wiring; currently used only in tests.
#[cfg_attr(not(test), expect(dead_code))]
pub fn multi_head_attention_prefill(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    seq_len: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    rope_config: &RoPEConfig,
    window_size: Option<usize>,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * n_head * head_dim);
    assert_eq!(k.len(), seq_len * n_head_kv * head_dim);
    assert_eq!(v.len(), seq_len * n_head_kv * head_dim);

    // Apply RoPE to Q and K
    apply_rope_with_config(q, seq_len, head_dim, 0, rope_config, None);
    apply_rope_with_config(k, seq_len, head_dim, 0, rope_config, None);

    let n_rep = n_head / n_head_kv;
    let mut output = vec![0.0f32; seq_len * n_head * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_head {
        let kv_head = h / n_rep;

        for i in 0..seq_len {
            let q_offset = i * n_head * head_dim + h * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];

            // Online softmax: single pass over causal context with optional sliding window
            let window_start = match window_size {
                Some(w) => (i + 1).saturating_sub(w),
                None => 0,
            };
            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let out_offset = i * n_head * head_dim + h * head_dim;
            let mut out_vec = vec![0.0f32; head_dim];

            for j in window_start..=i {
                let k_offset = j * n_head_kv * head_dim + kv_head * head_dim;
                let k_vec = &k[k_offset..k_offset + head_dim];
                let score = dot_product(q_vec, k_vec) * scale;

                let prev_max = max_val;
                if score > max_val {
                    max_val = score;
                }
                let exp_val = (score - max_val).exp();

                // Rescale
                let rescale = (prev_max - max_val).exp();
                sum_exp = sum_exp * rescale + exp_val;
                for o in &mut out_vec[..head_dim] {
                    *o *= rescale;
                }

                // Accumulate weighted V
                let v_offset = j * n_head_kv * head_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    out_vec[d] += exp_val * v[v_offset + d];
                }
            }

            // Normalize
            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                for o in &mut out_vec[..head_dim] {
                    *o *= inv_sum;
                }
            }

            output[out_offset..out_offset + head_dim].copy_from_slice(&out_vec);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_basic() {
        let head_dim = 4;
        let seq_len = 2;
        let mut x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

        apply_rope(&mut x, seq_len, head_dim, 0, 10000.0);

        // Position 0: theta = 0, cos(0) = 1, sin(0) = 0, no change
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 0.0).abs() < 1e-6);
        assert!((x[2] - 0.0).abs() < 1e-6);
        assert!((x[3] - 1.0).abs() < 1e-6);

        // Position 1: should be rotated
        // The values should have changed from the original
        assert!((x[4] - 1.0).abs() > 1e-6 || (x[5] - 0.0).abs() > 1e-6);
    }

    #[test]
    fn test_attention_prefill() {
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let seq_len = 3;

        let mut q = vec![0.1; seq_len * n_head * head_dim];
        let mut k = vec![0.1; seq_len * n_head_kv * head_dim];
        let v = vec![0.1; seq_len * n_head_kv * head_dim];

        let config = RoPEConfig::new(10000.0);
        let output = multi_head_attention_prefill(
            n_head, n_head_kv, head_dim, seq_len, &mut q, &mut k, &v, &config, None,
        );

        assert_eq!(output.len(), seq_len * n_head * head_dim);
        // Output should be non-zero
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_attention_prefill_with_window() {
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let seq_len = 4;

        let mut q = vec![0.1; seq_len * n_head * head_dim];
        let mut k = vec![0.1; seq_len * n_head_kv * head_dim];
        let v = vec![0.2; seq_len * n_head_kv * head_dim];

        let config = RoPEConfig::new(10000.0);

        // Without window
        let output_full = multi_head_attention_prefill(
            n_head, n_head_kv, head_dim, seq_len, &mut q, &mut k, &v, &config, None,
        );
        // With window=2 (each position only attends to last 2 tokens)
        let mut q2 = q.clone();
        let mut k2 = k.clone();
        let output_windowed = multi_head_attention_prefill(
            n_head,
            n_head_kv,
            head_dim,
            seq_len,
            &mut q2,
            &mut k2,
            &v,
            &config,
            Some(2),
        );

        // Both should have correct shape
        assert_eq!(output_full.len(), seq_len * n_head * head_dim);
        assert_eq!(output_windowed.len(), seq_len * n_head * head_dim);
        // Windowed output should differ from full (less context seen)
        assert_ne!(output_full, output_windowed);
    }

    #[test]
    fn test_rope_with_scaling_linear() {
        let head_dim = 4;
        let seq_len = 2;
        let mut x_vanilla = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut x_scaled = x_vanilla.clone();

        apply_rope(&mut x_vanilla, seq_len, head_dim, 0, 10000.0);

        // Linear scaling with factor 2 => position halved => pos=1 acts like pos=0.5
        let config = RoPEConfig {
            theta: 10000.0,
            scale_type: RopeScaleType::Linear,
            scale_factor: 2.0,
            original_max_seq_len: 4096,
            partial_dim: None,
        };
        apply_rope_with_config(&mut x_scaled, seq_len, head_dim, 0, &config, None);

        // Scaled should differ from vanilla
        assert_ne!(x_vanilla, x_scaled);
    }

    #[test]
    fn test_rope_with_ntk() {
        let head_dim = 4;
        let seq_len = 2;
        let mut x_vanilla = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut x_ntk = x_vanilla.clone();

        apply_rope(&mut x_vanilla, seq_len, head_dim, 0, 10000.0);

        // NTK scaling: theta is adjusted
        let config = RoPEConfig {
            theta: 10000.0,
            scale_type: RopeScaleType::NtkAware,
            scale_factor: 2.0,
            original_max_seq_len: 4096,
            partial_dim: None,
        };
        apply_rope_with_config(&mut x_ntk, seq_len, head_dim, 0, &config, None);

        assert_ne!(x_vanilla, x_ntk);
    }

    #[test]
    fn test_rope_partial_rotation() {
        let head_dim = 4;
        let seq_len = 1;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];

        // Partial rotation: only first 2 dims rotated, last 2 untouched
        let config = RoPEConfig {
            theta: 10000.0,
            scale_type: RopeScaleType::None,
            scale_factor: 1.0,
            original_max_seq_len: 4096,
            partial_dim: Some(2),
        };
        apply_rope_with_config(&mut x, seq_len, head_dim, 5, &config, None);

        // dim 0 and 1 rotated (values changed from input)
        assert!((x[0] - 1.0).abs() > 1e-6 || (x[1] - 2.0).abs() > 1e-6);
        // dim 2 and 3 untouched (3.0, 4.0)
        assert!((x[2] - 3.0).abs() < 1e-6);
        assert!((x[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_config_default_equals_vanilla() {
        let head_dim = 4;
        let seq_len = 2;
        let mut x_vanilla = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut x_config = x_vanilla.clone();

        apply_rope(&mut x_vanilla, seq_len, head_dim, 0, 10000.0);
        let config = RoPEConfig::new(10000.0);
        apply_rope_with_config(&mut x_config, seq_len, head_dim, 0, &config, None);

        assert_eq!(x_vanilla, x_config);
    }
}
