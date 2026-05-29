use crate::tensor::Tensor;
use llama_core::backend::QuantType;

fn dequantize_block(block: &[u8], quant_type: QuantType, out: &mut [f32]) {
    let block_size = quant_type.block_size();
    assert_eq!(out.len(), block_size, "output slice must match block size");
    match quant_type {
        QuantType::Q4_0 => {
            assert!(block.len() >= 18);
            let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for i in 0..16 {
                let byte = block[2 + i];
                let lo = ((byte & 0x0F) as i8).wrapping_sub(8);
                let hi = ((byte >> 4) as i8).wrapping_sub(8);
                out[i * 2] = f32::from(lo) * scale;
                out[i * 2 + 1] = f32::from(hi) * scale;
            }
        }
        QuantType::Q4_1 => {
            assert!(block.len() >= 20);
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
            for i in 0..16 {
                let byte = block[4 + i];
                let lo = (byte & 0x0F) as i8;
                let hi = (byte >> 4) as i8;
                out[i * 2] = f32::from(lo) * d + m;
                out[i * 2 + 1] = f32::from(hi) * d + m;
            }
        }
        QuantType::Q8_0 => {
            assert!(block.len() >= 34);
            let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            for i in 0..32 {
                out[i] = f32::from(block[2 + i] as i8) * scale;
            }
        }
    }
}

/// Default quantized matrix-vector product.
///
/// Dequantizes each row block-by-block and computes dot products.
/// This is a portable fallback — backends with format-specific kernels
/// should override [`Backend::mat_vec_quant`].
pub fn default_mat_vec_quant(
    weight: &[u8],
    quant_type: QuantType,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Vec<f32> {
    let block_size = quant_type.block_size();
    let block_bytes = quant_type.block_bytes();
    let n_blocks_per_row = cols.div_ceil(block_size);
    let mut dequant_buf = vec![0.0f32; block_size];
    (0..rows)
        .map(|r| {
            let mut sum = 0.0f32;
            for b in 0..n_blocks_per_row {
                let block_off = (r * n_blocks_per_row + b) * block_bytes;
                // Ensure we don't read past the buffer
                let block_end = (block_off + block_bytes).min(weight.len());
                dequantize_block(&weight[block_off..block_end], quant_type, &mut dequant_buf);
                let col_start = b * block_size;
                let n = cols.saturating_sub(col_start).min(block_size);
                for i in 0..n {
                    sum += dequant_buf[i] * input[col_start + i];
                }
            }
            sum
        })
        .collect()
}

/// Default matrix-vector product implementation using sequential iteration.
///
/// This is a portable fallback used by backends that don't accelerate
/// matrix-vector multiplication (or for small matrices where dispatch
/// overhead isn't worth it).
pub fn default_mat_vec(weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let start = r * cols;
            let row = &weight[start..start + cols];
            row.iter().zip(input.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Default matrix-matrix product.
pub fn default_mat_mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert_eq!(a_shape.len(), 2, "A must be 2-dimensional");
    assert_eq!(b_shape.len(), 2, "B must be 2-dimensional");
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let k2 = b_shape[1];
    assert_eq!(k, k2, "inner dimensions must match");
    let a_f32: &[f32] = bytemuck::cast_slice(a.data());
    let b_f32: &[f32] = bytemuck::cast_slice(b.data());
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a_f32[i * k + l] * b_f32[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    Tensor::from_f32(&[m, n], &c)
}

/// Default outer product.
pub fn default_out_prod(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for &ai in a {
        for &bi in b {
            out.push(ai * bi);
        }
    }
    out
}

/// Default identity-aware matmul.
pub fn default_mat_mul_id(
    a: &[f32],
    b: &[f32],
    a_rows: usize,
    _a_cols: usize,
    b_cols: usize,
) -> Vec<f32> {
    // Treat `a` as a diagonal matrix of size `a_rows`. Multiply each diagonal
    // element with the corresponding row of `b`.
    let mut out = vec![0.0f32; a_rows * b_cols];
    for i in 0..a_rows {
        let scale = a.get(i).copied().unwrap_or(0.0);
        for j in 0..b_cols {
            let b_idx = i * b_cols + j;
            let b_val = b.get(b_idx).copied().unwrap_or(0.0);
            out[i * b_cols + j] = scale * b_val;
        }
    }
    out
}

/// Default Hadamard product.
pub fn default_mul_mat_hadamard(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

// ── Default unary element-wise ──

/// Default absolute value.
pub fn default_abs(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.abs()).collect()
}

/// Default sign function.
pub fn default_sgn(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| {
            if *v > 0.0 {
                1.0
            } else if *v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Default negation.
pub fn default_neg(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| -v).collect()
}

/// Default step function.
pub fn default_step(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| if *v > 0.0 { 1.0 } else { 0.0 }).collect()
}

/// Default tanh.
pub fn default_tanh(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.tanh()).collect()
}

/// Default ELU.
pub fn default_elu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| if *v > 0.0 { *v } else { v.exp() - 1.0 })
        .collect()
}

/// Default ReLU.
pub fn default_relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

/// Default sigmoid.
pub fn default_sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// Default hard sigmoid.
pub fn default_hard_sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| (v / 6.0 + 0.5).clamp(0.0, 1.0)).collect()
}

/// Default hard swish.
pub fn default_hard_swish(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| v * (v / 6.0 + 0.5).clamp(0.0, 1.0))
        .collect()
}

/// Default exponential.
pub fn default_exp(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.exp()).collect()
}

/// Default expm1.
pub fn default_expm1(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.exp_m1()).collect()
}

/// Default softplus.
pub fn default_softplus(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| (1.0 + v.exp()).ln()).collect()
}

/// Default floor.
pub fn default_floor(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.floor()).collect()
}

/// Default ceil.
pub fn default_ceil(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.ceil()).collect()
}

/// Default round.
pub fn default_round(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.round()).collect()
}

/// Default truncate.
pub fn default_trunc(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.trunc()).collect()
}

/// Default sine.
pub fn default_sin(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.sin()).collect()
}

/// Default cosine.
pub fn default_cos(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.cos()).collect()
}

/// Default square.
pub fn default_sqr(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v * v).collect()
}

/// Default square root.
pub fn default_sqrt(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.sqrt()).collect()
}

/// Default SiLU activation.
pub fn default_silu(x: &[f32]) -> Vec<f32> {
    // SiLU (also called Swish) = x * sigmoid(x)
    x.iter()
        .map(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            v * s
        })
        .collect()
}

/// Default SiLU backward.
pub fn default_silu_back(x: &[f32]) -> Vec<f32> {
    // derivative of silu = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    x.iter()
        .map(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            s * (1.0 + v * (1.0 - s))
        })
        .collect()
}

/// Default leaky ReLU.
pub fn default_leaky_relu(x: &[f32], negative_slope: f32) -> Vec<f32> {
    x.iter()
        .map(|v| if *v > 0.0 { *v } else { v * negative_slope })
        .collect()
}

/// Default GELU Erf approximation.
pub fn default_gelu_erf(x: &[f32]) -> Vec<f32> {
    use libm;
    use std::f32::consts::FRAC_1_SQRT_2;

    x.iter()
        .map(|v| {
            let erf = libm::erff(v * FRAC_1_SQRT_2);
            0.5 * v * (1.0 + erf)
        })
        .collect()
}

/// Default GELU activation (standard Erf based implementation).
pub fn default_gelu(x: &[f32]) -> Vec<f32> {
    // GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
    use libm;
    use std::f32::consts::FRAC_1_SQRT_2;
    x.iter()
        .map(|v| 0.5 * v * (1.0 + libm::erff(v * FRAC_1_SQRT_2)))
        .collect()
}

/// Default GELU Quick approximation.
pub fn default_gelu_quick(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| v * (1.0 / (1.0 + (-1.702 * v).exp())))
        .collect()
}

// ── Default binary element-wise ──

/// Default element-wise addition.
pub fn default_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Default element-wise subtraction.
pub fn default_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Default element-wise multiplication.
pub fn default_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Default element-wise division.
pub fn default_div(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

/// Default ADD1 (scalar broadcast).
pub fn default_add1(a: &[f32], b: f32) -> Vec<f32> {
    a.iter().map(|x| x + b).collect()
}

// ── Default gated activations ──

/// Default SwiGLU.
pub fn default_swiglu(a: &[f32], b: &[f32], beta: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let s = 1.0 / (1.0 + (-x * beta).exp());
            s * x * y
        })
        .collect()
}

/// Default SwiGLU OAI.
pub fn default_swiglu_oai(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let s = 1.0 / (1.0 + (-x).exp());
            s * x * y
        })
        .collect()
}

/// Default GEGLU.
pub fn default_geglu(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let sqrt_2_over_pi = 0.797_884_6;
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            let tanh = (inner.exp() - (-inner).exp()) / (inner.exp() + (-inner).exp());
            let g = 0.5 * x * (1.0 + tanh);
            g * y
        })
        .collect()
}

/// Default REGLU.
pub fn default_reglu(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.max(0.0) * y)
        .collect()
}

/// Default GEGLU Erf.
pub fn default_geglu_erf(a: &[f32], b: &[f32]) -> Vec<f32> {
    use libm;
    use std::f32::consts::FRAC_1_SQRT_2;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let g = 0.5 * x * (1.0 + libm::erff(x * FRAC_1_SQRT_2));
            g * y
        })
        .collect()
}

/// Default GEGLU Quick.
pub fn default_geglu_quick(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let g = x * (1.0 / (1.0 + (-1.702 * x).exp()));
            g * y
        })
        .collect()
}

/// Default Xi approximation.
pub fn default_xielu(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let s = 1.0 / (1.0 + (-x).exp());
            s * x * y
        })
        .collect()
}

// ── Default normalization ──

/// Default RMSNorm implementation.
pub fn default_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ssq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ssq / n as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter().cycle())
        .map(|(&xi, &wi)| wi * (xi / rms))
        .collect()
}

/// Default RMSNorm backward.
pub fn default_rms_norm_back(x: &[f32], weight: &[f32], grad: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ssq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ssq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let n_inv = 1.0 / n as f32;
    grad.iter()
        .enumerate()
        .map(|(i, &g)| {
            let xi = x[i];
            let wi = weight[i % weight.len()];
            // d(L)/dx_i = (g * w_i / rms) - (g * x_i * w_i * x_i / (n * rms^3)) summed over all i
            // Simplified per-element contribution:
            let x_norm = xi * inv_rms;
            let dl_dxnorm = g * wi;
            dl_dxnorm * inv_rms - (x_norm * dl_dxnorm * xi * n_inv * inv_rms / (rms * rms))
        })
        .collect()
}

/// Default layer normalization.
pub fn default_norm(x: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    match bias {
        Some(bias) => x
            .iter()
            .enumerate()
            .map(|(i, v)| (v - mean) * inv_std * weight[i % weight.len()] + bias[i % bias.len()])
            .collect(),
        None => x
            .iter()
            .enumerate()
            .map(|(i, v)| (v - mean) * inv_std * weight[i % weight.len()])
            .collect(),
    }
}

/// Default group normalization.
pub fn default_group_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    n_groups: usize,
) -> Vec<f32> {
    let n = x.len();
    let per_group = n / n_groups;
    let mut out = vec![0.0f32; n];
    for g in 0..n_groups {
        let start = g * per_group;
        let end = start + per_group;
        let group = &x[start..end];
        let mean: f32 = group.iter().sum::<f32>() / per_group as f32;
        let var: f32 = group.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / per_group as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, &v) in group.iter().enumerate() {
            out[start + i] = (v - mean) * inv_std * weight[i % weight.len()] + bias[i % bias.len()];
        }
    }
    out
}

/// Default L2 normalization.
pub fn default_l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let norm = (sum_sq + eps).sqrt();
    x.iter().map(|v| v / norm).collect()
}

// ── Default reduction ──

/// Default sum.
pub fn default_sum(x: &[f32]) -> f32 {
    x.iter().sum()
}

/// Default mean.
pub fn default_mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f32>() / x.len() as f32
}

/// Default argmax.
pub fn default_argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Default count equal.
pub fn default_count_equal(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (*x - *y).abs() < f32::EPSILON)
        .count()
}

/// Default cumulative sum.
pub fn default_cumsum(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let mut running = 0.0f32;
    for &v in x {
        running += v;
        out.push(running);
    }
    out
}

/// Default sum rows.
pub fn default_sum_rows(x: &[f32], cols: usize) -> Vec<f32> {
    x.chunks(cols).map(|row| row.iter().sum()).collect()
}

/// Default softmax.
pub fn default_soft_max(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Default softmax backward.
pub fn default_soft_max_back(output: &[f32], grad: &[f32]) -> Vec<f32> {
    output
        .iter()
        .zip(grad.iter())
        .map(|(o, g)| {
            o * (g - output
                .iter()
                .zip(grad.iter())
                .map(|(oi, gi)| oi * gi)
                .sum::<f32>())
        })
        .collect()
}

/// Default cross-entropy loss.
pub fn default_cross_entropy_loss(prediction: &[f32], target: &[f32]) -> f32 {
    -target
        .iter()
        .zip(prediction.iter())
        .map(|(t, p)| t * p.ln().max(-100.0))
        .sum::<f32>()
}

/// Default cross-entropy loss backward.
pub fn default_cross_entropy_loss_back(prediction: &[f32], target: &[f32]) -> Vec<f32> {
    prediction
        .iter()
        .zip(target.iter())
        .map(|(p, t)| p - t)
        .collect()
}

// ── Default shape manipulation ──

/// Default concatenate.
pub fn default_concat(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

/// Default repeat.
pub fn default_repeat(x: &[f32], n_repeats: usize, block_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() * n_repeats);
    for chunk in x.chunks(block_size) {
        for _ in 0..n_repeats {
            out.extend_from_slice(chunk);
        }
    }
    out
}

/// Default repeat backward.
pub fn default_repeat_back(x: &[f32], n_repeats: usize, block_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() / n_repeats);
    for chunk in x.chunks(block_size * n_repeats) {
        let summed: Vec<f32> = chunk
            .chunks(block_size)
            .fold(vec![0.0f32; block_size], |acc, c| {
                acc.iter().zip(c.iter()).map(|(a, b)| a + b).collect()
            });
        out.extend_from_slice(&summed);
    }
    out
}

/// Default pad.
pub fn default_pad(x: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(target_len);
    out.extend_from_slice(x);
    out.resize(target_len, 0.0);
    out
}

/// Default reflect 1D padding.
pub fn default_pad_reflect_1d(x: &[f32], left_pad: usize, right_pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + left_pad + right_pad);
    // Left reflection
    for i in (0..left_pad).rev() {
        out.push(x[i.min(n - 1)]);
    }
    out.extend_from_slice(x);
    // Right reflection
    for i in 0..right_pad {
        out.push(x[n - 1 - i.min(n - 1)]);
    }
    out
}

/// Default roll (circular shift right).
pub fn default_roll(x: &[f32], shift: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let shift = shift % n;
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&x[n - shift..]);
    out.extend_from_slice(&x[..n - shift]);
    out
}

/// Default diag (create diagonal matrix).
pub fn default_diag(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0f32; n * n];
    for (i, &v) in x.iter().enumerate() {
        out[i * n + i] = v;
    }
    out
}

/// Default diag mask -inf.
pub fn default_diag_mask_inf(x: &[f32], n_cols: usize, offset: i32) -> Vec<f32> {
    let n_rows = x.len() / n_cols;
    let mut out = x.to_vec();
    for row in 0..n_rows {
        for col in 0..n_cols {
            let d = col as i32 - row as i32 - offset;
            if d > 0 {
                out[row * n_cols + col] = f32::NEG_INFINITY;
            }
        }
    }
    out
}

/// Default duplicate.
pub fn default_dup(x: &[f32]) -> Vec<f32> {
    x.to_vec()
}

/// Default contiguous.
pub fn default_cont(x: &[f32]) -> Vec<f32> {
    x.to_vec()
}

/// Default get rows.
pub fn default_get_rows(x: &[f32], indices: &[i32], n_cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * n_cols);
    for &idx in indices {
        let start = (idx as usize) * n_cols;
        let end = start + n_cols;
        if end <= x.len() {
            out.extend_from_slice(&x[start..end]);
        }
    }
    out
}

/// Default get rows backward.
pub fn default_get_rows_back(
    grad: &[f32],
    indices: &[i32],
    n_cols: usize,
    n_src_rows: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_src_rows * n_cols];
    for (i, &idx) in indices.iter().enumerate() {
        let start = i * n_cols;
        let row_start = (idx as usize) * n_cols;
        for j in 0..n_cols {
            out[row_start + j] += grad[start + j];
        }
    }
    out
}

/// Default set rows.
pub fn default_set_rows(dst: &[f32], src: &[f32], indices: &[i32], n_cols: usize) -> Vec<f32> {
    let mut out = dst.to_vec();
    for (i, &idx) in indices.iter().enumerate() {
        let dst_start = (idx as usize) * n_cols;
        let src_start = i * n_cols;
        out[dst_start..dst_start + n_cols].copy_from_slice(&src[src_start..src_start + n_cols]);
    }
    out
}

// ── Default convolution ──

/// Default 2D convolution (NCHW).
#[expect(clippy::too_many_arguments)]
pub fn default_conv_2d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    _kc: usize,
    kh: usize,
    kw: usize,
    filters: usize,
    s0: usize,
    s1: usize,
    p0: usize,
    p1: usize,
) -> Vec<f32> {
    let out_h = ((h + 2 * p0 - kh) / s0) + 1;
    let out_w = ((w + 2 * p1 - kw) / s1) + 1;
    let mut out = vec![0.0f32; n * filters * out_h * out_w];
    for batch in 0..n {
        for f in 0..filters {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for kc_ in 0..c {
                        for kh_ in 0..kh {
                            for kw_ in 0..kw {
                                let ih = oh * s0 + kh_;
                                let iw = ow * s1 + kw_;
                                if ih < p0 || ih >= h + p0 || iw < p1 || iw >= w + p1 {
                                    continue;
                                }
                                let x_idx =
                                    batch * (c * h * w) + kc_ * (h * w) + (ih - p0) * w + (iw - p1);
                                let k_idx = f * (c * kh * kw) + kc_ * (kh * kw) + kh_ * kw + kw_;
                                sum += x[x_idx] * kernel[k_idx];
                            }
                        }
                    }
                    out[batch * (filters * out_h * out_w)
                        + f * (out_h * out_w)
                        + oh * out_w
                        + ow] = sum;
                }
            }
        }
    }
    out
}

/// Default 2D depthwise convolution.
#[expect(clippy::too_many_arguments)]
pub fn default_conv_2d_dw(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    s0: usize,
    s1: usize,
    p0: usize,
    p1: usize,
) -> Vec<f32> {
    let out_h = ((h + 2 * p0 - kh) / s0) + 1;
    let out_w = ((w + 2 * p1 - kw) / s1) + 1;
    let mut out = vec![0.0f32; n * c * out_h * out_w];
    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for kh_ in 0..kh {
                        for kw_ in 0..kw {
                            let ih = oh * s0 + kh_;
                            let iw = ow * s1 + kw_;
                            if ih < p0 || ih >= h + p0 || iw < p1 || iw >= w + p1 {
                                continue;
                            }
                            let x_idx =
                                batch * (c * h * w) + ch * (h * w) + (ih - p0) * w + (iw - p1);
                            let k_idx = ch * (kh * kw) + kh_ * kw + kw_;
                            sum += x[x_idx] * kernel[k_idx];
                        }
                    }
                    out[batch * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow] = sum;
                }
            }
        }
    }
    out
}

/// Default 1D transposed convolution.
#[expect(clippy::too_many_arguments)]
pub fn default_conv_transpose_1d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    len: usize,
    _kc: usize,
    kw: usize,
    filters: usize,
    s: usize,
    p: usize,
) -> Vec<f32> {
    let out_len = (len - 1) * s + kw - 2 * p;
    let mut out = vec![0.0f32; n * filters * out_len];
    for batch in 0..n {
        for f in 0..filters {
            for i in 0..len {
                for kp in 0..kw {
                    let out_pos = i * s + kp;
                    if out_pos >= p as isize as usize && out_pos < out_len + p {
                        let actual_pos = out_pos - p;
                        if actual_pos < out_len {
                            let x_idx = batch * (c * len) + i;
                            let k_idx = f * (kw) + kp;
                            out[batch * (filters * out_len) + f * out_len + actual_pos] +=
                                x[x_idx % x.len()] * kernel[k_idx % kernel.len()];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Default 2D transposed convolution.
#[expect(clippy::too_many_arguments)]
pub fn default_conv_transpose_2d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    _kc: usize,
    kh: usize,
    kw: usize,
    filters: usize,
    s0: usize,
    s1: usize,
    p0: usize,
    p1: usize,
) -> Vec<f32> {
    let out_h = (h - 1) * s0 + kh - 2 * p0;
    let out_w = (w - 1) * s1 + kw - 2 * p1;
    let mut out = vec![0.0f32; n * filters * out_h * out_w];
    for batch in 0..n {
        for f in 0..filters {
            for ih in 0..h {
                for iw in 0..w {
                    for kh_ in 0..kh {
                        for kw_ in 0..kw {
                            let oh = ih * s0 + kh_;
                            let ow = iw * s1 + kw_;
                            if oh >= p0 && oh < out_h + p0 && ow >= p1 && ow < out_w + p1 {
                                let actual_oh = oh - p0;
                                let actual_ow = ow - p1;
                                let x_idx = batch * (c * h * w) + ih * w + iw;
                                let k_idx = f * (c * kh * kw) + kh_ * kw + kw_;
                                out[batch * (filters * out_h * out_w)
                                    + f * (out_h * out_w)
                                    + actual_oh * out_w
                                    + actual_ow] +=
                                    x[x_idx % x.len()] * kernel[k_idx % kernel.len()];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Default im2col.
#[expect(clippy::too_many_arguments)]
pub fn default_im2col(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    s0: usize,
    s1: usize,
    p0: usize,
    p1: usize,
) -> Vec<f32> {
    let out_h = ((h + 2 * p0 - kh) / s0) + 1;
    let out_w = ((w + 2 * p1 - kw) / s1) + 1;
    let col_len = c * kh * kw;
    let mut out = vec![0.0f32; n * col_len * out_h * out_w];
    for batch in 0..n {
        for col_idx in 0..col_len {
            let kc_ = col_idx / (kh * kw);
            let kh_ = (col_idx % (kh * kw)) / kw;
            let kw_ = col_idx % kw;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih = oh * s0 + kh_;
                    let iw = ow * s1 + kw_;
                    if ih >= p0 && ih < h + p0 && iw >= p1 && iw < w + p1 {
                        let x_idx = batch * (c * h * w) + kc_ * (h * w) + (ih - p0) * w + (iw - p1);
                        out[batch * (col_len * out_h * out_w)
                            + col_idx * (out_h * out_w)
                            + oh * out_w
                            + ow] = x[x_idx];
                    }
                }
            }
        }
    }
    out
}

/// Default 1D pooling.
#[expect(clippy::too_many_arguments)]
pub fn default_pool_1d(
    x: &[f32],
    n: usize,
    c: usize,
    len: usize,
    kw: usize,
    s: usize,
    p: usize,
    is_max: bool,
) -> Vec<f32> {
    let out_len = ((len + 2 * p - kw) / s) + 1;
    let mut out = vec![0.0f32; n * c * out_len];
    for batch in 0..n {
        for ch in 0..c {
            for oi in 0..out_len {
                let start = oi * s;
                let mut val = if is_max { f32::NEG_INFINITY } else { 0.0f32 };
                let mut count = 0;
                for ki in 0..kw {
                    let ii = start + ki;
                    if ii >= p && ii < len + p {
                        let x_idx = batch * (c * len) + ch * len + (ii - p);
                        if is_max {
                            val = val.max(x[x_idx]);
                        } else {
                            val += x[x_idx];
                            count += 1;
                        }
                    }
                }
                if !is_max && count > 0 {
                    val /= count as f32;
                }
                out[batch * (c * out_len) + ch * out_len + oi] = val;
            }
        }
    }
    out
}

/// Default 2D pooling.
#[expect(clippy::too_many_arguments)]
pub fn default_pool_2d(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    s0: usize,
    s1: usize,
    p0: usize,
    p1: usize,
    is_max: bool,
) -> Vec<f32> {
    let out_h = ((h + 2 * p0 - kh) / s0) + 1;
    let out_w = ((w + 2 * p1 - kw) / s1) + 1;
    let mut out = vec![0.0f32; n * c * out_h * out_w];
    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let start_h = oh * s0;
                    let start_w = ow * s1;
                    let mut val = if is_max { f32::NEG_INFINITY } else { 0.0f32 };
                    let mut count = 0;
                    for kh_ in 0..kh {
                        for kw_ in 0..kw {
                            let ih = start_h + kh_;
                            let iw = start_w + kw_;
                            if ih >= p0 && ih < h + p0 && iw >= p1 && iw < w + p1 {
                                let x_idx =
                                    batch * (c * h * w) + ch * (h * w) + (ih - p0) * w + (iw - p1);
                                if is_max {
                                    val = val.max(x[x_idx]);
                                } else {
                                    val += x[x_idx];
                                    count += 1;
                                }
                            }
                        }
                    }
                    if !is_max && count > 0 {
                        val /= count as f32;
                    }
                    out[batch * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow] = val;
                }
            }
        }
    }
    out
}

/// Default 3D convolution.
#[expect(clippy::too_many_arguments)]
pub fn default_conv_3d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    _kc: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    filters: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    p0: usize,
    p1: usize,
    p2: usize,
) -> Vec<f32> {
    let out_d = ((d + 2 * p0 - kd) / s0) + 1;
    let out_h = ((h + 2 * p1 - kh) / s1) + 1;
    let out_w = ((w + 2 * p2 - kw) / s2) + 1;
    let mut out = vec![0.0f32; n * filters * out_d * out_h * out_w];
    for batch in 0..n {
        for f in 0..filters {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for kc_ in 0..c {
                            for kd_ in 0..kd {
                                for kh_ in 0..kh {
                                    for kw_ in 0..kw {
                                        let id = od * s0 + kd_;
                                        let ih = oh * s1 + kh_;
                                        let iw = ow * s2 + kw_;
                                        if id < p0
                                            || id >= d + p0
                                            || ih < p1
                                            || ih >= h + p1
                                            || iw < p2
                                            || iw >= w + p2
                                        {
                                            continue;
                                        }
                                        let x_idx = batch * (c * d * h * w)
                                            + kc_ * (d * h * w)
                                            + (id - p0) * (h * w)
                                            + (ih - p1) * w
                                            + (iw - p2);
                                        let k_idx = f * (c * kd * kh * kw)
                                            + kc_ * (kd * kh * kw)
                                            + kd_ * (kh * kw)
                                            + kh_ * kw
                                            + kw_;
                                        sum += x[x_idx] * kernel[k_idx];
                                    }
                                }
                            }
                        }
                        out[batch * (filters * out_d * out_h * out_w)
                            + f * (out_d * out_h * out_w)
                            + od * (out_h * out_w)
                            + oh * out_w
                            + ow] = sum;
                    }
                }
            }
        }
    }
    out
}

/// Default 3D im2col.
#[expect(clippy::too_many_arguments)]
pub fn default_im2col_3d(
    x: &[f32],
    n: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    p0: usize,
    p1: usize,
    p2: usize,
) -> Vec<f32> {
    let out_d = ((d + 2 * p0 - kd) / s0) + 1;
    let out_h = ((h + 2 * p1 - kh) / s1) + 1;
    let out_w = ((w + 2 * p2 - kw) / s2) + 1;
    let col_len = c * kd * kh * kw;
    let mut out = vec![0.0f32; n * col_len * out_d * out_h * out_w];
    for batch in 0..n {
        for col_idx in 0..col_len {
            let kc_ = col_idx / (kd * kh * kw);
            let kd_ = (col_idx % (kd * kh * kw)) / (kh * kw);
            let kh_ = (col_idx % (kh * kw)) / kw;
            let kw_ = col_idx % kw;
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let id = od * s0 + kd_;
                        let ih = oh * s1 + kh_;
                        let iw = ow * s2 + kw_;
                        if id >= p0
                            && id < d + p0
                            && ih >= p1
                            && ih < h + p1
                            && iw >= p2
                            && iw < w + p2
                        {
                            let x_idx = batch * (c * d * h * w)
                                + kc_ * (d * h * w)
                                + (id - p0) * (h * w)
                                + (ih - p1) * w
                                + (iw - p2);
                            out[batch * (col_len * out_d * out_h * out_w)
                                + col_idx * (out_d * out_h * out_w)
                                + od * (out_h * out_w)
                                + oh * out_w
                                + ow] = x[x_idx];
                        }
                    }
                }
            }
        }
    }
    out
}

// ── Default special operations ──

/// Default Rotary Position Embedding.
pub fn default_rope(
    x: &[f32],
    positions: &[i32],
    n_dims: usize,
    n_heads: usize,
    theta: f32,
    _mode: i32,
) -> Vec<f32> {
    let head_size = x.len() / (positions.len() * n_heads);
    let half = n_dims / 2;
    let mut out = x.to_vec();
    for (seq, &pos) in positions.iter().enumerate() {
        for h in 0..n_heads {
            for i in 0..half {
                let idx = seq * (n_heads * head_size) + h * head_size + i;
                let idx2 = idx + half;
                if i >= n_dims / 2 {
                    continue;
                }
                let freq = pos as f32 / theta.powf(2.0 * i as f32 / n_dims as f32);
                let sin = freq.sin();
                let cos = freq.cos();
                let v0 = x[idx];
                let v1 = x[idx2];
                out[idx] = v0 * cos - v1 * sin;
                out[idx2] = v0 * sin + v1 * cos;
            }
        }
    }
    out
}

/// Default RoPE backward.
pub fn default_rope_back(
    x: &[f32],
    positions: &[i32],
    n_dims: usize,
    n_heads: usize,
    theta: f32,
    _mode: i32,
) -> Vec<f32> {
    // RoPE backward is same as forward with -sin
    let head_size = x.len() / (positions.len() * n_heads);
    let half = n_dims / 2;
    let mut out = x.to_vec();
    for (seq, &pos) in positions.iter().enumerate() {
        for h in 0..n_heads {
            for i in 0..half {
                let idx = seq * (n_heads * head_size) + h * head_size + i;
                let idx2 = idx + half;
                if i >= n_dims / 2 {
                    continue;
                }
                let freq = pos as f32 / theta.powf(2.0 * i as f32 / n_dims as f32);
                let sin = freq.sin();
                let cos = freq.cos();
                let v0 = x[idx];
                let v1 = x[idx2];
                out[idx] = v0 * cos + v1 * sin; // note: +sin for gradient
                out[idx2] = -v0 * sin + v1 * cos; // note: -sin
            }
        }
    }
    out
}

/// Default flash attention extension.
#[expect(clippy::too_many_arguments)]
pub fn default_flash_attn_ext(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    n_tokens_q: usize,
    n_tokens_kv: usize,
    head_size: usize,
    scale: f32,
    _max_bias: f32,
    _logit_softcap: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_tokens_q * n_heads * head_size];
    let n_groups = n_heads / n_kv_heads;
    for tok_q in 0..n_tokens_q {
        for h in 0..n_heads {
            let kv_h = h / n_groups;
            // Compute attention scores
            let mut scores = vec![0.0f32; n_tokens_kv];
            #[allow(clippy::needless_range_loop)]
            for tok_kv in 0..n_tokens_kv {
                let mut score = 0.0f32;
                for d in 0..head_size {
                    let q_idx = tok_q * (n_heads * head_size) + h * head_size + d;
                    let k_idx = tok_kv * (n_kv_heads * head_size) + kv_h * head_size + d;
                    score += q[q_idx] * k[k_idx];
                }
                scores[tok_kv] = score * scale;
            }
            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for s in &mut scores {
                *s = (*s - max_score).exp();
            }
            let sum: f32 = scores.iter().sum();
            for s in &mut scores {
                *s /= sum;
            }
            // Weighted sum of values
            for d in 0..head_size {
                let mut val = 0.0f32;
                #[allow(clippy::needless_range_loop)]
                for tok_kv in 0..n_tokens_kv {
                    let v_idx = tok_kv * (n_kv_heads * head_size) + kv_h * head_size + d;
                    val += scores[tok_kv] * v[v_idx];
                }
                output[tok_q * (n_heads * head_size) + h * head_size + d] = val;
            }
        }
    }
    output
}

/// Default SSM convolution (Mamba).
pub fn default_ssm_conv(
    x: &[f32],
    kernel: &[f32],
    state: &[f32],
    n_tokens: usize,
    d_inner: usize,
    d_conv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tokens * d_inner];
    for tok in 0..n_tokens {
        for d in 0..d_inner {
            let mut sum = 0.0f32;
            for k in 0..d_conv {
                let x_val = if tok >= k {
                    x[(tok - k) * d_inner + d]
                } else {
                    state[(d_conv - 1 - tok + k) * d_inner + d]
                };
                sum += x_val * kernel[d * d_conv + k];
            }
            out[tok * d_inner + d] = sum;
        }
    }
    out
}

/// Default SSM scan (Mamba).
pub fn default_ssm_scan(
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    cx: &[f32],
    d_inner: usize,
    d_state: usize,
) -> Vec<f32> {
    let n_tokens = x.len() / d_inner;
    let mut out = vec![0.0f32; n_tokens * d_inner];
    let mut state = cx.to_vec();
    for tok in 0..n_tokens {
        for d in 0..d_inner {
            let dt_val = dt[tok * d_inner + d];
            let a_val = a[d];
            let x_val = x[tok * d_inner + d];
            let decay = (-dt_val * a_val).exp();
            // Update state
            for s in 0..d_state {
                state[s * d_inner + d] = state[s * d_inner + d] * decay
                    + dt_val * b[tok * d_inner * d_state + d * d_state + s] * x_val;
            }
            out[tok * d_inner + d] = state[..d_inner * d_state]
                .chunks(d_inner)
                .map(|ch| ch[d])
                .sum::<f32>()
                + cx[d % cx.len()] * x_val;
        }
    }
    out
}

/// Default RWKV WKV6.
pub fn default_rwkv_wkv6(
    r: &[f32],
    w: &[f32],
    k: &[f32],
    v: &[f32],
    state: &[f32],
    n_tokens: usize,
    n_channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tokens * n_channels];
    let mut s = state.to_vec();
    for tok in 0..n_tokens {
        for c in 0..n_channels {
            let idx = tok * n_channels + c;
            let w_val = (-w[idx].exp()).exp(); // exp(-exp(w))
            let k_val = k[idx];
            let v_val = v[idx];
            let r_val = r[idx];
            // wkv computation
            let num = s[c] + k_val * v_val;
            let den = s[n_channels + c] + k_val;
            s[c] = s[c] * w_val + k_val * v_val;
            s[n_channels + c] = s[n_channels + c] * w_val + k_val;
            out[idx] = r_val * num / (den + 1e-8);
        }
    }
    out
}

/// Default RWKV WKV7.
pub fn default_rwkv_wkv7(
    r: &[f32],
    w: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    state: &[f32],
    n_tokens: usize,
    n_channels: usize,
    head_size: usize,
) -> Vec<f32> {
    let n_heads = n_channels / head_size;
    let mut out = vec![0.0f32; n_tokens * n_channels];
    let mut s = state.to_vec();
    for tok in 0..n_tokens {
        for h in 0..n_heads {
            for hs in 0..head_size {
                let idx = tok * n_channels + h * head_size + hs;
                let w_val = 1.0 / (1.0 + (-w[idx]).exp());
                let k_val = k[idx];
                let v_val = v[idx];
                let r_val = r[idx];
                let g_val = 1.0 / (1.0 + (-g[idx]).exp());
                let si = h * head_size + hs;
                let num = s[si] + k_val * v_val;
                let den = s[n_channels + si] + k_val;
                s[si] = s[si] * w_val + k_val * v_val;
                s[n_channels + si] = s[n_channels + si] * w_val + k_val;
                let r_sig = 1.0 / (1.0 + (-r_val).exp());
                out[idx] = r_sig * g_val * num / (den + 1e-8);
            }
        }
    }
    out
}

/// Default gated delta net.
pub fn default_gated_delta_net(
    x: &[f32],
    gate: &[f32],
    state: &[f32],
    n_tokens: usize,
    n_channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tokens * n_channels];
    let mut s = state.to_vec();
    for tok in 0..n_tokens {
        #[allow(clippy::needless_range_loop)]
        for c in 0..n_channels {
            let idx = tok * n_channels + c;
            let g = 1.0 / (1.0 + (-gate[idx]).exp());
            let x_val = x[idx];
            let delta = s[c] + x_val;
            out[idx] = g * delta;
            s[c] = (1.0 - g) * delta;
        }
    }
    out
}

/// Default gated linear attention.
pub fn default_gated_linear_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &[f32],
    n_tokens: usize,
    n_channels: usize,
    head_size: usize,
) -> Vec<f32> {
    let n_heads = n_channels / head_size;
    let mut out = vec![0.0f32; n_tokens * n_channels];
    let mut s = state.to_vec();
    for tok in 0..n_tokens {
        for h in 0..n_heads {
            for hs in 0..head_size {
                let q_idx = tok * n_channels + h * head_size + hs;
                let k_val = k[q_idx];
                let v_val = v[q_idx];
                let q_val = q[q_idx];
                let si = h * head_size + hs;
                let num = s[si] + k_val * v_val;
                let den = s[n_channels + si] + k_val;
                s[si] += k_val * v_val;
                s[n_channels + si] += k_val;
                out[q_idx] = q_val * num / (den + 1e-8);
            }
        }
    }
    out
}

/// Default timestep embedding.
pub fn default_timestep_embedding(timestep: usize, dim: usize, max_period: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    let half = dim / 2;
    let t = timestep as f32;
    for i in 0..half {
        let freq = t / max_period.powf(2.0 * i as f32 / dim as f32);
        out[i] = freq.sin();
        out[half + i] = freq.cos();
    }
    out
}

/// Default upscale (nearest neighbor).
pub fn default_upscale(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    scale_h: usize,
    scale_w: usize,
) -> Vec<f32> {
    let out_h = h * scale_h;
    let out_w = w * scale_w;
    let mut out = vec![0.0f32; n * c * out_h * out_w];
    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih = oh / scale_h;
                    let iw = ow / scale_w;
                    let x_idx = batch * (c * h * w) + ch * (h * w) + ih * w + iw;
                    let out_idx =
                        batch * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                    out[out_idx] = x[x_idx];
                }
            }
        }
    }
    out
}

/// Default triangular solve (forward/back substitution).
pub fn default_solve_tri(
    a: &[f32],
    b: &[f32],
    n: usize,
    lower: bool,
    _unit_diagonal: bool,
) -> Vec<f32> {
    let mut x = b.to_vec();
    if lower {
        for i in 0..n {
            for j in 0..i {
                x[i] -= a[i * n + j] * x[j];
            }
            x[i] /= a[i * n + i];
        }
    } else {
        for i in (0..n).rev() {
            for j in (i + 1..n).rev() {
                x[i] -= a[i * n + j] * x[j];
            }
            x[i] /= a[i * n + i];
        }
    }
    x
}

/// Default triangular matrix.
pub fn default_tri(n: usize, diagonal: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    if diagonal >= 0 {
        let d = diagonal as usize;
        for i in 0..n {
            for j in d..=i.min(n - 1 - d) {
                if j >= d && j <= i {
                    out[i * n + j - d] = 1.0;
                }
            }
        }
    }
    out
}

/// Default AdamW optimizer step.
pub fn default_opt_step_adamw(
    grad: &[f32],
    params: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    t: f32,
) {
    let bias_corr1 = 1.0 - beta1.powi(t as i32);
    let bias_corr2 = 1.0 - beta2.powi(t as i32);
    for i in 0..params.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        let m_hat = m[i] / bias_corr1;
        let v_hat = v[i] / bias_corr2;
        params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * params[i]);
    }
}

/// Default SGD optimizer step.
pub fn default_opt_step_sgd(
    grad: &[f32],
    params: &mut [f32],
    lr: f32,
    momentum: f32,
    velocity: &mut [f32],
) {
    for i in 0..params.len() {
        velocity[i] = momentum * velocity[i] + lr * grad[i];
        params[i] -= velocity[i];
    }
}

// ── Default miscellaneous ──

/// Default clamp.
pub fn default_clamp(x: &[f32], min: f32, max: f32) -> Vec<f32> {
    x.iter().map(|v| v.clamp(min, max)).collect()
}

/// Default scale.
pub fn default_scale(x: &[f32], scale: f32) -> Vec<f32> {
    x.iter().map(|v| v * scale).collect()
}

/// Default fill.
pub fn default_fill(n: usize, v: f32) -> Vec<f32> {
    vec![v; n]
}

/// Default arange.
pub fn default_arange(n: usize, start: f32, step: f32) -> Vec<f32> {
    (0..n).map(|i| start + i as f32 * step).collect()
}

/// Default top-k.
pub fn default_top_k(x: &[f32], k: usize) -> Vec<f32> {
    let k = k.min(x.len());
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_unstable_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap_or(std::cmp::Ordering::Equal));
    indices.truncate(k);
    indices.iter().map(|&i| x[i]).collect()
}

/// Default argsort.
pub fn default_argsort(x: &[f32], ascending: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    if ascending {
        indices.sort_unstable_by(|&a, &b| {
            x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indices.sort_unstable_by(|&a, &b| {
            x[b].partial_cmp(&x[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    indices
}

/// Default accumulate.
pub fn default_acc(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = a.to_vec();
    for (i, &v) in b.iter().enumerate() {
        if i < out.len() {
            out[i] += v;
        } else {
            out.push(v);
        }
    }
    out
}
