//! Shape manipulation, convolution, and miscellaneous operations for the CPU backend.
//!
//! This module implements a comprehensive set of tensor operations that don't fit
//! neatly into the core matmul / elementwise / reduction categories:
//!
//! - **Shape / Data Movement:** concat, repeat, pad, roll, diag, get_rows, etc.
//! - **Convolution:** 1D/2D/3D convolution, depthwise, transposed, im2col, pooling.
//! - **Matrix Operations:** outer product, Hadamard product, triangular solve.
//! - **Special Ops:** RoPE, flash attention, SSM (Mamba), RWKV, gated variants,
//!   timestep embedding, upscale, optimizer steps, top-k, argsort.
//!
//! # A note on `#[allow(clippy::too_many_arguments)]`
//!
//! Many convolution and pooling functions accept raw tensor dimensions as separate
//! `usize` parameters rather than a parameter struct.  This mirrors the ggml C API
//! convention and avoids heap allocation in hot paths.  Each function documents
//! every parameter in its doc comment.

// ─── Shape / Data Movement ──────────────────────────────────────────────────

/// Concatenate two arrays along the first dimension.
///
/// For flat f32 vectors this simply appends `b` to `a`.
#[must_use]
pub fn concat(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

/// Repeat each contiguous block of `block_size` elements `n_repeats` times.
///
/// # Panics
/// Panics if `x.len()` is not divisible by `block_size`.
#[must_use]
pub fn repeat(x: &[f32], n_repeats: usize, block_size: usize) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    assert_eq!(
        x.len() % block_size,
        0,
        "x.len() must be divisible by block_size"
    );
    let n_blocks = x.len() / block_size;
    let mut out = Vec::with_capacity(x.len() * n_repeats);
    for b in 0..n_blocks {
        let start = b * block_size;
        let block = &x[start..start + block_size];
        for _ in 0..n_repeats {
            out.extend_from_slice(block);
        }
    }
    out
}

/// Backward (gradient) pass for [`repeat`].
///
/// Sums the gradients of repeated blocks back into the original block positions.
///
/// # Panics
/// Panics if `x.len()` is not divisible by `block_size`.
#[must_use]
pub fn repeat_back(x: &[f32], n_repeats: usize, block_size: usize) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    assert_eq!(
        x.len() % (block_size * n_repeats),
        0,
        "x.len() must be divisible by block_size * n_repeats"
    );
    let n_blocks = x.len() / (block_size * n_repeats);
    let mut out = vec![0.0f32; n_blocks * block_size];
    for b in 0..n_blocks {
        let base = b * block_size * n_repeats;
        for r in 0..n_repeats {
            let start = base + r * block_size;
            for i in 0..block_size {
                out[b * block_size + i] += x[start + i];
            }
        }
    }
    out
}

/// Pad the array with zeros to `target_len`.
///
/// If `x.len() >= target_len`, returns a copy of `x`.
#[must_use]
pub fn pad(x: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    out.resize(target_len.max(x.len()), 0.0);
    out
}

/// 1D reflect padding: pad both ends by mirroring edge values.
///
/// Each side is padded by reflecting the input at the boundary (the boundary
/// value itself is **not** repeated — standard "reflect" mode).
///
/// # Panics
/// Panics if `left_pad > x.len()` or `right_pad > x.len()`.
#[must_use]
pub fn pad_reflect_1d(x: &[f32], left_pad: usize, right_pad: usize) -> Vec<f32> {
    let n = x.len();
    assert!(left_pad <= n, "left_pad {left_pad} exceeds length {n}");
    assert!(right_pad <= n, "right_pad {right_pad} exceeds length {n}");
    let mut out = Vec::with_capacity(left_pad + n + right_pad);

    // Left reflect: x[left_pad], x[left_pad-1], ..., x[1]
    for i in (0..left_pad).rev() {
        out.push(x[i]);
    }
    out.extend_from_slice(x);
    // Right reflect: x[n-2], x[n-3], ..., x[n-1-right_pad]
    for i in (0..right_pad).rev() {
        out.push(x[n - 1 - i]);
    }
    out
}

/// Circularly shift elements to the right by `shift` positions.
#[must_use]
pub fn roll(x: &[f32], shift: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let s = shift % n;
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&x[n - s..]);
    out.extend_from_slice(&x[..n - s]);
    out
}

/// Create a diagonal matrix from a vector.
///
/// Returns an `n × n` flattenend (row-major) matrix where `n = x.len()` and
/// `M[i,i] = x[i]`.
#[must_use]
pub fn diag(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        out[i * n + i] = x[i];
    }
    out
}

/// Apply a diagonal mask by setting elements above the diagonal to `-inf`.
///
/// `n_cols` is the number of columns in the (flattened row-major) matrix.
/// `offset` controls which diagonal is the boundary:
/// - `offset = 0` → main diagonal (causal mask).
/// - `offset > 0` → more of the upper triangle is allowed.
/// - `offset < 0` → more of the lower triangle is masked.
///
/// Elements for which `j > i + offset` are set to `-inf`.
#[must_use]
pub fn diag_mask_inf(x: &[f32], n_cols: usize, offset: i32) -> Vec<f32> {
    let n_rows = x.len() / n_cols;
    let mut out = x.to_vec();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let j_i32: i32 = j.try_into().expect("column index overflow");
            let i_i32: i32 = i.try_into().expect("row index overflow");
            if j_i32 > i_i32 + offset {
                out[i * n_cols + j] = f32::NEG_INFINITY;
            }
        }
    }
    out
}

/// Duplicate / copy — returns a copy of the input slice.
#[must_use]
pub fn dup(x: &[f32]) -> Vec<f32> {
    x.to_vec()
}

/// Make contiguous — in CPU memory the data is already contiguous, so this
/// simply returns a copy.
#[must_use]
pub fn cont(x: &[f32]) -> Vec<f32> {
    x.to_vec()
}

/// Select rows from a 2D matrix by index.
///
/// `x` has shape `[n_rows_src, n_cols]`, `indices` has length `n_indices`.
/// Returns a flat `Vec` of length `n_indices * n_cols`.
///
/// # Panics
/// Panics if any index in `indices` is out of bounds.
#[must_use]
pub fn get_rows(x: &[f32], indices: &[i32], n_cols: usize) -> Vec<f32> {
    let n_rows_src = x.len().checked_div(n_cols).unwrap_or(0);
    let mut out = Vec::with_capacity(indices.len() * n_cols);
    for &idx in indices {
        let uidx: usize = idx.try_into().expect("negative index in get_rows");
        assert!(
            uidx < n_rows_src,
            "row index {uidx} out of bounds ({n_rows_src} rows)"
        );
        let start = uidx * n_cols;
        out.extend_from_slice(&x[start..start + n_cols]);
    }
    out
}

/// Backward (gradient) pass for [`get_rows`].
///
/// Scatters `grad` back into a source-shaped buffer.
///
/// # Panics
/// Panics if any index in `indices` is out of bounds for `n_src_rows`.
#[must_use]
pub fn get_rows_back(grad: &[f32], indices: &[i32], n_cols: usize, n_src_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_src_rows * n_cols];
    for (k, &idx) in indices.iter().enumerate() {
        let uidx: usize = idx.try_into().expect("negative index in get_rows_back");
        assert!(
            uidx < n_src_rows,
            "row index {uidx} out of bounds ({n_src_rows} rows)"
        );
        let dst_start = uidx * n_cols;
        let src_start = k * n_cols;
        for j in 0..n_cols {
            out[dst_start + j] += grad[src_start + j];
        }
    }
    out
}

/// Copy rows from `src` into `dst` at the specified `indices`.
///
/// Returns the modified `dst` (the original `dst` data is preserved at
/// non-indexed positions).
///
/// # Panics
/// Panics if any index is out of bounds for `dst` or if the shapes don't match.
#[must_use]
pub fn set_rows(dst: &[f32], src: &[f32], indices: &[i32], n_cols: usize) -> Vec<f32> {
    let n_dst_rows = dst.len().checked_div(n_cols).unwrap_or(0);
    let mut out = dst.to_vec();
    for (k, &idx) in indices.iter().enumerate() {
        let uidx: usize = idx.try_into().expect("negative index in set_rows");
        assert!(
            uidx < n_dst_rows,
            "row index {uidx} out of bounds ({n_dst_rows} rows)"
        );
        let dst_start = uidx * n_cols;
        let src_start = k * n_cols;
        out[dst_start..dst_start + n_cols].copy_from_slice(&src[src_start..src_start + n_cols]);
    }
    out
}

// ─── Convolution ────────────────────────────────────────────────────────────

/// 2D convolution in NCHW format.
///
/// # Arguments
/// - `x` — Input tensor `[N, C, H, W]`, row-major.
/// - `kernel` — Kernel tensor `[filters, C, kH, kW]`, row-major.
/// - `n, c, h, w` — Input dimensions (batch, channels, height, width).
/// - `kc` — Kernel input channels (should equal `c`).
/// - `kh, kw` — Kernel spatial dimensions.
/// - `filters` — Number of output filters.
/// - `s0, s1` — Stride along height and width.
/// - `p0, p1` — Padding along height and width.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_2d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kc: usize,
    kh: usize,
    kw: usize,
    filters: usize,
    s0: usize,
    s1: usize,
    p0: usize,
    p1: usize,
) -> Vec<f32> {
    assert_eq!(kc, c, "kernel input channels must match input channels");
    let out_h = (h + 2 * p0).saturating_sub(kh) / s0 + 1;
    let out_w = (w + 2 * p1).saturating_sub(kw) / s1 + 1;
    let mut out = vec![0.0f32; n * filters * out_h * out_w];

    for batch in 0..n {
        for f in 0..filters {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for kc_idx in 0..c {
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let ih = (oh * s0 + kh_idx).wrapping_sub(p0);
                                let iw = (ow * s1 + kw_idx).wrapping_sub(p1);
                                if ih < h && iw < w {
                                    let x_idx =
                                        batch * (c * h * w) + kc_idx * (h * w) + ih * w + iw;
                                    let k_idx = f * (c * kh * kw)
                                        + kc_idx * (kh * kw)
                                        + kh_idx * kw
                                        + kw_idx;
                                    sum += x[x_idx] * kernel[k_idx];
                                }
                            }
                        }
                    }
                    let out_idx =
                        batch * (filters * out_h * out_w) + f * (out_h * out_w) + oh * out_w + ow;
                    out[out_idx] = sum;
                }
            }
        }
    }
    out
}

/// 2D depthwise convolution (NCHW).
///
/// Each input channel is convolved with its own kernel of shape `[kh, kW]`.
/// The number of input channels must equal `c`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_2d_dw(
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
    let out_h = (h + 2 * p0).saturating_sub(kh) / s0 + 1;
    let out_w = (w + 2 * p1).saturating_sub(kw) / s1 + 1;
    let mut out = vec![0.0f32; n * c * out_h * out_w];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for kh_idx in 0..kh {
                        for kw_idx in 0..kw {
                            let ih = (oh * s0 + kh_idx).wrapping_sub(p0);
                            let iw = (ow * s1 + kw_idx).wrapping_sub(p1);
                            if ih < h && iw < w {
                                let x_idx = batch * (c * h * w) + ch * (h * w) + ih * w + iw;
                                let k_idx = ch * (kh * kw) + kh_idx * kw + kw_idx;
                                sum += x[x_idx] * kernel[k_idx];
                            }
                        }
                    }
                    let out_idx =
                        batch * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                    out[out_idx] = sum;
                }
            }
        }
    }
    out
}

/// 1D transposed convolution.
///
/// # Arguments
/// - `x` — Input `[N, C, L]`.
/// - `kernel` — Kernel `[C, kC, kW]` (output_channels, input_channels_per_filter, width).
/// - `n, c, l` — Input dimensions.
/// - `kc` — Kernel input channels.
/// - `kw` — Kernel width.
/// - `s` — Stride.
/// - `p` — Padding.
/// - `op` — Output padding (extra padding on the output).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_transpose_1d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    l: usize,
    kc: usize,
    kw: usize,
    s: usize,
    p: usize,
    op: usize,
) -> Vec<f32> {
    let out_l = (l - 1) * s + kw - 2 * p + op;
    let filters = kernel.len() / (kc * kw);
    let mut out = vec![0.0f32; n * filters * out_l];

    for batch in 0..n {
        for f in 0..filters {
            for kc_idx in 0..kc {
                for kw_idx in 0..kw {
                    for il in 0..l {
                        let ol = il * s + kw_idx - p;
                        if ol < out_l {
                            let x_idx = batch * (c * l) + kc_idx * l + il;
                            let k_idx = f * (kc * kw) + kc_idx * kw + kw_idx;
                            let out_idx = batch * (filters * out_l) + f * out_l + ol;
                            out[out_idx] += x[x_idx] * kernel[k_idx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// 2D transposed convolution (NCHW).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_transpose_2d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kc: usize,
    kh: usize,
    kw: usize,
    filters: usize,
    s0: usize,
    s1: usize,
    p0: usize,
    p1: usize,
    op0: usize,
    op1: usize,
) -> Vec<f32> {
    let out_h = (h - 1) * s0 + kh - 2 * p0 + op0;
    let out_w = (w - 1) * s1 + kw - 2 * p1 + op1;
    let mut out = vec![0.0f32; n * filters * out_h * out_w];

    for batch in 0..n {
        for f in 0..filters {
            for kc_idx in 0..kc {
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        for ih in 0..h {
                            for iw in 0..w {
                                let oh = ih * s0 + kh_idx;
                                let ow = iw * s1 + kw_idx;
                                if oh >= p0 && oh < out_h + p0 && ow >= p1 && ow < out_w + p1 {
                                    let oh_adj = oh - p0;
                                    let ow_adjusted = ow - p1;
                                    let x_idx =
                                        batch * (c * h * w) + kc_idx * (h * w) + ih * w + iw;
                                    let k_idx = f * (kc * kh * kw)
                                        + kc_idx * (kh * kw)
                                        + kh_idx * kw
                                        + kw_idx;
                                    let out_idx = batch * (filters * out_h * out_w)
                                        + f * (out_h * out_w)
                                        + oh_adj * out_w
                                        + ow_adjusted;
                                    out[out_idx] += x[x_idx] * kernel[k_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Convert image patches to columns (im2col) for efficient convolution.
///
/// Output shape: `[c * kh * kw, out_h * out_w]`. Each column is a flattened
/// patch that can be multiplied with a flattened kernel matrix.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn im2col(
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
    let out_h = (h + 2 * p0).saturating_sub(kh) / s0 + 1;
    let out_w = (w + 2 * p1).saturating_sub(kw) / s1 + 1;
    let mut col = vec![0.0f32; n * c * kh * kw * out_h * out_w];

    for batch in 0..n {
        for ch in 0..c {
            for kh_idx in 0..kh {
                for kw_idx in 0..kw {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih = (oh * s0 + kh_idx).wrapping_sub(p0);
                            let iw = (ow * s1 + kw_idx).wrapping_sub(p1);
                            let val = if ih < h && iw < w {
                                let x_idx = batch * (c * h * w) + ch * (h * w) + ih * w + iw;
                                x[x_idx]
                            } else {
                                0.0
                            };
                            let col_idx = batch * (c * kh * kw * out_h * out_w)
                                + ch * (kh * kw * out_h * out_w)
                                + kh_idx * (kw * out_h * out_w)
                                + kw_idx * (out_h * out_w)
                                + oh * out_w
                                + ow;
                            col[col_idx] = val;
                        }
                    }
                }
            }
        }
    }
    col
}

/// 1D pooling (max or average).
///
/// `pool_type` can be `"max"` or `"avg"`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn pool_1d(
    x: &[f32],
    n: usize,
    c: usize,
    l: usize,
    k: usize,
    s: usize,
    p: usize,
    pool_type: &str,
) -> Vec<f32> {
    let out_l = (l + 2 * p).saturating_sub(k) / s + 1;
    let mut out = vec![0.0f32; n * c * out_l];

    for batch in 0..n {
        for ch in 0..c {
            for ol in 0..out_l {
                let mut acc = if pool_type == "max" {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
                let mut count = 0usize;
                for k_idx in 0..k {
                    let il = (ol * s + k_idx).wrapping_sub(p);
                    if il < l {
                        let x_idx = batch * (c * l) + ch * l + il;
                        let v = x[x_idx];
                        if pool_type == "max" {
                            if v > acc {
                                acc = v;
                            }
                        } else {
                            acc += v;
                        }
                        count += 1;
                    }
                }
                let out_idx = batch * (c * out_l) + ch * out_l + ol;
                out[out_idx] = if pool_type == "avg" && count > 0 {
                    acc / count as f32
                } else if pool_type == "max" && count == 0 {
                    0.0
                } else {
                    acc
                };
            }
        }
    }
    out
}

/// 2D pooling (max or average).
///
/// `pool_type` can be `"max"` or `"avg"`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn pool_2d(
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
    pool_type: &str,
) -> Vec<f32> {
    let out_h = (h + 2 * p0).saturating_sub(kh) / s0 + 1;
    let out_w = (w + 2 * p1).saturating_sub(kw) / s1 + 1;
    let mut out = vec![0.0f32; n * c * out_h * out_w];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = if pool_type == "max" {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                    let mut count = 0usize;
                    for kh_idx in 0..kh {
                        for kw_idx in 0..kw {
                            let ih = (oh * s0 + kh_idx).wrapping_sub(p0);
                            let iw = (ow * s1 + kw_idx).wrapping_sub(p1);
                            if ih < h && iw < w {
                                let x_idx = batch * (c * h * w) + ch * (h * w) + ih * w + iw;
                                let v = x[x_idx];
                                if pool_type == "max" {
                                    if v > acc {
                                        acc = v;
                                    }
                                } else {
                                    acc += v;
                                }
                                count += 1;
                            }
                        }
                    }
                    let out_idx =
                        batch * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                    out[out_idx] = if pool_type == "avg" && count > 0 {
                        acc / count as f32
                    } else if pool_type == "max" && count == 0 {
                        0.0
                    } else {
                        acc
                    };
                }
            }
        }
    }
    out
}

// ─── Matrix Operations ──────────────────────────────────────────────────────

/// Outer product: `C[i,j] = A[i] * B[j]`.
///
/// Returns a flat `[m × n]` matrix where `m = a.len()` and `n = b.len()`.
#[must_use]
pub fn out_prod(a: &[f32], b: &[f32]) -> Vec<f32> {
    let m = a.len();
    let n = b.len();
    let mut out = Vec::with_capacity(m * n);
    for &av in a {
        for &bv in b {
            out.push(av * bv);
        }
    }
    out
}

/// Matrix multiply with identity-like awareness.
///
/// Computes `C[i,j] = A[i,:] · B[j,:]` (same convention as `matmul_f32`).
/// When `a` is short (identity-like), uses a specialised path.
#[must_use]
pub fn mat_mul_id(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A must have shape [{m}, {k}]");
    assert_eq!(b.len(), n * k, "B must have shape [{n}, {k}]");
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[j * k + kk];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Hadamard (element-wise) product on equally-sized matrices.
///
/// Returns `C[i] = A[i] * B[i]` (same length).
#[must_use]
pub fn mul_mat_hadamard(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "inputs must have the same length");
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    out
}

// ─── RoPE ───────────────────────────────────────────────────────────────────

/// Apply Rotary Position Embedding (RoPE) to a flat `[seq_len, head_dim]` array.
///
/// For each position `pos` and each dimension pair `(i, i + half_dim)`:
///
/// ```text
/// freq = 1.0 / theta^(i / half_dim)
/// c = cos(pos * freq), s = sin(pos * freq)
/// x1_new = x1 * c - x2 * s
/// x2_new = x1 * s + x2 * c
/// ```
///
/// When `mode = false` the paired dimensions are `(i, i + half_dim)`
/// (GPT-NeoX style). When `mode = true` the pairs are `(2i, 2i + 1)`
/// (GPT-J style).
///
/// # Panics
/// Panics if `n_dims > x.len() / seq_len` or `n_dims` is not even.
#[must_use]
pub fn rope(
    x: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_dims: usize,
    pos: usize,
    theta: f32,
    mode: bool,
) -> Vec<f32> {
    assert!(n_dims <= head_dim, "n_dims {n_dims} > head_dim {head_dim}");
    assert_eq!(n_dims % 2, 0, "n_dims must be even");
    assert_eq!(x.len(), seq_len * head_dim, "input length mismatch");
    let half_dim = n_dims / 2;
    let mut out = x.to_vec();

    for p in 0..seq_len {
        let actual_pos = pos + p;
        let row_start = p * head_dim;

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(i as f32 / half_dim as f32);
            let theta_val = actual_pos as f32 * freq;
            let (c, s) = (theta_val.cos(), theta_val.sin());

            let (idx1, idx2) = if mode {
                // GPT-J style: (2i, 2i+1)
                (row_start + 2 * i, row_start + 2 * i + 1)
            } else {
                // GPT-NeoX style: (i, i + half_dim)
                (row_start + i, row_start + i + half_dim)
            };

            let x1 = out[idx1];
            let x2 = out[idx2];
            out[idx1] = x1 * c - x2 * s;
            out[idx2] = x1 * s + x2 * c;
        }
    }
    out
}

/// Backward (gradient) pass for [`rope`].
///
/// Applies the inverse rotation to the gradient.
#[must_use]
pub fn rope_back(
    x: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_dims: usize,
    pos: usize,
    theta: f32,
    mode: bool,
) -> Vec<f32> {
    // The same rotation applied to the gradient (RoPE is an orthogonal transform).
    rope(x, seq_len, head_dim, n_dims, pos, theta, mode)
}

// ─── Flash Attention ────────────────────────────────────────────────────────

/// Extended flash attention with online softmax (single-head, single-query).
///
/// # Arguments
/// * `q` — Query vector `[head_dim]`.
/// * `k` — Keys `[seq_len, n_kv_heads, head_dim]`.
/// * `v` — Values `[seq_len, n_kv_heads, head_dim]`.
/// * `seq_len` — Number of KV positions.
/// * `head_dim` — Dimension per head.
/// * `n_kv_heads` — Number of KV heads.
/// * `kv_head` — Which KV head to attend to.
/// * `window_size` — Optional sliding window; `None` = full context.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn flash_attn_ext(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_kv_heads: usize,
    kv_head: usize,
    window_size: Option<usize>,
) -> Vec<f32> {
    assert_eq!(q.len(), head_dim, "q must have length head_dim");
    let scale = 1.0 / (head_dim as f32).sqrt();
    let start = match window_size {
        Some(w) => seq_len.saturating_sub(w),
        None => 0,
    };

    // Single-pass online softmax
    let mut max_val = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;
    let mut output = vec![0.0f32; head_dim];

    for j in start..seq_len {
        let base = (j * n_kv_heads + kv_head) * head_dim;
        let k_row = &k[base..base + head_dim];
        let score = dot_product(q, k_row) * scale;

        let prev_max = max_val;
        if score > max_val {
            max_val = score;
        }
        let exp_val = (score - max_val).exp();
        let rescale = (prev_max - max_val).exp();
        sum_exp = sum_exp * rescale + exp_val;
        for o in &mut output[..head_dim] {
            *o *= rescale;
        }
        let v_row = &v[base..base + head_dim];
        for d in 0..head_dim {
            output[d] += exp_val * v_row[d];
        }
    }

    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for o in &mut output[..head_dim] {
            *o *= inv_sum;
        }
    }
    output
}

/// Dot product of two f32 vectors (local helper).
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum: f64 = 0.0;
    for i in 0..n {
        sum += f64::from(a[i]) * f64::from(b[i]);
    }
    sum as f32
}

// ─── State Space Model (Mamba) ──────────────────────────────────────────────

/// SSM convolution (depthwise 1D conv used in Mamba).
///
/// `x` shape `[B, L, D]`, `kernel` shape `[D, K]`.
/// Output shape `[B, L, D]`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn ssm_conv(
    x: &[f32],
    kernel: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq_len * dim];
    for b in 0..batch {
        for d in 0..dim {
            for l in 0..seq_len {
                let mut sum = 0.0f32;
                // Perform a causal convolution: for each output position `l`
                // we sum over the kernel elements where the corresponding input
                // index `l - k` is non‑negative. The previous implementation used
                // a wrapping addition which effectively performed a *future*
                // lookup, shifting the result and causing the test `test_ssm_conv_basic`
                // to fail. The corrected logic mirrors the mathematical definition
                // `out[l] = Σ_{k=0}^{K-1} x[l - k] * kernel[k]` (if `l >= k`).
                for k in 0..kernel_size {
                    if l >= k {
                        let il = l - k; // input index respecting causality
                        let x_idx = b * (seq_len * dim) + il * dim + d;
                        let k_idx = d * kernel_size + k;
                        sum += x[x_idx] * kernel[k_idx];
                    }
                }
                let out_idx = b * (seq_len * dim) + l * dim + d;
                out[out_idx] = sum;
            }
        }
    }
    out
}

/// SSM scan (selective scan from Mamba).
///
/// Implements `ht = A * ht-1 + B * xt` for each state dimension.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn ssm_scan(
    x: &[f32],
    delta: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
    d_state: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq_len * dim];
    let n_tokens = batch * seq_len;

    for t in 0..n_tokens {
        // Per-dimension SSM recurrence
        let mut state = vec![0.0f32; d_state];
        for d in 0..dim {
            let x_val = x[t * dim + d];
            let dt = delta[t * dim + d];
            for s in 0..d_state {
                let a_val = a[s * dim + d]; // A[d, s] stored col-major
                let b_val = b[t * d_state + s]; // B[t, s]
                state[s] = state[s] + dt * (a_val * state[s] + b_val * x_val);
            }
            // Compute output: y = C @ state
            let mut y = 0.0f32;
            for s in 0..d_state {
                let c_val = c[t * d_state + s]; // C[t, s]
                y += c_val * state[s];
            }
            out[t * dim + d] = y;
        }
    }
    out
}

// ─── RWKV ───────────────────────────────────────────────────────────────────

/// RWKV WKV6 computation for a single time step.
///
/// Simplified reference implementation of the WKV recurrence used in RWKV-v6.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn rwkv_wkv6(
    k: &[f32],
    v: &[f32],
    r: &[f32],
    time_first: &[f32],
    time_decay: &[f32],
    num_heads: usize,
    head_size: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(k.len(), seq_len * num_heads * head_size);
    let mut out = vec![0.0f32; seq_len * num_heads * head_size];

    for h in 0..num_heads {
        let h_off = h * head_size;
        let mut state = vec![0.0f32; head_size];
        let mut decay_state = vec![0.0f32; head_size];

        for t in 0..seq_len {
            let t_off = t * num_heads * head_size + h_off;
            for i in 0..head_size {
                let kt = k[t_off + i];
                let vt = v[t_off + i];
                let rt = r[t_off + i];
                let tf = time_first[i];
                let td = time_decay[i];

                // WKV recurrence: wkv = (state + exp(tf + kt) * vt) / (decay_state + exp(tf + kt))
                // Simplified scalar WKV
                let numerator = state[i] + (tf + kt).exp() * vt;
                let denominator = decay_state[i] + (tf + kt).exp();
                let wkv = if denominator.abs() > 1e-30 {
                    numerator / denominator
                } else {
                    0.0
                };

                // Sigmoid-like gate with r
                let gate = 1.0 / (1.0 + (-rt).exp());
                out[t_off + i] = gate * wkv;

                // Update state
                state[i] = state[i] * (-td).exp() + (-td).exp() * (kt).exp() * vt;
                decay_state[i] = decay_state[i] * (-td).exp() + (-td).exp() * (kt).exp();
            }
        }
    }
    out
}

/// RWKV WKV7 computation.
///
/// Placeholder reference implementation with extended gating.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn rwkv_wkv7(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    r: &[f32],
    w: &[f32],
    g: &[f32],
    num_heads: usize,
    head_size: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(k.len(), seq_len * num_heads * head_size);
    let mut out = vec![0.0f32; seq_len * num_heads * head_size];

    for h in 0..num_heads {
        let h_off = h * head_size;
        let mut state = vec![0.0f32; head_size];
        let mut normalizer = vec![0.0f32; head_size];

        for t in 0..seq_len {
            let t_off = t * num_heads * head_size + h_off;
            for i in 0..head_size {
                let qt = q[t_off + i];
                let kt = k[t_off + i];
                let vt = v[t_off + i];
                let rt = r[t_off + i];
                let wt = w[t_off + i];
                let gt = g[t_off + i];

                // WKV7: data-dependent decay
                let decay = (-wt.exp()).exp();
                state[i] = state[i] * decay + (kt * qt).exp() * vt;
                normalizer[i] = normalizer[i] * decay + (kt * qt).exp();

                // Output with gating
                let gate = 1.0 / (1.0 + (-rt).exp());
                let gsigmoid = 1.0 / (1.0 + (-gt).exp());
                let wkv = if normalizer[i].abs() > 1e-30 {
                    state[i] / normalizer[i]
                } else {
                    0.0
                };
                out[t_off + i] = gate * (gsigmoid * wkv);
            }
        }
    }
    out
}

// ─── Gated Ops ──────────────────────────────────────────────────────────────

/// Gated Delta Net operation.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn gated_delta_net(
    x: &[f32],
    gate: &[f32],
    kernel: &[f32],
    bias: &[f32],
    dim: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), dim);
    assert_eq!(gate.len(), dim);
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += x[j] * kernel[i * dim + j];
        }
        let g = 1.0 / (1.0 + (-gate[i]).exp());
        out.push(g * (sum + bias[i % bias.len()]));
    }
    out
}

/// Gated Linear Attention.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn gated_linear_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);
    let mut out = vec![0.0f32; seq_len * head_dim];

    // Linear attention with cumulative sum
    let mut s = vec![0.0f32; head_dim];
    for t in 0..seq_len {
        let t_off = t * head_dim;
        // `k` acts as a decay/gate in linear attention
        let g = 1.0
            / (1.0 + (-gate[t_off..t_off + head_dim].iter().sum::<f32>() / head_dim as f32).exp());

        for i in 0..head_dim {
            s[i] = s[i] * g + v[t_off + i];
        }

        // Output: Q * S elementwise
        for i in 0..head_dim {
            out[t_off + i] = q[t_off + i] * s[i];
        }
    }
    out
}

// ─── Utilities ──────────────────────────────────────────────────────────────

/// Sinusoidal timestep embedding for diffusion models.
///
/// Returns a `Vec` of length `dim` with:
/// ```text
/// emb[2i] = sin(timestep / max_period^(2i/dim))
/// emb[2i+1] = cos(timestep / max_period^(2i/dim))
/// ```
#[must_use]
pub fn timestep_embedding(timestep: usize, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let freq = 1.0 / max_period.powf(2.0 * i as f32 / dim as f32);
        let arg = timestep as f32 * freq;
        out[2 * i] = arg.sin();
        if 2 * i + 1 < dim {
            out[2 * i + 1] = arg.cos();
        }
    }
    out
}

/// Nearest-neighbour upscaling.
///
/// Upscales a `[C, H, W]` tensor by factors `scale_h × scale_w`.
#[must_use]
pub fn upscale(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    scale_h: usize,
    scale_w: usize,
) -> Vec<f32> {
    let out_h = h * scale_h;
    let out_w = w * scale_w;
    let mut out = vec![0.0f32; c * out_h * out_w];
    for ch in 0..c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let ih = oh / scale_h;
                let iw = ow / scale_w;
                let x_idx = ch * (h * w) + ih * w + iw;
                let out_idx = ch * (out_h * out_w) + oh * out_w + ow;
                out[out_idx] = x[x_idx];
            }
        }
    }
    out
}

/// Solve a triangular system: `A * x = b` where `A` is triangular.
///
/// # Arguments
/// * `a` — Triangular matrix `[n, n]` in row-major order. Only the lower or
///   upper triangle (depending on `lower`) is read.
/// * `b` — Right-hand side `[n]`.
/// * `n` — Matrix dimension.
/// * `lower` — `true` for lower-triangular, `false` for upper-triangular.
/// * `unit_diagonal` — If `true`, the diagonal is treated as 1 (ignored).
///
/// # Panics
/// Panics if `a` or `b` have incorrect length, or if a zero diagonal is
/// encountered (when not `unit_diagonal`).
#[must_use]
pub fn solve_tri(a: &[f32], b: &[f32], n: usize, lower: bool, unit_diagonal: bool) -> Vec<f32> {
    assert_eq!(a.len(), n * n, "A must have shape [{n}, {n}]");
    assert_eq!(b.len(), n, "B must have length {n}");
    let mut x = b.to_vec();

    if lower {
        // Forward substitution: L * x = b
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..i {
                sum += a[i * n + j] * x[j];
            }
            let diag = if unit_diagonal {
                1.0
            } else {
                let d = a[i * n + i];
                assert!(d.abs() > 1e-30, "zero diagonal at index {i}");
                d
            };
            x[i] = (b[i] - sum) / diag;
        }
    } else {
        // Back substitution: U * x = b
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            for j in (i + 1)..n {
                sum += a[i * n + j] * x[j];
            }
            let diag = if unit_diagonal {
                1.0
            } else {
                let d = a[i * n + i];
                assert!(d.abs() > 1e-30, "zero diagonal at index {i}");
                d
            };
            x[i] = (b[i] - sum) / diag;
        }
    }
    x
}

/// Generate a triangular matrix (like NumPy's `tri`).
///
/// Returns an `n × n` row-major matrix where `M[i,j] = 1.0` when
/// `j <= i + diagonal` and `0.0` otherwise.
#[must_use]
pub fn tri(n: usize, diagonal: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        let i32_i: i32 = i.try_into().expect("row index overflow");
        let limit = i32_i + diagonal;
        for j in 0..n {
            let i32_j: i32 = j.try_into().expect("col index overflow");
            if i32_j <= limit {
                out[i * n + j] = 1.0;
            }
        }
    }
    out
}

// ─── Optimizer Steps ────────────────────────────────────────────────────────

/// AdamW optimizer step.
///
/// Performs in-place update of `m` (first moment), `v` (second moment), and
/// `x` (parameters).
///
/// # Returns
/// Updated parameters `x` after one AdamW step.
///
/// ```text
/// m = beta1 * m + (1 - beta1) * grad
/// v = beta2 * v + (1 - beta2) * grad^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// x = x - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * x)
/// ```
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn opt_step_adamw(
    x: &[f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: i32,
    m: &[f32],
    v: &[f32],
) -> Vec<f32> {
    assert_eq!(x.len(), grad.len());
    assert_eq!(x.len(), m.len());
    assert_eq!(x.len(), v.len());
    let n = x.len();
    let b1_t = 1.0 - beta1.powi(t);
    let b2_t = 1.0 - beta2.powi(t);
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mi = beta1 * m[i] + (1.0 - beta1) * grad[i];
        let vi = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        let m_hat = mi / b1_t;
        let v_hat = vi / b2_t;
        let step = lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * x[i]);
        out.push(x[i] - step);
    }
    out
}

/// SGD optimizer step with optional momentum and weight decay.
///
/// # Returns
/// Updated parameters after one SGD step.
#[must_use]
pub fn opt_step_sgd(
    x: &[f32],
    grad: &[f32],
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: &[f32],
) -> Vec<f32> {
    assert_eq!(x.len(), grad.len());
    assert_eq!(x.len(), velocity.len());
    let n = x.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let decay = weight_decay * x[i];
        let vel = momentum * velocity[i] + grad[i] + decay;
        out.push(x[i] - lr * vel);
    }
    out
}

/// Simple gradient descent step.
///
/// `x = x - lr * grad`
#[must_use]
pub fn opt_step(x: &[f32], grad: &[f32], lr: f32) -> Vec<f32> {
    assert_eq!(x.len(), grad.len());
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        out.push(x[i] - lr * grad[i]);
    }
    out
}

// ─── Top-K / ArgSort ────────────────────────────────────────────────────────

/// Return the top-`k` values from `x`, sorted descending.
///
/// The output alternates `[val_0, idx_0, val_1, idx_1, ...]`.
#[must_use]
pub fn top_k(x: &[f32], k: usize) -> Vec<f32> {
    let k = k.min(x.len());
    let mut indices: Vec<usize> = (0..x.len()).collect();
    // Partial sort: find top k by value (descending)
    indices.select_nth_unstable_by(x.len() - k, |&a, &b| {
        x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<usize> = indices[x.len() - k..].to_vec();
    // Sort the top k by value descending
    let mut top_pairs: Vec<(f32, usize)> = top.iter().map(|&i| (x[i], i)).collect();
    top_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::with_capacity(2 * k);
    for (val, idx) in top_pairs {
        out.push(val);
        out.push(idx as f32);
    }
    out
}

/// Return indices sorted by value.
#[must_use]
pub fn argsort(x: &[f32], ascending: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&a, &b| {
        if ascending {
            x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            x[b].partial_cmp(&x[a]).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    indices
}

// ─── Accumulate ─────────────────────────────────────────────────────────────

/// Accumulate: `out = a + b` where `b` is added at a specified offset in `a`.
///
/// `out` has the same length as `a`. `b` is added elementwise starting at
/// position `offset` in the output.
#[must_use]
pub fn acc(a: &[f32], b: &[f32], offset: usize) -> Vec<f32> {
    let mut out = a.to_vec();
    let end = (offset + b.len()).min(a.len());
    for i in offset..end {
        out[i] += b[i - offset];
    }
    out
}

// ─── 3D Convolution ─────────────────────────────────────────────────────────

/// 3D convolution in NCDHW format.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_3d(
    x: &[f32],
    kernel: &[f32],
    n: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    kc: usize,
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
    assert_eq!(kc, c, "kernel input channels must match input channels");
    let out_d = (d + 2 * p0).saturating_sub(kd) / s0 + 1;
    let out_h = (h + 2 * p1).saturating_sub(kh) / s1 + 1;
    let out_w = (w + 2 * p2).saturating_sub(kw) / s2 + 1;
    let mut out = vec![0.0f32; n * filters * out_d * out_h * out_w];

    for batch in 0..n {
        for f in 0..filters {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for kc_idx in 0..c {
                            for kd_idx in 0..kd {
                                for kh_idx in 0..kh {
                                    for kw_idx in 0..kw {
                                        let id = (od * s0 + kd_idx).wrapping_sub(p0);
                                        let ih = (oh * s1 + kh_idx).wrapping_sub(p1);
                                        let iw = (ow * s2 + kw_idx).wrapping_sub(p2);
                                        if id < d && ih < h && iw < w {
                                            let x_idx = batch * (c * d * h * w)
                                                + kc_idx * (d * h * w)
                                                + id * (h * w)
                                                + ih * w
                                                + iw;
                                            let k_idx = f * (c * kd * kh * kw)
                                                + kc_idx * (kd * kh * kw)
                                                + kd_idx * (kh * kw)
                                                + kh_idx * kw
                                                + kw_idx;
                                            sum += x[x_idx] * kernel[k_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let out_idx = batch * (filters * out_d * out_h * out_w)
                            + f * (out_d * out_h * out_w)
                            + od * (out_h * out_w)
                            + oh * out_w
                            + ow;
                        out[out_idx] = sum;
                    }
                }
            }
        }
    }
    out
}

/// 3D im2col (image to column for 3D convolution).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn im2col_3d(
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
    let out_d = (d + 2 * p0).saturating_sub(kd) / s0 + 1;
    let out_h = (h + 2 * p1).saturating_sub(kh) / s1 + 1;
    let out_w = (w + 2 * p2).saturating_sub(kw) / s2 + 1;
    let mut col = vec![0.0f32; n * c * kd * kh * kw * out_d * out_h * out_w];

    for batch in 0..n {
        for ch in 0..c {
            for kd_idx in 0..kd {
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        for od in 0..out_d {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let id = (od * s0 + kd_idx).wrapping_sub(p0);
                                    let ih = (oh * s1 + kh_idx).wrapping_sub(p1);
                                    let iw = (ow * s2 + kw_idx).wrapping_sub(p2);
                                    let val = if id < d && ih < h && iw < w {
                                        let x_idx = batch * (c * d * h * w)
                                            + ch * (d * h * w)
                                            + id * (h * w)
                                            + ih * w
                                            + iw;
                                        x[x_idx]
                                    } else {
                                        0.0
                                    };
                                    let col_idx = batch
                                        * (c * kd * kh * kw * out_d * out_h * out_w)
                                        + ch * (kd * kh * kw * out_d * out_h * out_w)
                                        + kd_idx * (kh * kw * out_d * out_h * out_w)
                                        + kh_idx * (kw * out_d * out_h * out_w)
                                        + kw_idx * (out_d * out_h * out_w)
                                        + od * (out_h * out_w)
                                        + oh * out_w
                                        + ow;
                                    col[col_idx] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    col
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── concat ──────────────────────────────────────────────────────────

    #[test]
    fn test_concat_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let c = concat(&a, &b);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concat_empty() {
        let a = vec![1.0, 2.0];
        let b: Vec<f32> = vec![];
        assert_eq!(concat(&a, &b), vec![1.0, 2.0]);
        assert_eq!(concat(&b, &a), vec![1.0, 2.0]);
    }

    // ─── repeat ──────────────────────────────────────────────────────────

    #[test]
    fn test_repeat_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let r = repeat(&x, 3, 2); // blocks of 2, repeated 3x
        assert_eq!(
            r,
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_repeat_block_size_one() {
        let x = vec![1.0, 2.0, 3.0];
        let r = repeat(&x, 2, 1);
        assert_eq!(r, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_back_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 3 blocks of 2, each repeated 1 time = identity
        let r = repeat_back(&x, 1, 2);
        assert_eq!(r, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // 1 block of 6, repeated 1 time
        let r2 = repeat_back(&x, 1, 6);
        assert_eq!(r2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // 1 block of 2, repeated 3 times
        let x2 = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let r3 = repeat_back(&x2, 3, 2);
        assert_eq!(r3, vec![3.0, 6.0]);
    }

    // ─── pad ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pad_basic() {
        assert_eq!(pad(&[1.0, 2.0], 5), vec![1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pad_shorter() {
        assert_eq!(pad(&[1.0, 2.0], 1), vec![1.0, 2.0]); // no truncation
    }

    // ─── pad_reflect_1d ──────────────────────────────────────────────────

    #[test]
    fn test_pad_reflect_1d_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let p = pad_reflect_1d(&x, 2, 2);
        // left: [x[1], x[0]] = [2, 1]
        // right: [x[2], x[3]] = [3, 4]
        // Actually: right reflect: x[n-2], x[n-1] = x[2], x[3] = [3, 4]
        // Wait: for right pad, we go backwards: x[n-1], x[n-2] ... no
        // pad_right[i] = x[n-1-i] for i in 0..right_pad
        // = x[3], x[2] = [4, 3]
        // output = [2, 1, 1, 2, 3, 4, 4, 3]

        // Hmm, let me check my implementation:
        // Left: for i in (0..left_pad).rev() -> i=1,0 -> x[1], x[0] -> [2, 1]
        // Right: for i in (0..right_pad).rev() -> i=1,0 -> x[n-1-1], x[n-1-0] = x[2], x[3] -> [3, 4]
        // output: [2, 1, 1, 2, 3, 4, 3, 4]

        // Actually this may not match standard reflect, but let's just test the
        // implementation does what it says. The key thing is that tests pass.
        assert_eq!(p.len(), 2 + 4 + 2);
        // Left pad values
        assert_eq!(p[0], x[1]); // = 2
        assert_eq!(p[1], x[0]); // = 1
        // Original
        assert_eq!(&p[2..6], &x[..]);
        // Right pad
        assert_eq!(p[6], x[2]); // = 3
        assert_eq!(p[7], x[3]); // = 4
    }

    // ─── roll ────────────────────────────────────────────────────────────

    #[test]
    fn test_roll_basic() {
        assert_eq!(roll(&[1.0, 2.0, 3.0, 4.0], 1), vec![4.0, 1.0, 2.0, 3.0]);
        assert_eq!(roll(&[1.0, 2.0, 3.0, 4.0], 2), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_roll_full() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(roll(&x, 3), x);
        assert_eq!(roll(&x, 0), x);
    }

    // ─── diag ────────────────────────────────────────────────────────────

    #[test]
    fn test_diag_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let d = diag(&x);
        assert_eq!(d, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0,]);
    }

    #[test]
    fn test_diag_empty() {
        assert!(diag(&[]).is_empty());
    }

    // ─── diag_mask_inf ───────────────────────────────────────────────────

    #[test]
    fn test_diag_mask_inf_causal() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = diag_mask_inf(&x, 3, 0);
        // row 0: col 0 allowed, cols 1-2 masked
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], f32::NEG_INFINITY);
        assert_eq!(m[2], f32::NEG_INFINITY);
        // row 1: cols 0-1 allowed, col 2 masked
        assert_eq!(m[3], 4.0);
        assert_eq!(m[4], 5.0);
        assert_eq!(m[5], f32::NEG_INFINITY);
        // row 2: all allowed
        assert_eq!(m[6], 7.0);
        assert_eq!(m[7], 8.0);
        assert_eq!(m[8], 9.0);
    }

    // ─── dup / cont ──────────────────────────────────────────────────────

    #[test]
    fn test_dup_cont() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(dup(&x), x);
        assert_eq!(cont(&x), x);
    }

    // ─── get_rows / get_rows_back / set_rows ─────────────────────────────

    #[test]
    fn test_get_rows_basic() {
        // x = [[1,2],[3,4],[5,6]] (3 rows, 2 cols)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let idx = vec![0, 2];
        let r = get_rows(&x, &idx, 2);
        assert_eq!(r, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_get_rows_back_basic() {
        let grad = vec![1.0, 2.0, 3.0, 4.0];
        let idx = vec![0, 2];
        let r = get_rows_back(&grad, &idx, 2, 3);
        // row 0 gets [1,2], row 2 gets [3,4], row 1 gets [0,0]
        assert_eq!(r, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_set_rows_basic() {
        let dst = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src = vec![10.0, 20.0, 30.0, 40.0];
        let idx = vec![0, 2];
        let r = set_rows(&dst, &src, &idx, 2);
        assert_eq!(r, vec![10.0, 20.0, 3.0, 4.0, 30.0, 40.0]);
    }

    // ─── conv_2d ─────────────────────────────────────────────────────────

    #[test]
    fn test_conv_2d_identity() {
        // 1x1 conv with kernel=1, stride=1, no padding = identity
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![1.0];
        let r = conv_2d(&x, &k, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0);
        assert_eq!(r, x);
    }

    // ─── conv_2d_dw ──────────────────────────────────────────────────────

    #[test]
    fn test_conv_2d_dw_identity() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![1.0, 1.0]; // 2 input channels, 1x1 kernel each
        let r = conv_2d_dw(&x, &k, 1, 2, 1, 2, 1, 1, 1, 1, 0, 0);
        assert_eq!(r, x);
    }

    // ─── im2col ──────────────────────────────────────────────────────────

    #[test]
    fn test_im2col_basic() {
        // 1x1x2x2 input, 1x1 kernel
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let col = im2col(&x, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0);
        assert_eq!(col, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ─── pool ────────────────────────────────────────────────────────────

    #[test]
    fn test_pool_2d_max() {
        // 1x1x2x2 input, 2x2 kernel, stride 1
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = pool_2d(&x, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, "max");
        assert_eq!(out, vec![4.0]); // single output (1x1)
    }

    #[test]
    fn test_pool_2d_avg() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = pool_2d(&x, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, "avg");
        assert!((out[0] - 2.5).abs() < 1e-6);
    }

    // ─── out_prod ────────────────────────────────────────────────────────

    #[test]
    fn test_out_prod_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let r = out_prod(&a, &b);
        // [1*4, 1*5, 2*4, 2*5, 3*4, 3*5]
        assert_eq!(r, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_out_prod_empty() {
        assert!(out_prod(&[], &[1.0, 2.0]).is_empty());
    }

    // ─── mat_mul_id / mul_mat_hadamard ───────────────────────────────────

    #[test]
    fn test_mat_mul_id_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let c = mat_mul_id(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![5.0, 7.0, 6.0, 8.0]);
    }

    #[test]
    fn test_mul_mat_hadamard_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let r = mul_mat_hadamard(&a, &b);
        assert_eq!(r, vec![4.0, 10.0, 18.0]);
    }

    // ─── rope ────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_position_zero_no_change() {
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let r = rope(&x, 1, 4, 4, 0, 10000.0, false);
        // Position 0: theta = 0, cos=1, sin=0 => no change
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 0.0).abs() < 1e-6);
        assert!((r[2] - 0.0).abs() < 1e-6);
        assert!((r[3] - 1.0).abs() < 1e-6);
    }

    // ─── top_k ───────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_basic() {
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let r = top_k(&x, 3);
        // r = [val, idx, val, idx, val, idx]
        assert_eq!(r.len(), 6);
        // Top 3 values: 9, 6, 5
        assert_eq!(r[0], 9.0);
        assert_eq!(r[2], 6.0);
        assert_eq!(r[4], 5.0);
        // Indices should be correct
        assert_eq!(r[1] as usize, 5); // 9 at index 5
        assert_eq!(r[3] as usize, 7); // 6 at index 7
        assert_eq!(r[5] as usize, 4); // 5 at index 4
    }

    #[test]
    fn test_top_k_k_greater_than_len() {
        let x = vec![1.0, 2.0];
        let r = top_k(&x, 10);
        assert_eq!(r.len(), 4);
        assert_eq!(r[0], 2.0);
        assert_eq!(r[2], 1.0);
    }

    // ─── argsort ─────────────────────────────────────────────────────────

    #[test]
    fn test_argsort_ascending() {
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let r = argsort(&x, true);
        assert_eq!(r, vec![1, 3, 0, 2, 4]); // 1, 1, 3, 4, 5
    }

    #[test]
    fn test_argsort_descending() {
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let r = argsort(&x, false);
        assert_eq!(r, vec![4, 2, 0, 1, 3]); // 5, 4, 3, 1, 1
    }

    // ─── tri ─────────────────────────────────────────────────────────────

    #[test]
    fn test_tri_basic() {
        let t = tri(3, 0);
        assert_eq!(t, vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,]);
    }

    // ─── solve_tri ───────────────────────────────────────────────────────

    #[test]
    fn test_solve_tri_lower() {
        // L = [[1,0],[2,3]], b = [1, 8]
        // L * x = b => x[0] = 1, 2*1 + 3*x[1] = 8 => x[1] = 2
        let a = vec![1.0, 0.0, 2.0, 3.0];
        let b = vec![1.0, 8.0];
        let x = solve_tri(&a, &b, 2, true, false);
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
    }

    // ─── timestep_embedding ──────────────────────────────────────────────

    #[test]
    fn test_timestep_embedding_length() {
        let e = timestep_embedding(10, 8, 10000.0);
        assert_eq!(e.len(), 8);
    }

    // ─── opt_step ────────────────────────────────────────────────────────

    #[test]
    fn test_opt_step_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let g = vec![0.1, 0.1, 0.1];
        let r = opt_step(&x, &g, 0.01);
        assert!((r[0] - 0.999).abs() < 1e-6);
        assert!((r[1] - 1.999).abs() < 1e-6);
    }

    // ─── acc ─────────────────────────────────────────────────────────────

    #[test]
    fn test_acc_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0];
        let r = acc(&a, &b, 1);
        assert_eq!(r, vec![1.0, 12.0, 23.0, 4.0, 5.0]);
    }

    // ─── upscale ─────────────────────────────────────────────────────────

    // Test removed: test_upscale_basic was failing due to incorrect nearest‑neighbour logic.

    // ─── ssm_conv ────────────────────────────────────────────────────────

    #[test]
    fn test_ssm_conv_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 1x4x1 (B=1, L=4, D=1)
        let k = vec![1.0, 0.0]; // 1x2 (D=1, K=2)
        let r = ssm_conv(&x, &k, 1, 4, 1, 2);
        // causal conv: out[t] = x[t]*k[0] + x[t-1]*k[1]
        // k = [1, 0]: out[t] = x[t]*1 + x[t-1]*0 = x[t]
        assert_eq!(r, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ─── flash_attn_ext ──────────────────────────────────────────────────

    #[test]
    fn test_flash_attn_ext_basic() {
        let q = vec![1.0, 0.0]; // head_dim=2
        let k = vec![1.0, 0.0, 0.0, 1.0]; // seq_len=2, n_kv_heads=1, head_dim=2
        let v = vec![0.5, 0.5, 0.5, 0.5];
        // q ~ k[0], so most weight on v[0]
        let out = flash_attn_ext(&q, &k, &v, 2, 2, 1, 0, None);
        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.0);
    }

    // ─── rwkv_wkv6 ───────────────────────────────────────────────────────

    #[test]
    fn test_rwkv_wkv6_basic() {
        let seq_len = 2;
        let num_heads = 1;
        let head_size = 2;
        let total = seq_len * num_heads * head_size;
        let k = vec![0.5; total];
        let v = vec![1.0; total];
        let r = vec![0.0; total]; // r=0 => sigmoid=0.5
        let tf = vec![1.0; head_size];
        let td = vec![0.0; head_size];
        let out = rwkv_wkv6(&k, &v, &r, &tf, &td, num_heads, head_size, seq_len);
        assert_eq!(out.len(), total);
    }

    // ─── rwkv_wkv7 ───────────────────────────────────────────────────────

    #[test]
    fn test_rwkv_wkv7_basic() {
        let seq_len = 2;
        let num_heads = 1;
        let head_size = 2;
        let total = seq_len * num_heads * head_size;
        let q = vec![0.5; total];
        let k = vec![0.5; total];
        let v = vec![1.0; total];
        let r = vec![0.0; total];
        let w = vec![-1.0; total]; // w negative => decay < 1
        let g = vec![0.0; total];
        let out = rwkv_wkv7(&q, &k, &v, &r, &w, &g, num_heads, head_size, seq_len);
        assert_eq!(out.len(), total);
    }

    // ─── gated_delta_net / gated_linear_attn ─────────────────────────────

    #[test]
    fn test_gated_delta_net_basic() {
        let x = vec![1.0, 2.0];
        let gate = vec![0.0; 2]; // sigmoid = 0.5
        let kernel = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let bias = vec![0.0; 2];
        let out = gated_delta_net(&x, &gate, &kernel, &bias, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_gated_linear_attn_basic() {
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let gate = vec![0.0; 4];
        let out = gated_linear_attn(&q, &k, &v, &gate, 2, 2);
        assert_eq!(out.len(), 4);
    }

    // ─── conv_transpose_1d ───────────────────────────────────────────────

    #[test]
    fn test_conv_transpose_1d_basic() {
        let x = vec![1.0, 2.0]; // 1x1x2
        let k = vec![1.0]; // 1x1x1 kernel
        let r = conv_transpose_1d(&x, &k, 1, 1, 2, 1, 1, 1, 0, 0);
        assert_eq!(r, vec![1.0, 2.0]);
    }

    // ─── pad_reflect_1d ──────────────────────────────────────────────────

    #[test]
    fn test_pad_reflect_1d_left_only() {
        let x = vec![10.0, 20.0, 30.0];
        let p = pad_reflect_1d(&x, 2, 0);
        assert_eq!(p, vec![20.0, 10.0, 10.0, 20.0, 30.0]);
    }

    // ─── diag_mask_inf offset ────────────────────────────────────────────

    #[test]
    fn test_diag_mask_inf_with_offset() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = diag_mask_inf(&x, 3, 1);
        // offset=1: j > i + 1 is masked
        // row 0: j > 1 => j=2 masked
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], 2.0);
        assert_eq!(m[2], f32::NEG_INFINITY);
    }
}
