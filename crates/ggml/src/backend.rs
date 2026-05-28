//! Hardware backend trait for tensor operations.
//!
//! Defines the [`Backend`] trait that all hardware backends (CPU, CUDA, etc.)
//! must implement.
//!
//! This is the plugin interface: adding a new hardware backend means creating a
//! new crate that implements [`Backend`] and registering it with the registry.
//!
//! # Quantized types
//!
//! The [`QuantType`](llama_core::QuantType) enum and [`Backend::mat_vec_quant`] method support
//! quantized weight formats (Q4_0, Q8_0, Q4_1).  Backends can override
//! `mat_vec_quant` with format-specific kernels for 2–4× throughput
//! improvement over dequantize-then-compute.
//!
//! [`QuantType`] and [`BackendInfo`] are re-exported from `llama_core` for
//! convenience (see the re-exports at the bottom of this module).

use crate::tensor::Tensor;

pub use crate::defaults::*;
pub use llama_core::backend::{BackendInfo, QuantType};

/// Enumeration of all GGML operation types.
///
/// Each variant corresponds to a tensor operation that a backend can
/// accelerate.  Use [`execute_op`] to dispatch on this enum, or call
/// the corresponding method on [`Backend`] directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpType {
    // ── Matrix operations ──
    /// Matrix-vector product: `y = W @ x`
    MatVec,
    /// Quantized matrix-vector product
    MatVecQuant,
    /// Matrix-matrix product: `C = A @ B`
    MatMul,
    /// Outer product: `C[i,j] = A[i] * B[j]`
    OutProd,
    /// Matrix-matrix identity-aware product
    MatMulId,
    /// Element-wise Hadamard product on matrices
    MulMatHadamard,

    // ── Unary element-wise activations ──
    /// Absolute value
    Abs,
    /// Sign function (-1, 0, 1)
    Sgn,
    /// Negation
    Neg,
    /// Step function (1 if x > 0 else 0)
    Step,
    /// Hyperbolic tangent
    Tanh,
    /// Exponential Linear Unit
    Elu,
    /// Rectified Linear Unit
    Relu,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hard sigmoid: clamp(x/6 + 0.5, 0, 1)
    HardSigmoid,
    /// Hard swish: x * hard_sigmoid(x)
    HardSwish,
    /// Exponential
    Exp,
    /// exp(x) - 1 (numerically stable)
    Expm1,
    /// Softplus: log(1 + exp(x))
    Softplus,
    /// Floor
    Floor,
    /// Ceil
    Ceil,
    /// Round to nearest integer
    Round,
    /// Truncate toward zero
    Trunc,
    /// Sine
    Sin,
    /// Cosine
    Cos,
    /// Square: x²
    Sqr,
    /// Square root
    Sqrt,
    /// SiLU (Swish) backward gradient
    SiluBack,
    /// Leaky ReLU
    LeakyRelu,
    /// GELU Erf approximation
    GeluErf,
    /// GELU Quick approximation
    GeluQuick,

    // ── Activation aliases ──
    /// SiLU (Sigmoid Linear Unit) alias
    Silu,
    /// GELU (Gaussian Error Linear Unit) alias
    Gelu,

    // ── Binary element-wise ──
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Addition with scalar broadcast: c[i] = a[i] + b
    Add1,

    // ── Gated activations ──
    /// SwiGLU: sigmoid(a·β) * a * b
    Swiglu,
    /// SwiGLU with β=1
    SwigluOai,
    /// GEGLU: GELU(a) * b
    Geglu,
    /// REGLU: ReLU(a) * b
    Reglu,
    /// GEGLU with Erf approximation
    GegluErf,
    /// GEGLU with Quick approximation
    GegluQuick,
    /// Xi approximation
    Xielu,

    // ── Normalization ──
    /// Layer normalization
    Norm,
    /// Group normalization
    GroupNorm,
    /// L2 normalization
    L2Norm,
    /// RMS normalization
    RmsNorm,
    /// RMS normalization backward
    RmsNormBack,

    // ── Reduction ──
    /// Sum of all elements
    Sum,
    /// Mean of all elements
    Mean,
    /// Index of maximum value
    Argmax,
    /// Count equal elements
    CountEqual,
    /// Cumulative sum
    Cumsum,
    /// Sum along rows
    SumRows,
    /// Softmax
    SoftMax,
    /// Softmax backward
    SoftMaxBack,
    /// Cross-entropy loss
    CrossEntropyLoss,
    /// Cross-entropy loss backward
    CrossEntropyLossBack,

    // ── Shape manipulation ──
    /// Concatenate two tensors
    Concat,
    /// Repeat elements
    Repeat,
    /// Repeat backward gradient
    RepeatBack,
    /// Zero-pad to target length
    Pad,
    /// 1D reflect padding
    PadReflect1d,
    /// Circular shift
    Roll,
    /// Create diagonal matrix from vector
    Diag,
    /// Apply diagonal -inf mask
    DiagMaskInf,
    /// Copy / duplicate
    Dup,
    /// Make contiguous
    Cont,
    /// Gather rows by index
    GetRows,
    /// GetRows backward gradient
    GetRowsBack,
    /// Scatter rows by index
    SetRows,

    // ── Convolution ──
    /// 2D convolution
    Conv2d,
    /// 2D depth-wise convolution
    Conv2dDw,
    /// 1D transposed convolution
    ConvTranspose1d,
    /// 2D transposed convolution
    ConvTranspose2d,
    /// Image to column
    Im2col,
    /// 1D pooling
    Pool1d,
    /// 2D pooling
    Pool2d,
    /// 3D convolution
    Conv3d,
    /// 3D image to column
    Im2col3d,

    // ── Special operations ──
    /// Rotary Position Embedding
    Rope,
    /// RoPE backward
    RopeBack,
    /// Flash attention
    FlashAttnExt,
    /// State Space Model convolution
    SsmConv,
    /// State Space Model scan
    SsmScan,
    /// RWKV WKV6
    RwkvWkv6,
    /// RWKV WKV7
    RwkvWkv7,
    /// Gated Delta Net
    GatedDeltaNet,
    /// Gated Linear Attention
    GatedLinearAttn,
    /// Timestep embedding
    TimestepEmbedding,
    /// Nearest-neighbor upscale
    Upscale,
    /// Triangular solve
    SolveTri,
    /// Triangular matrix
    Tri,

    // ── Optimizer steps ──
    /// AdamW optimizer step
    OptStepAdamW,
    /// SGD optimizer step
    OptStepSGD,

    // ── Miscellaneous ──
    /// Clamp values to [min, max]
    Clamp,
    /// Scale by factor
    Scale,
    /// Fill with constant
    Fill,
    /// Generate arithmetic sequence
    Arange,
    /// Top-K selection
    TopK,
    /// Argsort
    Argsort,
    /// Accumulate
    Acc,
}

/// A hardware backend capable of executing tensor operations.
///
/// This is **the** plugin interface for supporting different hardware.
/// Each backend implements the core math operations needed by the
/// inference engine. The trait is object-safe so backends can be
/// used polymorphically via `Arc<dyn Backend>`.
///
/// # Extending
///
/// To add a new hardware backend:
///
/// 1. Create a struct for your backend (e.g. `VulkanBackend`).
/// 2. Implement `Backend` for it.
/// 3. Register it with `BackendRegistry` so it participates in auto-selection.
///
/// # Notes on object-safety
///
/// The trait avoids generic parameters and uses only `&[f32]` / `Vec<f32>`
/// signatures so it remains object-safe. Parallelism is handled internally
/// by each backend.
pub trait Backend: Send + Sync {
    // ──────────────────────────────────────────────────────────────────────────
    //  Info
    // ──────────────────────────────────────────────────────────────────────────

    /// Returns information about this backend.
    fn info(&self) -> BackendInfo;

    // ──────────────────────────────────────────────────────────────────────────
    //  Matrix operations
    // ──────────────────────────────────────────────────────────────────────────

    /// Matrix-vector product: `y = weight @ input`
    ///
    /// `weight` has shape `(rows, cols)` in row-major order.
    /// `input` has length `cols`.
    /// Returns a vector of length `rows`.
    fn mat_vec(&self, weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32>;

    /// Quantized matrix-vector product: `y = quantized_weight @ input`
    ///
    /// `weight` is the quantized weight data in `quant_type` format with shape
    /// `(rows, cols)` in row-major order.  `input` has length `cols`.
    /// Returns a vector of length `rows`.
    ///
    /// The default implementation dequantizes each row block-by-block and
    /// falls back to dot-product logic.  Backends with format-specific
    /// kernels (SIMD, GPU) should override this.
    fn mat_vec_quant(
        &self,
        weight: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        default_mat_vec_quant(weight, quant_type, rows, cols, input)
    }

    /// Matrix-matrix product: `C = A @ B`
    ///
    /// The default implementation computes a simple scalar matmul.
    /// Backends with optimized kernels (tiled, GPU) should override this.
    fn mat_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        default_mat_mul(a, b)
    }

    /// Outer product: `C[i,j] = A[i] * B[j]`
    fn out_prod(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_out_prod(a, b)
    }

    /// Identity-aware matrix multiply (sparse/diagonal support).
    fn mat_mul_id(
        &self,
        a: &[f32],
        b: &[f32],
        a_rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Vec<f32> {
        default_mat_mul_id(a, b, a_rows, a_cols, b_cols)
    }

    /// Hadamard (element-wise) product on matrix data.
    fn mul_mat_hadamard(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_mul_mat_hadamard(a, b)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Unary element-wise
    // ──────────────────────────────────────────────────────────────────────────

    /// Absolute value: `y[i] = |x[i]|`
    fn abs(&self, x: &[f32]) -> Vec<f32> {
        default_abs(x)
    }

    /// Sign function: `y[i] = sign(x[i])` (-1, 0, 1)
    fn sgn(&self, x: &[f32]) -> Vec<f32> {
        default_sgn(x)
    }

    /// Negation: `y[i] = -x[i]`
    fn neg(&self, x: &[f32]) -> Vec<f32> {
        default_neg(x)
    }

    /// Step function: `y[i] = 1.0 if x[i] > 0 else 0.0`
    fn step(&self, x: &[f32]) -> Vec<f32> {
        default_step(x)
    }

    /// Hyperbolic tangent: `y[i] = tanh(x[i])`
    fn tanh(&self, x: &[f32]) -> Vec<f32> {
        default_tanh(x)
    }

    /// Exponential Linear Unit: `y[i] = x[i] if x[i] > 0 else exp(x[i]) - 1`
    fn elu(&self, x: &[f32]) -> Vec<f32> {
        default_elu(x)
    }

    /// Rectified Linear Unit: `y[i] = max(0, x[i])`
    fn relu(&self, x: &[f32]) -> Vec<f32> {
        default_relu(x)
    }

    /// Sigmoid: `y[i] = 1 / (1 + exp(-x[i]))`
    fn sigmoid(&self, x: &[f32]) -> Vec<f32> {
        default_sigmoid(x)
    }

    /// Hard sigmoid: `y[i] = clamp(x[i] / 6 + 0.5, 0, 1)`
    fn hard_sigmoid(&self, x: &[f32]) -> Vec<f32> {
        default_hard_sigmoid(x)
    }

    /// Hard swish: `y[i] = x[i] * hard_sigmoid(x[i])`
    fn hard_swish(&self, x: &[f32]) -> Vec<f32> {
        default_hard_swish(x)
    }

    /// Exponential: `y[i] = exp(x[i])`
    fn exp(&self, x: &[f32]) -> Vec<f32> {
        default_exp(x)
    }

    /// `exp(x) - 1` (numerically stable for small x)
    fn expm1(&self, x: &[f32]) -> Vec<f32> {
        default_expm1(x)
    }

    /// Softplus: `y[i] = log(1 + exp(x[i]))`
    fn softplus(&self, x: &[f32]) -> Vec<f32> {
        default_softplus(x)
    }

    /// Floor: `y[i] = floor(x[i])`
    fn floor(&self, x: &[f32]) -> Vec<f32> {
        default_floor(x)
    }

    /// Ceil: `y[i] = ceil(x[i])`
    fn ceil(&self, x: &[f32]) -> Vec<f32> {
        default_ceil(x)
    }

    /// Round to nearest integer (ties away from zero)
    fn round(&self, x: &[f32]) -> Vec<f32> {
        default_round(x)
    }

    /// Truncate toward zero
    fn trunc(&self, x: &[f32]) -> Vec<f32> {
        default_trunc(x)
    }

    /// Sine: `y[i] = sin(x[i])`
    fn sin(&self, x: &[f32]) -> Vec<f32> {
        default_sin(x)
    }

    /// Cosine: `y[i] = cos(x[i])`
    fn cos(&self, x: &[f32]) -> Vec<f32> {
        default_cos(x)
    }

    /// Square: `y[i] = x[i] * x[i]`
    fn sqr(&self, x: &[f32]) -> Vec<f32> {
        default_sqr(x)
    }

    /// Square root: `y[i] = sqrt(x[i])`
    fn sqrt(&self, x: &[f32]) -> Vec<f32> {
        default_sqrt(x)
    }

    /// SiLU backward gradient
    fn silu_back(&self, x: &[f32]) -> Vec<f32> {
        default_silu_back(x)
    }

    /// Leaky ReLU: `y[i] = x[i] if x[i] > 0 else negative_slope * x[i]`
    fn leaky_relu(&self, x: &[f32], negative_slope: f32) -> Vec<f32> {
        default_leaky_relu(x, negative_slope)
    }

    /// GELU with Erf approximation:
    /// `y[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))`
    fn gelu_erf(&self, x: &[f32]) -> Vec<f32> {
        default_gelu_erf(x)
    }

    /// GELU with Quick approximation:
    /// `y[i] = x[i] * sigmoid(1.702 * x[i])`
    fn gelu_quick(&self, x: &[f32]) -> Vec<f32> {
        default_gelu_quick(x)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Activation aliases (matching C++ ggml naming)
    // ──────────────────────────────────────────────────────────────────────────

    /// Sigmoid Linear Unit (SiLU) activation: `y = x * sigmoid(x)`
    fn silu(&self, x: &[f32]) -> Vec<f32> {
        default_silu(x)
    }

    /// Gaussian Error Linear Unit (GELU) activation:
    /// `y = x * Φ(x)` where Φ is the standard Gaussian CDF.
    fn gelu(&self, x: &[f32]) -> Vec<f32> {
        default_gelu(x)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Binary element-wise
    // ──────────────────────────────────────────────────────────────────────────

    /// Element-wise addition: `c[i] = a[i] + b[i]`
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_add(a, b)
    }

    /// Element-wise subtraction: `c[i] = a[i] - b[i]`
    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_sub(a, b)
    }

    /// Element-wise multiplication: `c[i] = a[i] * b[i]`
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_mul(a, b)
    }

    /// Element-wise division: `c[i] = a[i] / b[i]`
    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_div(a, b)
    }

    /// ADD1: `c[i] = a[i] + b` (b is a scalar broadcast)
    fn add1(&self, a: &[f32], b: f32) -> Vec<f32> {
        default_add1(a, b)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Gated activations
    // ──────────────────────────────────────────────────────────────────────────

    /// SwiGLU: `y = sigmoid(a * beta) * a * b`
    fn swiglu(&self, a: &[f32], b: &[f32], beta: f32) -> Vec<f32> {
        default_swiglu(a, b, beta)
    }

    /// SwiGLU with beta=1
    fn swiglu_oai(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_swiglu_oai(a, b)
    }

    /// GEGLU: `y = GELU(a) * b`
    fn geglu(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_geglu(a, b)
    }

    /// REGLU: `y = ReLU(a) * b`
    fn reglu(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_reglu(a, b)
    }

    /// GEGLU with Erf approximation
    fn geglu_erf(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_geglu_erf(a, b)
    }

    /// GEGLU with Quick approximation
    fn geglu_quick(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_geglu_quick(a, b)
    }

    /// Xi approximation
    fn xielu(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_xielu(a, b)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Normalization
    // ──────────────────────────────────────────────────────────────────────────

    /// Root Mean Square Normalization: `y = (x / RMS(x)) * weight`
    /// where `RMS(x) = sqrt(mean(x^2) + eps)`.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        default_rms_norm(x, weight, eps)
    }

    /// RMS normalization backward
    fn rms_norm_back(&self, x: &[f32], weight: &[f32], grad: &[f32], eps: f32) -> Vec<f32> {
        default_rms_norm_back(x, weight, grad, eps)
    }

    /// Layer normalization: `y = (x - mean) / sqrt(var + eps) * weight + bias`
    fn norm(&self, x: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
        default_norm(x, weight, bias, eps)
    }

    /// Group normalization
    fn group_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        eps: f32,
        n_groups: usize,
    ) -> Vec<f32> {
        default_group_norm(x, weight, bias, eps, n_groups)
    }

    /// L2 normalization: `y[i] = x[i] / sqrt(sum(x[j]^2) + eps)`
    fn l2_norm(&self, x: &[f32], eps: f32) -> Vec<f32> {
        default_l2_norm(x, eps)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Reduction
    // ──────────────────────────────────────────────────────────────────────────

    /// Sum of all elements
    fn sum(&self, x: &[f32]) -> f32 {
        default_sum(x)
    }

    /// Mean of all elements
    fn mean(&self, x: &[f32]) -> f32 {
        default_mean(x)
    }

    /// Index of the maximum value
    fn argmax(&self, x: &[f32]) -> usize {
        default_argmax(x)
    }

    /// Count equal elements between two arrays
    fn count_equal(&self, a: &[f32], b: &[f32]) -> usize {
        default_count_equal(a, b)
    }

    /// Cumulative sum (flattened)
    fn cumsum(&self, x: &[f32]) -> Vec<f32> {
        default_cumsum(x)
    }

    /// Sum rows: for 2D input `[rows, cols]`, sum each row
    fn sum_rows(&self, x: &[f32], cols: usize) -> Vec<f32> {
        default_sum_rows(x, cols)
    }

    /// Softmax: `exp(x - max) / sum(exp(x - max))`
    fn soft_max(&self, x: &[f32]) -> Vec<f32> {
        default_soft_max(x)
    }

    /// Softmax backward gradient
    fn soft_max_back(&self, output: &[f32], grad: &[f32]) -> Vec<f32> {
        default_soft_max_back(output, grad)
    }

    /// Cross-entropy loss
    fn cross_entropy_loss(&self, prediction: &[f32], target: &[f32]) -> f32 {
        default_cross_entropy_loss(prediction, target)
    }

    /// Cross-entropy loss backward gradient
    fn cross_entropy_loss_back(&self, prediction: &[f32], target: &[f32]) -> Vec<f32> {
        default_cross_entropy_loss_back(prediction, target)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Shape manipulation
    // ──────────────────────────────────────────────────────────────────────────

    /// Concatenate two arrays
    fn concat(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_concat(a, b)
    }

    /// Repeat elements in blocks
    fn repeat(&self, x: &[f32], n_repeats: usize, block_size: usize) -> Vec<f32> {
        default_repeat(x, n_repeats, block_size)
    }

    /// Repeat backward gradient
    fn repeat_back(&self, x: &[f32], n_repeats: usize, block_size: usize) -> Vec<f32> {
        default_repeat_back(x, n_repeats, block_size)
    }

    /// Zero-pad to target length
    fn pad(&self, x: &[f32], target_len: usize) -> Vec<f32> {
        default_pad(x, target_len)
    }

    /// 1D reflect padding
    fn pad_reflect_1d(&self, x: &[f32], left_pad: usize, right_pad: usize) -> Vec<f32> {
        default_pad_reflect_1d(x, left_pad, right_pad)
    }

    /// Circular shift right
    fn roll(&self, x: &[f32], shift: usize) -> Vec<f32> {
        default_roll(x, shift)
    }

    /// Create diagonal matrix from vector
    fn diag(&self, x: &[f32]) -> Vec<f32> {
        default_diag(x)
    }

    /// Apply diagonal -inf mask
    fn diag_mask_inf(&self, x: &[f32], n_cols: usize, offset: i32) -> Vec<f32> {
        default_diag_mask_inf(x, n_cols, offset)
    }

    /// Copy / duplicate
    fn dup(&self, x: &[f32]) -> Vec<f32> {
        default_dup(x)
    }

    /// Make contiguous (copy)
    fn cont(&self, x: &[f32]) -> Vec<f32> {
        default_cont(x)
    }

    /// Gather rows by index
    fn get_rows(&self, x: &[f32], indices: &[i32], n_cols: usize) -> Vec<f32> {
        default_get_rows(x, indices, n_cols)
    }

    /// GetRows backward gradient
    fn get_rows_back(
        &self,
        grad: &[f32],
        indices: &[i32],
        n_cols: usize,
        n_src_rows: usize,
    ) -> Vec<f32> {
        default_get_rows_back(grad, indices, n_cols, n_src_rows)
    }

    /// Scatter rows by index
    fn set_rows(&self, dst: &[f32], src: &[f32], indices: &[i32], n_cols: usize) -> Vec<f32> {
        default_set_rows(dst, src, indices, n_cols)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Convolution
    // ──────────────────────────────────────────────────────────────────────────

    /// 2D convolution (NCHW format)
    #[expect(clippy::too_many_arguments)]
    fn conv_2d(
        &self,
        x: &[f32],
        kernel: &[f32],
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        filters: usize,
        s0: usize,
        s1: usize,
        p0: usize,
        p1: usize,
    ) -> Vec<f32> {
        default_conv_2d(x, kernel, n, c, h, w, 0, kh, kw, filters, s0, s1, p0, p1)
    }

    /// 2D depth-wise convolution
    fn conv_2d_dw(
        &self,
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
        default_conv_2d_dw(x, kernel, n, c, h, w, kh, kw, s0, s1, p0, p1)
    }

    /// 1D transposed convolution
    fn conv_transpose_1d(
        &self,
        x: &[f32],
        kernel: &[f32],
        n: usize,
        c: usize,
        len: usize,
        kc: usize,
        kw: usize,
        filters: usize,
        s: usize,
        p: usize,
    ) -> Vec<f32> {
        default_conv_transpose_1d(x, kernel, n, c, len, kc, kw, filters, s, p)
    }

    /// 2D transposed convolution
    fn conv_transpose_2d(
        &self,
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
        default_conv_transpose_2d(x, kernel, n, c, h, w, kc, kh, kw, filters, s0, s1, p0, p1)
    }

    /// Image to column for efficient convolution
    fn im2col(
        &self,
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
        default_im2col(x, n, c, h, w, kh, kw, s0, s1, p0, p1)
    }

    /// 1D pooling (max or average)
    fn pool_1d(
        &self,
        x: &[f32],
        n: usize,
        c: usize,
        len: usize,
        kw: usize,
        s: usize,
        p: usize,
        is_max: bool,
    ) -> Vec<f32> {
        default_pool_1d(x, n, c, len, kw, s, p, is_max)
    }

    /// 2D pooling (max or average)
    fn pool_2d(
        &self,
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
        default_pool_2d(x, n, c, h, w, kh, kw, s0, s1, p0, p1, is_max)
    }

    /// 3D convolution
    fn conv_3d(
        &self,
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
        default_conv_3d(
            x, kernel, n, c, d, h, w, kc, kd, kh, kw, filters, s0, s1, s2, p0, p1, p2,
        )
    }

    /// 3D im2col
    fn im2col_3d(
        &self,
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
        default_im2col_3d(x, n, c, d, h, w, kd, kh, kw, s0, s1, s2, p0, p1, p2)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Special operations
    // ──────────────────────────────────────────────────────────────────────────

    /// Rotary Position Embedding
    fn rope(
        &self,
        x: &[f32],
        positions: &[i32],
        n_dims: usize,
        n_heads: usize,
        theta: f32,
        mode: i32,
    ) -> Vec<f32> {
        default_rope(x, positions, n_dims, n_heads, theta, mode)
    }

    /// RoPE backward gradient
    fn rope_back(
        &self,
        x: &[f32],
        positions: &[i32],
        n_dims: usize,
        n_heads: usize,
        theta: f32,
        mode: i32,
    ) -> Vec<f32> {
        default_rope_back(x, positions, n_dims, n_heads, theta, mode)
    }

    /// Flash attention extension
    #[expect(clippy::too_many_arguments)]
    fn flash_attn_ext(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        n_tokens_q: usize,
        n_tokens_kv: usize,
        head_size: usize,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> Vec<f32> {
        default_flash_attn_ext(
            q,
            k,
            v,
            n_heads,
            n_kv_heads,
            n_tokens_q,
            n_tokens_kv,
            head_size,
            scale,
            max_bias,
            logit_softcap,
        )
    }

    /// State Space Model convolution (Mamba)
    fn ssm_conv(
        &self,
        x: &[f32],
        kernel: &[f32],
        state: &[f32],
        n_tokens: usize,
        d_inner: usize,
        d_conv: usize,
    ) -> Vec<f32> {
        default_ssm_conv(x, kernel, state, n_tokens, d_inner, d_conv)
    }

    /// State Space Model scan (Mamba)
    fn ssm_scan(
        &self,
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        cx: &[f32],
        d_inner: usize,
        d_state: usize,
    ) -> Vec<f32> {
        default_ssm_scan(x, dt, a, b, cx, d_inner, d_state)
    }

    /// RWKV WKV6 operation
    fn rwkv_wkv6(
        &self,
        r: &[f32],
        w: &[f32],
        k: &[f32],
        v: &[f32],
        state: &[f32],
        n_tokens: usize,
        n_channels: usize,
    ) -> Vec<f32> {
        default_rwkv_wkv6(r, w, k, v, state, n_tokens, n_channels)
    }

    /// RWKV WKV7 operation
    fn rwkv_wkv7(
        &self,
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
        default_rwkv_wkv7(r, w, k, v, g, state, n_tokens, n_channels, head_size)
    }

    /// Gated Delta Net
    fn gated_delta_net(
        &self,
        x: &[f32],
        gate: &[f32],
        state: &[f32],
        n_tokens: usize,
        n_channels: usize,
    ) -> Vec<f32> {
        default_gated_delta_net(x, gate, state, n_tokens, n_channels)
    }

    /// Gated Linear Attention
    fn gated_linear_attn(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        state: &[f32],
        n_tokens: usize,
        n_channels: usize,
        head_size: usize,
    ) -> Vec<f32> {
        default_gated_linear_attn(q, k, v, state, n_tokens, n_channels, head_size)
    }

    /// Timestep embedding (sinusoidal)
    fn timestep_embedding(&self, timestep: usize, dim: usize, max_period: f32) -> Vec<f32> {
        default_timestep_embedding(timestep, dim, max_period)
    }

    /// Nearest-neighbor upscale
    fn upscale(
        &self,
        x: &[f32],
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        scale_h: usize,
        scale_w: usize,
    ) -> Vec<f32> {
        default_upscale(x, n, c, h, w, scale_h, scale_w)
    }

    /// Triangular solve: solve `A * x = b` where A is triangular
    fn solve_tri(
        &self,
        a: &[f32],
        b: &[f32],
        n: usize,
        lower: bool,
        unit_diagonal: bool,
    ) -> Vec<f32> {
        default_solve_tri(a, b, n, lower, unit_diagonal)
    }

    /// Generate triangular matrix
    fn tri(&self, n: usize, diagonal: i32) -> Vec<f32> {
        default_tri(n, diagonal)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Optimizer steps
    // ──────────────────────────────────────────────────────────────────────────

    /// AdamW optimizer step
    fn opt_step_adamw(
        &self,
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
        default_opt_step_adamw(grad, params, m, v, lr, beta1, beta2, eps, wd, t);
    }

    /// SGD optimizer step
    fn opt_step_sgd(
        &self,
        grad: &[f32],
        params: &mut [f32],
        lr: f32,
        momentum: f32,
        velocity: &mut [f32],
    ) {
        default_opt_step_sgd(grad, params, lr, momentum, velocity);
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Miscellaneous
    // ──────────────────────────────────────────────────────────────────────────

    /// Clamp values to `[min, max]`
    fn clamp(&self, x: &[f32], min: f32, max: f32) -> Vec<f32> {
        default_clamp(x, min, max)
    }

    /// Scale by factor: `y[i] = x[i] * scale`
    fn scale(&self, x: &[f32], scale: f32) -> Vec<f32> {
        default_scale(x, scale)
    }

    /// Fill vector with constant value
    fn fill(&self, n: usize, v: f32) -> Vec<f32> {
        default_fill(n, v)
    }

    /// Generate arithmetic sequence: `[start, start+step, start+2*step, ...]`
    fn arange(&self, n: usize, start: f32, step: f32) -> Vec<f32> {
        default_arange(n, start, step)
    }

    /// Top-K: returns the k largest values
    fn top_k(&self, x: &[f32], k: usize) -> Vec<f32> {
        default_top_k(x, k)
    }

    /// Argsort: returns indices sorted by value
    fn argsort(&self, x: &[f32], ascending: bool) -> Vec<usize> {
        default_argsort(x, ascending)
    }

    /// Accumulate: `y = a + b` where b is placed at an offset
    fn acc(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        default_acc(a, b)
    }
}

// ─── Default (CPU fallback) implementations ──────────────────────────────────
