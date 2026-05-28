// OpType enumeration moved from backend.rs for modularity.

/// Enumeration of all GGML operation types.
/// Each variant corresponds to a tensor operation that a backend can accelerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpType {
    // ── Matrix operations ──
    /// Matrix-vector multiplication
    MatVec,
    /// Matrix-vector multiplication with quantized matrix
    MatVecQuant,
    /// Matrix-matrix multiplication
    MatMul,
    /// Outer product of two vectors
    OutProd,
    /// Matrix-matrix multiplication with identity matrix
    MatMulId,
    /// Hadamard (element-wise) product after matrix multiplication
    MulMatHadamard,

    // ── Unary element-wise activations ──
    /// Absolute value
    Abs,
    /// Sign function
    Sgn,
    /// Negation
    Neg,
    /// Step function
    Step,
    /// Hyperbolic tangent
    Tanh,
    /// Exponential linear unit
    Elu,
    /// Rectified linear unit
    Relu,
    /// Sigmoid function
    Sigmoid,
    /// Hard sigmoid approximation
    HardSigmoid,
    /// Hard swish activation
    HardSwish,
    /// Exponential function
    Exp,
    /// Exponential minus one
    Expm1,
    /// Softplus activation
    Softplus,
    /// Floor function
    Floor,
    /// Ceiling function
    Ceil,
    /// Round to nearest integer
    Round,
    /// Truncate toward zero
    Trunc,
    /// Sine function
    Sin,
    /// Cosine function
    Cos,
    /// Square function
    Sqr,
    /// Square root function
    Sqrt,
    /// SiLU backward pass
    SiluBack,
    /// Leaky ReLU activation
    LeakyRelu,
    /// GELU activation (error function variant)
    GeluErf,
    /// GELU activation (quick approximation)
    GeluQuick,
    /// SiLU (sigmoid linear unit) activation
    Silu,
    /// GELU activation
    Gelu,

    // ── Binary element-wise operations ──
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Add scalar (x + 1)
    Add1,
    /// SiGLU activation
    Swiglu,
    /// SiGLU activation (OpenAI variant)
    SwigluOai,
    /// GEGLU activation (error function variant)
    GegluErf,
    /// GEGLU activation (quick approximation)
    GegluQuick,
    /// GeGLU activation
    Geglu,
    /// ReGLU activation
    Reglu,
    /// XieLU activation
    Xielu,

    // ── Reduction operations ──
    /// L2 normalization
    Norm,
    /// Group normalization
    GroupNorm,
    /// L2 norm (square root of sum of squares)
    L2Norm,
    /// Root mean square normalization
    RmsNorm,
    /// Root mean square normalization backward pass
    RmsNormBack,
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// Argmax (index of maximum value)
    Argmax,
    /// Count equal elements
    CountEqual,
    /// Cumulative sum
    Cumsum,
    /// Sum rows of matrix
    SumRows,
    /// Softmax activation
    SoftMax,
    /// Softmax backward pass
    SoftMaxBack,
    /// Cross entropy loss
    CrossEntropyLoss,
    /// Cross entropy loss backward pass
    CrossEntropyLossBack,
    /// Concatenation of tensors
    Concat,
    /// Repeat tensor elements
    Repeat,
    /// Repeat backward pass
    RepeatBack,
    /// Padding operation
    Pad,
    /// 1D reflect padding
    PadReflect1d,
    /// Roll (circular shift) operation
    Roll,
    /// Diagonal extraction/construction
    Diag,
    /// Diagonal with negative infinity masking
    DiagMaskInf,
    /// Duplicate tensor along dimension
    Dup,
    /// Continue (no-op)
    Cont,
    /// Get rows from tensor
    GetRows,
    /// Get rows backward pass
    GetRowsBack,
    /// Set rows in tensor
    SetRows,

    // ── Convolution operations ──
    /// 2D convolution
    Conv2d,
    /// 2D depth-wise convolution
    Conv2dDw,
    /// 1D transposed convolution
    ConvTranspose1d,
    /// 2D transposed convolution
    ConvTranspose2d,
    /// Image to column transformation
    Im2col,
    /// 1D pooling operation
    Pool1d,
    /// 2D pooling operation
    Pool2d,
    /// 3D convolution
    Conv3d,
    /// 3D image to column transformation
    Im2col3d,

    // ── Positional embeddings ──
    /// Rotary positional embedding
    Rope,
    /// Rotary positional embedding backward pass
    RopeBack,
    /// Flash attention extended
    FlashAttnExt,
    /// State space model convolution
    SsmConv,
    /// State space model scan
    SsmScan,
    /// RWKV wkv6 computation
    RwkvWkv6,
    /// RWKV wkv7 computation
    RwkvWkv7,
    /// Gated delta network
    GatedDeltaNet,
    /// Gated linear attention
    GatedLinearAttn,
    /// Timestep embedding
    TimestepEmbedding,
    /// Upscale operation
    Upscale,
    /// Solve triangular system
    SolveTri,
    /// Triangular matrix operation
    Tri,

    // ── Optimization operations ──
    /// AdamW optimizer step
    OptStepAdamW,
    /// SGD optimizer step
    OptStepSGD,

    // ── Misc operations ──
    /// Clamp values to range
    Clamp,
    /// Scale tensor by scalar
    Scale,
    /// Fill tensor with constant value
    Fill,
    /// Generate arithmetic sequence
    Arange,
    /// Top-K selection
    TopK,
    /// Argsort (indices that would sort tensor)
    Argsort,
    /// Accumulate (add with offset)
    Acc,
}
