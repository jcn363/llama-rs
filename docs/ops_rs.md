# GGML Operations — Rust Implementation

Comprehensive reference for GGML tensor operations in the `llama-rs` Rust port.

## Overview

The `ggml` crate defines **108 operation types** via the `OpType` enum and **104 `Backend` trait methods** (each with a default CPU implementation). Two concrete backends exist:

| Backend  | Crate       | GPU-Accelerated Ops      | Notes                                                 |
| -------- | ----------- | ------------------------ | ----------------------------------------------------- |
| **CPU**  | `ggml-cpu`  | SIMD matvec (AVX/SSE4.2) | All 104 operations supported                          |
| **CUDA** | `ggml-cuda` | mat_vec only (cuBLAS)    | 1 GPU op, 5 explicit CPU fallbacks, 98 trait defaults |

## Operation Categories

Operations are defined in `crates/ggml/src/backend.rs` as the `OpType` enum (108 variants) with corresponding `Backend` trait methods (104 methods).

### Matrix Operations (6)

| Operation        | OpType           | Backend Method       | CPU     | CUDA            | Notes                            |
| ---------------- | ---------------- | -------------------- | ------- | --------------- | -------------------------------- |
| MAT_VEC          | `MatVec`         | `mat_vec()`          | ✅ SIMD | ✅ cuBLAS       | Only GPU-accelerated operation   |
| MAT_VEC_QUANT    | `MatVecQuant`    | `mat_vec_quant()`    | ✅      | ❌ CPU fallback | Dequantizes on CPU then computes |
| MAT_MUL          | `MatMul`         | `mat_mul()`          | ✅      | ❌ CPU default  | Full matrix-matrix multiply      |
| OUT_PROD         | `OutProd`        | `out_prod()`         | ✅      | ❌ CPU default  | Outer product                    |
| MAT_MUL_ID       | `MatMulId`       | `mat_mul_id()`       | ✅      | ❌ CPU default  | Identity-aware matmul            |
| MUL_MAT_HADAMARD | `MulMatHadamard` | `mul_mat_hadamard()` | ✅      | ❌ CPU default  | Element-wise Hadamard product    |

### Unary Activations (25)

| Operation    | OpType        | Backend Method   | CPU | CUDA           |
| ------------ | ------------- | ---------------- | --- | -------------- |
| ABS          | `Abs`         | `abs()`          | ✅  | ❌ CPU default |
| SGN          | `Sgn`         | `sgn()`          | ✅  | ❌ CPU default |
| NEG          | `Neg`         | `neg()`          | ✅  | ❌ CPU default |
| STEP         | `Step`        | `step()`         | ✅  | ❌ CPU default |
| TANH         | `Tanh`        | `tanh()`         | ✅  | ❌ CPU default |
| ELU          | `Elu`         | `elu()`          | ✅  | ❌ CPU default |
| RELU         | `Relu`        | `relu()`         | ✅  | ❌ CPU default |
| SIGMOID      | `Sigmoid`     | `sigmoid()`      | ✅  | ❌ CPU default |
| HARD_SIGMOID | `HardSigmoid` | `hard_sigmoid()` | ✅  | ❌ CPU default |
| HARD_SWISH   | `HardSwish`   | `hard_swish()`   | ✅  | ❌ CPU default |
| EXP          | `Exp`         | `exp()`          | ✅  | ❌ CPU default |
| EXPM1        | `Expm1`       | `expm1()`        | ✅  | ❌ CPU default |
| SOFTPLUS     | `Softplus`    | `softplus()`     | ✅  | ❌ CPU default |
| FLOOR        | `Floor`       | `floor()`        | ✅  | ❌ CPU default |
| CEIL         | `Ceil`        | `ceil()`         | ✅  | ❌ CPU default |
| ROUND        | `Round`       | `round()`        | ✅  | ❌ CPU default |
| TRUNC        | `Trunc`       | `trunc()`        | ✅  | ❌ CPU default |
| SIN          | `Sin`         | `sin()`          | ✅  | ❌ CPU default |
| COS          | `Cos`         | `cos()`          | ✅  | ❌ CPU default |
| SQR          | `Sqr`         | `sqr()`          | ✅  | ❌ CPU default |
| SQRT         | `Sqrt`        | `sqrt()`         | ✅  | ❌ CPU default |
| SILU_BACK    | `SiluBack`    | `silu_back()`    | ✅  | ❌ CPU default |
| LEAKY_RELU   | `LeakyRelu`   | `leaky_relu()`   | ✅  | ❌ CPU default |
| GELU_ERF     | `GeluErf`     | `gelu_erf()`     | ✅  | ❌ CPU default |
| GELU_QUICK   | `GeluQuick`   | `gelu_quick()`   | ✅  | ❌ CPU default |

### Activation Aliases (2)

| Operation | OpType | Backend Method | CPU | CUDA            | Notes                      |
| --------- | ------ | -------------- | --- | --------------- | -------------------------- |
| SILU      | `Silu` | `silu()`       | ✅  | ⚠️ CPU fallback | Explicit override, not GPU |
| GELU      | `Gelu` | `gelu()`       | ✅  | ⚠️ CPU fallback | Explicit override, not GPU |

### Binary Element-wise (5)

| Operation | OpType | Backend Method | CPU | CUDA            | Notes                      |
| --------- | ------ | -------------- | --- | --------------- | -------------------------- |
| ADD       | `Add`  | `add()`        | ✅  | ⚠️ CPU fallback | Explicit override, not GPU |
| SUB       | `Sub`  | `sub()`        | ✅  | ❌ CPU default  |                            |
| MUL       | `Mul`  | `mul()`        | ✅  | ⚠️ CPU fallback | Explicit override, not GPU |
| DIV       | `Div`  | `div()`        | ✅  | ❌ CPU default  |                            |
| ADD1      | `Add1` | `add1()`       | ✅  | ❌ CPU default  | Scalar broadcast           |

### Gated Activations (7)

| Operation   | OpType       | Backend Method  | CPU | CUDA           |
| ----------- | ------------ | --------------- | --- | -------------- |
| SWIGLU      | `Swiglu`     | `swiglu()`      | ✅  | ❌ CPU default |
| SWIGLU_OAI  | `SwigluOai`  | `swiglu_oai()`  | ✅  | ❌ CPU default |
| GEGLU       | `Geglu`      | `geglu()`       | ✅  | ❌ CPU default |
| REGLU       | `Reglu`      | `reglu()`       | ✅  | ❌ CPU default |
| GEGLU_ERF   | `GegluErf`   | `geglu_erf()`   | ✅  | ❌ CPU default |
| GEGLU_QUICK | `GegluQuick` | `geglu_quick()` | ✅  | ❌ CPU default |
| XIELU       | `Xielu`      | `xielu()`       | ✅  | ❌ CPU default |

### Normalization (5)

| Operation     | OpType        | Backend Method    | CPU | CUDA            | Notes                      |
| ------------- | ------------- | ----------------- | --- | --------------- | -------------------------- |
| RMS_NORM      | `RmsNorm`     | `rms_norm()`      | ✅  | ⚠️ CPU fallback | Explicit override, not GPU |
| RMS_NORM_BACK | `RmsNormBack` | `rms_norm_back()` | ✅  | ❌ CPU default  |                            |
| NORM          | `Norm`        | `norm()`          | ✅  | ❌ CPU default  | Layer normalization        |
| GROUP_NORM    | `GroupNorm`   | `group_norm()`    | ✅  | ❌ CPU default  |                            |
| L2_NORM       | `L2Norm`      | `l2_norm()`       | ✅  | ❌ CPU default  |                            |

### Reduction (10)

| Operation               | OpType                 | Backend Method              | CPU | CUDA           |
| ----------------------- | ---------------------- | --------------------------- | --- | -------------- |
| SUM                     | `Sum`                  | `sum()`                     | ✅  | ❌ CPU default |
| MEAN                    | `Mean`                 | `mean()`                    | ✅  | ❌ CPU default |
| ARGMAX                  | `Argmax`               | `argmax()`                  | ✅  | ❌ CPU default |
| COUNT_EQUAL             | `CountEqual`           | `count_equal()`             | ✅  | ❌ CPU default |
| CUMSUM                  | `Cumsum`               | `cumsum()`                  | ✅  | ❌ CPU default |
| SUM_ROWS                | `SumRows`              | `sum_rows()`                | ✅  | ❌ CPU default |
| SOFT_MAX                | `SoftMax`              | `soft_max()`                | ✅  | ❌ CPU default |
| SOFT_MAX_BACK           | `SoftMaxBack`          | `soft_max_back()`           | ✅  | ❌ CPU default |
| CROSS_ENTROPY_LOSS      | `CrossEntropyLoss`     | `cross_entropy_loss()`      | ✅  | ❌ CPU default |
| CROSS_ENTROPY_LOSS_BACK | `CrossEntropyLossBack` | `cross_entropy_loss_back()` | ✅  | ❌ CPU default |

### Shape Manipulation (13)

| Operation      | OpType         | Backend Method     | CPU | CUDA           |
| -------------- | -------------- | ------------------ | --- | -------------- |
| CONCAT         | `Concat`       | `concat()`         | ✅  | ❌ CPU default |
| REPEAT         | `Repeat`       | `repeat()`         | ✅  | ❌ CPU default |
| REPEAT_BACK    | `RepeatBack`   | `repeat_back()`    | ✅  | ❌ CPU default |
| PAD            | `Pad`          | `pad()`            | ✅  | ❌ CPU default |
| PAD_REFLECT_1D | `PadReflect1d` | `pad_reflect_1d()` | ✅  | ❌ CPU default |
| ROLL           | `Roll`         | `roll()`           | ✅  | ❌ CPU default |
| DIAG           | `Diag`         | `diag()`           | ✅  | ❌ CPU default |
| DIAG_MASK_INF  | `DiagMaskInf`  | `diag_mask_inf()`  | ✅  | ❌ CPU default |
| DUP            | `Dup`          | `dup()`            | ✅  | ❌ CPU default |
| CONT           | `Cont`         | `cont()`           | ✅  | ❌ CPU default |
| GET_ROWS       | `GetRows`      | `get_rows()`       | ✅  | ❌ CPU default |
| GET_ROWS_BACK  | `GetRowsBack`  | `get_rows_back()`  | ✅  | ❌ CPU default |
| SET_ROWS       | `SetRows`      | `set_rows()`       | ✅  | ❌ CPU default |

### Convolution (9)

| Operation         | OpType            | Backend Method        | CPU | CUDA           |
| ----------------- | ----------------- | --------------------- | --- | -------------- |
| CONV_2D           | `Conv2d`          | `conv_2d()`           | ✅  | ❌ CPU default |
| CONV_2D_DW        | `Conv2dDw`        | `conv_2d_dw()`        | ✅  | ❌ CPU default |
| CONV_TRANSPOSE_1D | `ConvTranspose1d` | `conv_transpose_1d()` | ✅  | ❌ CPU default |
| CONV_TRANSPOSE_2D | `ConvTranspose2d` | `conv_transpose_2d()` | ✅  | ❌ CPU default |
| IM2COL            | `Im2col`          | `im2col()`            | ✅  | ❌ CPU default |
| POOL_1D           | `Pool1d`          | `pool_1d()`           | ✅  | ❌ CPU default |
| POOL_2D           | `Pool2d`          | `pool_2d()`           | ✅  | ❌ CPU default |
| CONV_3D           | `Conv3d`          | `conv_3d()`           | ✅  | ❌ CPU default |
| IM2COL_3D         | `Im2col3d`        | `im2col_3d()`         | ✅  | ❌ CPU default |

### Special Operations / Positional Embeddings (14)

| Operation          | OpType              | Backend Method         | CPU | CUDA           |
| ------------------ | ------------------- | ---------------------- | --- | -------------- |
| ROPE               | `Rope`              | `rope()`               | ✅  | ❌ CPU default |
| ROPE_BACK          | `RopeBack`          | `rope_back()`          | ✅  | ❌ CPU default |
| FLASH_ATTN_EXT     | `FlashAttnExt`      | `flash_attn_ext()`     | ✅  | ❌ CPU default |
| SSM_CONV           | `SsmConv`           | `ssm_conv()`           | ✅  | ❌ CPU default |
| SSM_SCAN           | `SsmScan`           | `ssm_scan()`           | ✅  | ❌ CPU default |
| RWKV_WKV6          | `RwkvWkv6`          | `rwkv_wkv6()`          | ✅  | ❌ CPU default |
| RWKV_WKV7          | `RwkvWkv7`          | `rwkv_wkv7()`          | ✅  | ❌ CPU default |
| GATED_DELTA_NET    | `GatedDeltaNet`     | `gated_delta_net()`    | ✅  | ❌ CPU default |
| GATED_LINEAR_ATTN  | `GatedLinearAttn`   | `gated_linear_attn()`  | ✅  | ❌ CPU default |
| TIMESTEP_EMBEDDING | `TimestepEmbedding` | `timestep_embedding()` | ✅  | ❌ CPU default |
| UPSCALE            | `Upscale`           | `upscale()`            | ✅  | ❌ CPU default |
| SOLVE_TRI          | `SolveTri`          | `solve_tri()`          | ✅  | ❌ CPU default |
| TRI                | `Tri`               | `tri()`                | ✅  | ❌ CPU default |

### Optimizer Steps (2)

| Operation      | OpType         | Backend Method     | CPU | CUDA           |
| -------------- | -------------- | ------------------ | --- | -------------- |
| OPT_STEP_ADAMW | `OptStepAdamW` | `opt_step_adamw()` | ✅  | ❌ CPU default |
| OPT_STEP_SGD   | `OptStepSGD`   | `opt_step_sgd()`   | ✅  | ❌ CPU default |

### Miscellaneous (7)

| Operation | OpType    | Backend Method | CPU | CUDA           |
| --------- | --------- | -------------- | --- | -------------- |
| CLAMP     | `Clamp`   | `clamp()`      | ✅  | ❌ CPU default |
| SCALE     | `Scale`   | `scale()`      | ✅  | ❌ CPU default |
| FILL      | `Fill`    | `fill()`       | ✅  | ❌ CPU default |
| ARANGE    | `Arange`  | `arange()`     | ✅  | ❌ CPU default |
| TOP_K     | `TopK`    | `top_k()`      | ✅  | ❌ CPU default |
| ARGSORT   | `Argsort` | `argsort()`    | ✅  | ❌ CPU default |
| ACC       | `Acc`     | `acc()`        | ✅  | ❌ CPU default |

## Legend

| Symbol | Meaning                                                           |
| ------ | ----------------------------------------------------------------- |
| ✅     | Fully supported (GPU-accelerated or SIMD-optimized)               |
| ⚠️     | Explicitly overridden but uses CPU fallback (not GPU-accelerated) |
| ❌     | Uses default CPU implementation from `ggml::defaults`             |

## CUDA Backend Details

The CUDA backend (`crates/ggml-cuda`) provides minimal GPU acceleration:

### GPU-Accelerated Operations

| Operation | Implementation               | Notes                                     |
| --------- | ---------------------------- | ----------------------------------------- |
| `mat_vec` | cuBLAS `sgemm` (C = A × B^T) | Copies H2D, computes, copies D2H per call |

### Explicit CPU Fallbacks

These operations are overridden in `CudaBackend` but intentionally delegate to CPU defaults:

| Operation  | Reason                                |
| ---------- | ------------------------------------- |
| `add`      | "Cheap operation" — CPU is sufficient |
| `mul`      | "Cheap operation" — CPU is sufficient |
| `rms_norm` | "CPU fallback implementation"         |
| `silu`     | "CPU fallback implementation"         |
| `gelu`     | "CPU fallback implementation"         |

### Structural Limitations

1. **No custom CUDA kernels**: Zero `.cu` or `.cuh` files — only cuBLAS `sgemm`
2. **F32 only**: `copy_to_device()` rejects non-F32 dtypes; quantized formats cannot be transferred to GPU
3. **No persistent GPU state**: Every `mat_vec` call copies weight + input to GPU, computes, copies back
4. **No quantized GPU matmul**: `mat_vec_quant()` falls back to CPU dequantization

## CPU Backend Details

The CPU backend (`crates/ggml-cpu`) provides SIMD-accelerated matrix operations:

| Operation       | SIMD         | Notes                                            |
| --------------- | ------------ | ------------------------------------------------ |
| `mat_vec`       | AVX / SSE4.2 | Block-tiled, 2.5–3.2× speedup for large matrices |
| `mat_vec_quant` | Scalar       | Dequantizes Q4_0/Q4_1/Q8_0 block-by-block        |
| All other ops   | Scalar       | Via `ggml::defaults` implementations             |

## Source References

| File                             | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `crates/ggml/src/backend.rs`     | `OpType` enum (108 variants) + `Backend` trait (104 methods) |
| `crates/ggml/src/defaults.rs`    | Default CPU implementations for all 104 Backend methods      |
| `crates/ggml-cpu/src/backend.rs` | CPU backend with SIMD matvec override                        |
| `crates/ggml-cuda/src/lib.rs`    | CUDA backend with cuBLAS matvec                              |
