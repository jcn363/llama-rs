# GGML Operations — Rust Implementation

Comprehensive reference for GGML tensor operations in the `llama-rs` Rust port.

## Overview

The `ggml` crate defines **105 `Backend` trait methods** (2 required, 103 with default CPU implementations). Two concrete backends exist:

| Backend  | Crate       | GPU-Accelerated Ops                      | Notes                                        |
| -------- | ----------- | ---------------------------------------- | -------------------------------------------- |
| **CPU**  | `ggml-cpu`  | mat_vec + mat_mul + mat_vec_quant (SIMD) | All 105 operations supported                 |
| **CUDA** | `ggml-cuda` | mat_vec + mat_mul (cuBLAS)               | 2 GPU ops, 103 trait defaults (CPU fallback) |

## Architecture

```text
Backend trait (105 methods)
  ├── info()              — required, no default
  ├── mat_vec()           — required, overridden (CPU: SIMD, CUDA: cuBLAS)
  ├── mat_mul()           — overridden (CPU: SIMD block-tiled, CUDA: cuBLAS gemm)
  └── 102 methods         — default implementations in ggml::defaults
       ├── abs, sgn, neg, step, tanh, elu, relu, sigmoid, ...
       ├── add, sub, mul, div, add1
       ├── swiglu, geglu, reglu, xielu, ...
       ├── rms_norm, norm, group_norm, l2_norm
       ├── sum, mean, argmax, soft_max, cross_entropy_loss, ...
       ├── concat, repeat, pad, roll, get_rows, set_rows, ...
       ├── conv_2d, pool_2d, im2col, rope, flash_attn_ext, ...
       └── clamp, scale, fill, arange, top_k, argsort, acc
```

## Operation Categories

### Matrix Operations (6)

| Operation        | Backend Method       | CPU                 | CUDA           | Notes                                  |
| ---------------- | -------------------- | ------------------- | -------------- | -------------------------------------- |
| MAT_VEC          | `mat_vec()`          | ✅ SIMD dot         | ✅ cuBLAS      | GPU-accelerated matrix-vector product  |
| MAT_VEC_QUANT    | `mat_vec_quant()`    | ✅ quant kernels    | ❌ CPU default | Direct Q4_0/Q4_1/Q8_0 dot, no dequant  |
| MAT_MUL          | `mat_mul()`          | ✅ SIMD block-tiled | ✅ cuBLAS gemm | GPU-accelerated matrix-matrix multiply |
| OUT_PROD         | `out_prod()`         | ✅                  | ❌ CPU default | Outer product                          |
| MAT_MUL_ID       | `mat_mul_id()`       | ✅                  | ❌ CPU default | Identity-aware matmul                  |
| MUL_MAT_HADAMARD | `mul_mat_hadamard()` | ✅                  | ❌ CPU default | Element-wise Hadamard product          |

### Unary Activations (25)

| Operation    | Backend Method   | CPU | CUDA           |
| ------------ | ---------------- | --- | -------------- |
| ABS          | `abs()`          | ✅  | ❌ CPU default |
| SGN          | `sgn()`          | ✅  | ❌ CPU default |
| NEG          | `neg()`          | ✅  | ❌ CPU default |
| STEP         | `step()`         | ✅  | ❌ CPU default |
| TANH         | `tanh()`         | ✅  | ❌ CPU default |
| ELU          | `elu()`          | ✅  | ❌ CPU default |
| RELU         | `relu()`         | ✅  | ❌ CPU default |
| SIGMOID      | `sigmoid()`      | ✅  | ❌ CPU default |
| HARD_SIGMOID | `hard_sigmoid()` | ✅  | ❌ CPU default |
| HARD_SWISH   | `hard_swish()`   | ✅  | ❌ CPU default |
| EXP          | `exp()`          | ✅  | ❌ CPU default |
| EXPM1        | `expm1()`        | ✅  | ❌ CPU default |
| SOFTPLUS     | `softplus()`     | ✅  | ❌ CPU default |
| FLOOR        | `floor()`        | ✅  | ❌ CPU default |
| CEIL         | `ceil()`         | ✅  | ❌ CPU default |
| ROUND        | `round()`        | ✅  | ❌ CPU default |
| TRUNC        | `trunc()`        | ✅  | ❌ CPU default |
| SIN          | `sin()`          | ✅  | ❌ CPU default |
| COS          | `cos()`          | ✅  | ❌ CPU default |
| SQR          | `sqr()`          | ✅  | ❌ CPU default |
| SQRT         | `sqrt()`         | ✅  | ❌ CPU default |
| SILU_BACK    | `silu_back()`    | ✅  | ❌ CPU default |
| LEAKY_RELU   | `leaky_relu()`   | ✅  | ❌ CPU default |
| GELU_ERF     | `gelu_erf()`     | ✅  | ❌ CPU default |
| GELU_QUICK   | `gelu_quick()`   | ✅  | ❌ CPU default |

### Activation Aliases (2)

| Operation | Backend Method | CPU | CUDA           |
| --------- | -------------- | --- | -------------- |
| SILU      | `silu()`       | ✅  | ❌ CPU default |
| GELU      | `gelu()`       | ✅  | ❌ CPU default |

### Binary Element-wise (5)

| Operation | Backend Method | CPU | CUDA           |
| --------- | -------------- | --- | -------------- |
| ADD       | `add()`        | ✅  | ❌ CPU default |
| SUB       | `sub()`        | ✅  | ❌ CPU default |
| MUL       | `mul()`        | ✅  | ❌ CPU default |
| DIV       | `div()`        | ✅  | ❌ CPU default |
| ADD1      | `add1()`       | ✅  | ❌ CPU default |

### Gated Activations (7)

| Operation   | Backend Method  | CPU | CUDA           |
| ----------- | --------------- | --- | -------------- |
| SWIGLU      | `swiglu()`      | ✅  | ❌ CPU default |
| SWIGLU_OAI  | `swiglu_oai()`  | ✅  | ❌ CPU default |
| GEGLU       | `geglu()`       | ✅  | ❌ CPU default |
| REGLU       | `reglu()`       | ✅  | ❌ CPU default |
| GEGLU_ERF   | `geglu_erf()`   | ✅  | ❌ CPU default |
| GEGLU_QUICK | `geglu_quick()` | ✅  | ❌ CPU default |
| XIELU       | `xielu()`       | ✅  | ❌ CPU default |

### Normalization (5)

| Operation     | Backend Method    | CPU | CUDA           |
| ------------- | ----------------- | --- | -------------- |
| RMS_NORM      | `rms_norm()`      | ✅  | ❌ CPU default |
| RMS_NORM_BACK | `rms_norm_back()` | ✅  | ❌ CPU default |
| NORM          | `norm()`          | ✅  | ❌ CPU default |
| GROUP_NORM    | `group_norm()`    | ✅  | ❌ CPU default |
| L2_NORM       | `l2_norm()`       | ✅  | ❌ CPU default |

### Reduction (10)

| Operation               | Backend Method              | CPU | CUDA           |
| ----------------------- | --------------------------- | --- | -------------- |
| SUM                     | `sum()`                     | ✅  | ❌ CPU default |
| MEAN                    | `mean()`                    | ✅  | ❌ CPU default |
| ARGMAX                  | `argmax()`                  | ✅  | ❌ CPU default |
| COUNT_EQUAL             | `count_equal()`             | ✅  | ❌ CPU default |
| CUMSUM                  | `cumsum()`                  | ✅  | ❌ CPU default |
| SUM_ROWS                | `sum_rows()`                | ✅  | ❌ CPU default |
| SOFT_MAX                | `soft_max()`                | ✅  | ❌ CPU default |
| SOFT_MAX_BACK           | `soft_max_back()`           | ✅  | ❌ CPU default |
| CROSS_ENTROPY_LOSS      | `cross_entropy_loss()`      | ✅  | ❌ CPU default |
| CROSS_ENTROPY_LOSS_BACK | `cross_entropy_loss_back()` | ✅  | ❌ CPU default |

### Shape Manipulation (13)

| Operation      | Backend Method     | CPU | CUDA           |
| -------------- | ------------------ | --- | -------------- |
| CONCAT         | `concat()`         | ✅  | ❌ CPU default |
| REPEAT         | `repeat()`         | ✅  | ❌ CPU default |
| REPEAT_BACK    | `repeat_back()`    | ✅  | ❌ CPU default |
| PAD            | `pad()`            | ✅  | ❌ CPU default |
| PAD_REFLECT_1D | `pad_reflect_1d()` | ✅  | ❌ CPU default |
| ROLL           | `roll()`           | ✅  | ❌ CPU default |
| DIAG           | `diag()`           | ✅  | ❌ CPU default |
| DIAG_MASK_INF  | `diag_mask_inf()`  | ✅  | ❌ CPU default |
| DUP            | `dup()`            | ✅  | ❌ CPU default |
| CONT           | `cont()`           | ✅  | ❌ CPU default |
| GET_ROWS       | `get_rows()`       | ✅  | ❌ CPU default |
| GET_ROWS_BACK  | `get_rows_back()`  | ✅  | ❌ CPU default |
| SET_ROWS       | `set_rows()`       | ✅  | ❌ CPU default |

### Convolution (9)

| Operation         | Backend Method        | CPU | CUDA           |
| ----------------- | --------------------- | --- | -------------- |
| CONV_2D           | `conv_2d()`           | ✅  | ❌ CPU default |
| CONV_2D_DW        | `conv_2d_dw()`        | ✅  | ❌ CPU default |
| CONV_TRANSPOSE_1D | `conv_transpose_1d()` | ✅  | ❌ CPU default |
| CONV_TRANSPOSE_2D | `conv_transpose_2d()` | ✅  | ❌ CPU default |
| IM2COL            | `im2col()`            | ✅  | ❌ CPU default |
| POOL_1D           | `pool_1d()`           | ✅  | ❌ CPU default |
| POOL_2D           | `pool_2d()`           | ✅  | ❌ CPU default |
| CONV_3D           | `conv_3d()`           | ✅  | ❌ CPU default |
| IM2COL_3D         | `im2col_3d()`         | ✅  | ❌ CPU default |

### Special Operations (14)

| Operation          | Backend Method         | CPU | CUDA           |
| ------------------ | ---------------------- | --- | -------------- |
| ROPE               | `rope()`               | ✅  | ❌ CPU default |
| ROPE_BACK          | `rope_back()`          | ✅  | ❌ CPU default |
| FLASH_ATTN_EXT     | `flash_attn_ext()`     | ✅  | ❌ CPU default |
| SSM_CONV           | `ssm_conv()`           | ✅  | ❌ CPU default |
| SSM_SCAN           | `ssm_scan()`           | ✅  | ❌ CPU default |
| RWKV_WKV6          | `rwkv_wkv6()`          | ✅  | ❌ CPU default |
| RWKV_WKV7          | `rwkv_wkv7()`          | ✅  | ❌ CPU default |
| GATED_DELTA_NET    | `gated_delta_net()`    | ✅  | ❌ CPU default |
| GATED_LINEAR_ATTN  | `gated_linear_attn()`  | ✅  | ❌ CPU default |
| TIMESTEP_EMBEDDING | `timestep_embedding()` | ✅  | ❌ CPU default |
| UPSCALE            | `upscale()`            | ✅  | ❌ CPU default |
| SOLVE_TRI          | `solve_tri()`          | ✅  | ❌ CPU default |
| TRI                | `tri()`                | ✅  | ❌ CPU default |

### Optimizer Steps (2)

| Operation      | Backend Method     | CPU | CUDA           |
| -------------- | ------------------ | --- | -------------- |
| OPT_STEP_ADAMW | `opt_step_adamw()` | ✅  | ❌ CPU default |
| OPT_STEP_SGD   | `opt_step_sgd()`   | ✅  | ❌ CPU default |

### Miscellaneous (7)

| Operation | Backend Method | CPU | CUDA           |
| --------- | -------------- | --- | -------------- |
| CLAMP     | `clamp()`      | ✅  | ❌ CPU default |
| SCALE     | `scale()`      | ✅  | ❌ CPU default |
| FILL      | `fill()`       | ✅  | ❌ CPU default |
| ARANGE    | `arange()`     | ✅  | ❌ CPU default |
| TOP_K     | `top_k()`      | ✅  | ❌ CPU default |
| ARGSORT   | `argsort()`    | ✅  | ❌ CPU default |
| ACC       | `acc()`        | ✅  | ❌ CPU default |

## Legend

| Symbol | Meaning                                               |
| ------ | ----------------------------------------------------- |
| ✅     | Fully supported (GPU-accelerated or SIMD-optimized)   |
| ❌     | Uses default CPU implementation from `ggml::defaults` |

## CUDA Backend Details

The CUDA backend (`crates/ggml-cuda`) provides GPU acceleration for matrix operations:

### GPU-Accelerated Operations

| Operation | Implementation               | Notes                                                |
| --------- | ---------------------------- | ---------------------------------------------------- |
| `mat_vec` | cuBLAS `sgemm` (C = A × B^T) | Copies H2D, computes, copies D2H per call            |
| `mat_mul` | cuBLAS `sgemm` (C = A × B^T) | Copies both tensors to GPU, gemm, copies result back |

### Fallback Mechanism

All 103 non-mat_vec/mat_mul operations use the `Backend` trait's default implementations, which delegate to `ggml::defaults::*` functions. These are pure scalar CPU implementations. This is the **intended design**: the CUDA backend accelerates the hot path (matrix-vector and matrix-matrix multiply) while relying on correct CPU fallback for everything else.

### Structural Limitations

1. **No custom CUDA kernels**: Zero `.cu` or `.cuh` files — only cuBLAS `sgemm`
2. **F32 only**: `copy_to_device()` rejects non-F32 dtypes; quantized formats cannot be transferred to GPU
3. **No persistent GPU state**: Every call copies tensors to GPU, computes, copies back
4. **No quantized GPU matmul**: `mat_vec_quant()` falls back to CPU dequantization

## CPU Backend Details

The CPU backend (`crates/ggml-cpu`) provides SIMD-accelerated matrix operations:

| Operation       | SIMD / Kernel          | Notes                                           |
| --------------- | ---------------------- | ----------------------------------------------- |
| `mat_vec`       | AVX / SSE4.2 dot       | 8-wide unrolling, 2.5–3.2x speedup              |
| `mat_mul`       | AVX / SSE4.2 block     | 16×16 block-tiled, parallel row execution       |
| `mat_vec_quant` | Q4_0/Q4_1/Q8_0 kernels | Direct quantized dot, 2-4x over dequant-compute |
| All other ops   | Scalar                 | Via `ggml::defaults` implementations            |

## Source References

| File                             | Description                                                          |
| -------------------------------- | -------------------------------------------------------------------- |
| `crates/ggml/src/backend.rs`     | `Backend` trait (105 methods)                                        |
| `crates/ggml/src/defaults.rs`    | Default CPU implementations for all 102 non-required Backend methods |
| `crates/ggml-cpu/src/backend.rs` | CPU backend with SIMD mat_vec + mat_mul + quant_dot overrides        |
| `crates/ggml-cuda/src/lib.rs`    | CUDA backend with cuBLAS mat_vec + mat_mul                           |
