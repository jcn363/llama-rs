# llama-rs Improvement Plan

> **For agentic workers:** Use subagent-driven-development or inline executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Address highest-impact improvement opportunities: quantized computation, backend expansion, tokenizer correctness, server batching, and hardware code paths.

**Architecture:** Six independent phases ordered by impact. Each produces working, testable code.

**Tech Stack:** Rust edition 2024, AVX/SSE4.2 SIMD, CUDA/cuBLAS, axum/tokio, bumpalo, criterion

---

## Phase 1: Quantized Dot Product Kernels

### Task 1.1: Add quantized dot product trait + Q4_0 kernel ✅

**Files:**
- ✅ Create: `crates/ggml-cpu/src/quant_dot.rs` — **EXISTS**
- ✅ Create: `crates/ggml-cpu/src/quant_dot/` (directory for sub-modules) — **EXISTS**
- ✅ Create: `crates/ggml-cpu/src/quant_dot/q4_0.rs` — **EXISTS**
- Modify: `crates/ggml-cpu/src/lib.rs`

**Step 1.1.1: Define the QuantDot trait and module structure in `quant_dot.rs`**

```rust
use half::f16;

/// Trait for quantized dot product implementations.
/// `block_size` is the number of f32 values per quantized block.
pub trait QuantDot: Send + Sync {
    fn dot_block(&self, quantized: &[u8], input: &[f32]) -> f32;
    fn block_size(&self) -> usize;
    fn block_bytes(&self) -> usize;
}

/// Compute dot product of a quantized weight row with an f32 input vector.
pub fn quant_dot_row<T: QuantDot>(
    kernel: &T,
    quantized_row: &[u8],
    input: &[f32],
    cols: usize,
) -> f32 {
    let block_size = kernel.block_size();
    let block_bytes = kernel.block_bytes();
    let n_blocks = cols.div_ceil(block_size);
    let mut sum = 0.0f32;
    for b in 0..n_blocks {
        let q_start = b * block_bytes;
        let i_start = b * block_size;
        let q_block = &quantized_row[q_start..q_start + block_bytes];
        let remaining = cols.saturating_sub(i_start);
        let i_end = i_start + remaining.min(block_size);
        let mut padded = [0.0f32; 32];
        let actual = &input[i_start..i_end];
        padded[..actual.len()].copy_from_slice(actual);
        sum += kernel.dot_block(q_block, &padded[..block_size]);
    }
    sum
}

pub mod q4_0;
```

**Step 1.1.2: Implement Q4_0 kernel in `quant_dot/q4_0.rs`**

Q4_0 block layout: [f16 scale (2 bytes)] + [16 bytes of 4-bit nibbles] = 18 bytes → 32 f32 values.

```rust
use crate::quant_dot::QuantDot;
use half::f16;

pub struct Q4_0Dot;

impl QuantDot for Q4_0Dot {
    fn block_size(&self) -> usize { 32 }
    fn block_bytes(&self) -> usize { 18 }

    fn dot_block(&self, quantized: &[u8], input: &[f32]) -> f32 {
        let scale = f16::from_le_bytes([quantized[0], quantized[1]]).to_f32();
        let mut sum = 0.0f32;
        for i in 0..16 {
            let byte = quantized[2 + i];
            let lo = ((byte & 0x0F) as i8 - 8) as f32;
            let hi = ((byte >> 4) as i8 - 8) as f32;
            sum += lo * input[i * 2];
            sum += hi * input[i * 2 + 1];
        }
        sum * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_dot::QuantDot;

    #[test]
    fn test_q4_0_dot_block_simple() {
        let kernel = Q4_0Dot;
        // scale=1.0, values all 1 (nibble 0x1 means value = (1-8) = -7)
        let scale = half::f16::from_f32(1.0).to_le_bytes();
        let mut quantized = Vec::with_capacity(18);
        quantized.extend_from_slice(&scale);
        quantized.extend(std::iter::repeat(0x11u8).take(16));
        let input = [1.0f32; 32];
        let result = kernel.dot_block(&quantized, &input);
        // each of 32 values: -7 * 1.0 = -7, sum = 32 * -7 = -224
        assert!((result - (-224.0)).abs() < 1e-3);
    }
}
```

**Step 1.1.3: Wire into `crates/ggml-cpu/src/lib.rs`**

Add `pub mod quant_dot;` and `pub use quant_dot::*;` (or just the trait).

**Step 1.1.4: Tests pass**: `cargo test -p ggml-cpu`

### Task 1.2: Add Q8_0 and Q4_1 quantized dot kernels ✅

**Files:**
- ✅ Create: `crates/ggml-cpu/src/quant_dot/q8_0.rs` — **EXISTS**
- ✅ Create: `crates/ggml-cpu/src/quant_dot/q4_1.rs` — **EXISTS**
- ✅ Modify: `crates/ggml-cpu/src/quant_dot.rs` (re-export new modules) — **EXISTS**

**Q8_0** block: [f16 scale (2 bytes)] + [32 bytes i8 values] = 34 bytes → 32 f32 values.
**Q4_1** block: [f16 scale (2 bytes)] + [f16 min (2 bytes)] + [16 bytes nibbles] = 20 bytes → 32 f32 values.

### Task 1.3: Wire quantized dot into Backend trait ✅ (mostly complete)

**Files:**
- ✅ Modify: `crates/ggml/src/backend.rs` — add `mat_vec_quant` method with default (dequantize) impl — **EXISTS (lines 72-81)**
- Modify: `crates/ggml-cpu/src/backend.rs` — implement `mat_vec_quant` dispatching to QuantDot kernels
- ✅ Modify: `crates/llama/src/lib.rs` — add `GgmlType` re-export, `get_quantized_raw()` method on `TensorData` — **EXISTS (line 145)**
- ✅ Modify: `crates/llama/src/context.rs` — use `mat_vec_quant` when tensor is quantized — **EXISTS (mat_vec_weight function, lines 321-350)**
- Modify: `crates/ggml-cpu/benches/cpu_bench.rs` — add quantized matvec benchmark

---

## Phase 2: Broader Backend Trait

### Task 2.1: Add RMSNorm + activations to Backend trait ✅ (already in backend.rs)

**Files:**
- ✅ Modify: `crates/ggml/src/backend.rs` — add `rms_norm`, `silu`, `gelu` methods with default impls — **EXISTS (rms_norm: line 333, silu: line 249, gelu: line 255)**
- Modify: `crates/ggml-cpu/src/backend.rs` — override with SIMD-accelerated versions
- Modify: `crates/ggml-cuda/src/lib.rs` — override with GPU versions
- Modify: `crates/llama/src/context.rs` — replace direct fn calls with `self.backend.*`

**Default impls** in `backend.rs` call existing code from `crates/llama/src/inference.rs` (move the functions to ggml).

**Move** `rms_norm`, `silu`, `gelu` from `llama/src/inference.rs` to `ggml/src/backend.rs` (or a shared utils module).

### Task 2.2: Add softmax to Backend trait

**Files:**
- Modify: `crates/ggml/src/backend.rs`
- Modify: `crates/ggml-cpu/src/lib.rs` (SIMD softmax)
- Modify: `crates/llama/src/attention.rs`

### Task 2.3: CUDA accelerate new backend ops

**Files:**
- Modify: `crates/ggml-cuda/src/lib.rs`
- Create: `crates/ggml-cuda/src/kernels.cu` (or inline PTX)

---

## Phase 3: Tokenizer Replacement

### Task 3.1: Refactor tokenizer into module ✅ (already exists as single file)

**Files:**
- ✅ Modify: `crates/llama/src/tokenizer.rs` → split into `tokenizer/` — **EXISTS as single file (not subdirectory)**
- Create: `crates/llama/src/tokenizer/mod.rs`
- Create: `crates/llama/src/tokenizer/simple.rs` (existing code moved)
- Create: `crates/llama/src/tokenizer/bpe.rs`

Move existing `SimpleTokenizer` to `simple.rs`. Add BPE detection from `tokenizer.ggml.model` GGUF metadata.

### Task 3.2: Add BPE tokenizer implementation

Implement proper pre-tokenization (split on whitespace/punctuation), byte fallback, and BOS/EOS handling.

---

## Phase 4: Server Batching

### Task 4.1: Add batch processing module

**Files:**
- Create: `crates/llama-server/src/batch.rs`
- Modify: `crates/llama-server/src/main.rs`

### Task 4.2: Wire batching into server handlers

---

## Phase 5: AVX2/FMA Code Paths ⚠️ (bdver1 target doesn't support AVX2/FMA)

> **Note:** The project targets `bdver1` (AMD Bulldozer) which only supports AVX (not AVX2) and lacks FMA. These tasks are for future hardware targets or optional feature-gated paths.

### Task 5.1: Add AVX2 + FMA dot product (for future hardware)

**Files:**
- Modify: `crates/ggml-cpu/src/simd.rs` — add `dot_f32_avx2_fma`
- Modify: `crates/ggml-cpu/src/cpu_features.rs` — add `has_avx2()`, `has_fma()`
- Modify: `crates/ggml-cpu/benches/cpu_bench.rs` — add benchmark

### Task 5.2: Add AVX2 + FMA quantized dot kernels (for future hardware)

**Files:**
- Modify: `crates/ggml-cpu/src/quant_dot/q4_0.rs` — FMA variants
- Modify: `crates/ggml-cpu/src/quant_dot/q8_0.rs` — FMA variants

---

## Phase 6: Testing & CI

### Task 6.1: Property-based tests for SIMD

**Files:**
- Modify: `crates/ggml-cpu/Cargo.toml` — add `proptest` dev-dep
- Create: `crates/ggml-cpu/tests/simd_proptest.rs`

### Task 6.2: Add Miri CI job

**Files:**
- Modify: `.github/workflows/ci.yml` — add `cargo miri test` step

### Task 6.3: Tests for ggml core

**Files:**
- Create: `crates/ggml/tests/tensor_test.rs`
- Modify: `crates/ggml/src/tensor.rs` — `#[cfg(test)]` tests

### Task 6.4: Server integration tests

**Files:**
- Create: `crates/llama-server/tests/health_test.rs`
