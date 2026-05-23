# Phase 4 & 5 Implementation Plan

## Overview
Implement Phases 4 (Advanced Features) and 5 (Optimization & Tuning) from MARKET.md, following the Implementation Recommendations for configuration-driven design, benchmarking, and modular extension points.

---

## Batch 1: RoPE with Dynamic Scaling (Phase 4)

**Goal**: Support Linear, NTK-aware, DynamicNTK scaling and partial rotation (Phi-3).

### Files to Modify

| File | Changes |
|---|---|
| `crates/llama/src/lib.rs` | Add `RopeScaleType` enum, `RoPEConfig` struct; add `rope_config` field to `Model`; re-export |
| `crates/llama/src/attention.rs` | Add `apply_rope_with_config()` function; update `multi_head_attention_with_cache` to accept `&RoPEConfig`; keep old `apply_rope` as wrapper |
| `crates/llama/src/model.rs` | Load `rope.freq_scale`, `rope.scaling.type`, `rope.scaling.factor`, `rope.scaling.original_max_position_embeddings`, `rope.dimension_count` |
| `crates/llama/src/context.rs` | Pass `&self.model.rope_config` instead of `rope_theta` |

### Algorithm (apply_rope_with_config)
```
rot_dim = config.partial_dim.unwrap_or(head_dim)
half_dim = rot_dim / 2
eff_theta = match config.scale_type {
    NtkAware => theta * max(scale, 1)^(d/(d-2))
    DynamicNtk => theta * max(scale * actual/max_seq, 1)^(d/(d-2))
    _ => theta
}
eff_pos = match config.scale_type {
    Linear => pos / scale_factor
    DynamicNtk => pos / s
    _ => pos
}
for i in 0..half_dim: rotate pair (i, i+half_dim)
// dims rot_dim..head_dim untouched (partial)
```

### Tests
- `test_rope_with_scaling_linear` — linear scaling matches expectations
- `test_rope_with_ntk` — NTK scaling differs from vanilla
- `test_rope_partial_rotation` — dims beyond partial_dim untouched
- `test_rope_config_default_equals_vanilla` — backward compat

---

## Batch 2: ReLU² Activation (Phase 4)

**Goal**: Add ReLU² for Phi variants using partial RoPE as heuristic.

### Files

| File | Changes |
|---|---|
| `crates/llama/src/inference.rs` | Add `relu_squared(x) = max(0,x)²` |
| `crates/llama/src/context.rs` | Add import + dispatch arm for Phi-3 with partial RoPE |

### Tests
- `test_relu_squared` — [-2,-1,0,1,2,3] → [0,0,0,1,4,9]

---

## Batch 3: QK-Norm for Gemma2 (Phase 4)

**Goal**: Normalize Q and K heads after RoPE, before attention softmax.

### Files

| File | Changes |
|---|---|
| `crates/llama/src/lib.rs` | Add `has_qk_norm: bool` to `Model` |
| `crates/llama/src/model.rs` | Detect `blk.0.attn_q_norm.weight` tensor |
| `crates/llama/src/attention.rs` | Add `qk_norm` + `norm_eps` params to `multi_head_attention_with_cache`; apply per-head RMSNorm |
| `crates/llama/src/context.rs` | Load QK-norm weights per-layer; pass to attention |

### Tests
- `test_qk_norm_applied` — identity weights produce valid output

---

## Batch 4: Sliding Window in Prefill (Phase 4)

**Goal**: Remove dead-code attr, add `window_size`, ready for production wiring.

### Files

| File | Changes |
|---|---|
| `crates/llama/src/attention.rs` | Remove `#[cfg_attr(not(test), ...)]`; add `&RoPEConfig` + `window_size` params; constrain inner loop |

### Tests
- `test_attention_prefill_with_window` — verify windowed output shape

---

## Batch 5: Configuration-Driven Design (Implement. Recommendation #3)

**Goal**: Add runtime strategy selection for KV cache, parallelism, and attention.

### Files

| File | Changes |
|---|---|
| `crates/llama/src/context.rs` | Add `parallel_min_rows`, `cache_strategy` to `ModelConfig` |
| `crates/common/src/lib.rs` | Add `--cache-strategy`, `--parallel-min-rows` CLI args; add `to_model_config()` |
| `crates/llama-cli/src/main.rs` | Use `CommonArgs::to_model_config()`; fix `n_batch` |
| `crates/llama-server/src/main.rs` | Same fixes |
| `crates/llama/src/kv_cache.rs` | Add `CacheStrategy` enum (`Full`, `Prefix`), `truncate()` |

### Key Types
```rust
pub enum CacheStrategy { Full, Prefix }

pub struct ModelConfig {
    pub n_threads: usize,
    pub use_cuda: bool,
    pub n_ctx: usize,
    pub n_batch: usize,       // now independently configurable
    pub parallel_min_rows: usize,  // default: 128
    pub cache_strategy: CacheStrategy,  // default: Full
}
```

---

## Batch 6: KV Cache Optimizations (Phase 5)

### 6a. O(1) KV Cache Reset
- File: `crates/llama/src/kv_cache.rs`
- Remove `.fill(0.0)` from `reset()` — just set `cur_len = 0`

### 6b. Batch KV Cache Push
- File: `crates/llama/src/kv_cache.rs`
- Add `push_batch(k, v, n_tokens)` method

### 6c. Prefix Caching
- File: `crates/llama/src/kv_cache.rs`
- Add `truncate(new_len)` to shrink cache length

---

## Batch 7: Parallel Matmul Threshold (Phase 5)

**Goal**: Skip thread spawning for small tensors where overhead > benefit.

### Files

| File | Changes |
|---|---|
| `crates/ggml-cpu/src/backend.rs` | Add `parallel_min_rows: usize` field to `CpuBackend` |
| `crates/ggml-cpu/src/matmul.rs` | Gate parallel dispatch behind `m >= MIN_PARALLEL_ROWS` (128) |

---

## Batch 8: New Benchmarks (Implement. Recommendation #4)

| File | What |
|---|---|
| `crates/llama/benches/kv_cache_bench.rs` | push_single, push_batch, reset, prefix_find |
| `crates/llama/benches/attention_bench.rs` | flash_attention scaling (seq 64..4096), sliding window |
| `crates/ggml-cpu/benches/cpu_bench.rs` | Extend: parallel threshold scan for small matmuls |
| `crates/llama/Cargo.toml` | Add `[[bench]]` targets |

---

## Execution Order

```
Batch 1 (RoPE) ─┬─> Batch 2 (ReLU²) ──> Batch 4 (Prefill)
                 │
                 └─> Batch 3 (QK-Norm)
                 
Batch 5 (Config) ──> Batch 6 (KV Cache) ──> Batch 8 (Benchmarks)
                 
Batch 7 (Parallel) ──> Batch 8 (Benchmarks)
```

Batches 1–4 (Phase 4) can be merged into one commit. Batches 5–7 (Phase 5) into another. Batch 8 into a third.

## Verification

```bash
cargo build --workspace           # zero warnings
cargo fmt --all -- --check        # clean
cargo clippy --workspace -- -D warnings  # clean
cargo test -p gguf -p ggml-cpu -p common -p ggml -p llama  # all pass
cargo run -p llama-cli -- -m model.gguf -p "Hello" -n 8   # produces output
```
