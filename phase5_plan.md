# Phase 5: Optimization & Tuning — Implementation Plan

## Overview

Four optimization areas for the llama-rs inference engine, ordered by impact:

1. **KV Cache** — O(1) reset, batch push, prefix caching
2. **Parallelization** — minimum-size thresholds, wire CpuBackend into inference
3. **Configuration** — cache strategy, parallelism tuning, batch size
4. **Benchmarks** — measure all optimizations

---

## Step 1: O(1) KV Cache Reset

**Goal**: Replace `O(max_seq * n_head_kv * head_dim)` reset with `O(1)`.

**Rationale**: `reset()` is called between generations. Filling both arrays with 0.0 writes 2×max_seq×n_head_kv×head_dim floats. Since `cur_len` gates reads, the values between `cur_len` and `max_seq` are never read — overwriting them is pure waste. For a 32-layer 4096-seq model with n_head_kv=8 and head_dim=128, that's ~268 MB of zeroing per reset.

### Files to modify

**`crates/llama/src/kv_cache.rs`**

- Change `reset()` to just set `cur_len = 0`. Remove the `.fill(0.0)` calls.
- The allocated `Vec`s stay at capacity (no reallocation needed).

```rust
pub fn reset(&mut self) {
    self.cur_len = 0;
    // keys/values NOT zeroed — old data beyond cur_len is never read
}
```

### Test strategy

- Add unit test in `kv_cache.rs`:
  - Push data, reset, verify `cur_len == 0`
  - Push again, verify old data is overwritten correctly
  - Verify that `get()` panics correctly on positions >= cur_len after reset

---

## Step 2: Batch KV Cache Push

**Goal**: Add `push_batch()` to insert multiple tokens in one call (used during prefill).

**Rationale**: Currently `multi_head_attention_with_cache` calls `push()` in a loop, once per token. A single `push_batch()` eliminates the per-token bounds checks and copy overhead.

### Files to modify

**`crates/llama/src/kv_cache.rs`**

Add:

```rust
/// Append multiple tokens' keys and values at once.
/// `k` and `v` must each be of length `n_tokens * n_head_kv * head_dim`.
pub fn push_batch(&mut self, k: &[f32], v: &[f32], n_tokens: usize) {
    let token_len = self.n_head_kv * self.head_dim;
    let total_len = n_tokens * token_len;
    assert_eq!(k.len(), total_len);
    assert_eq!(v.len(), total_len);
    assert!(self.cur_len + n_tokens <= self.max_seq, "KV cache overflow");
    let offset = self.cur_len * token_len;
    self.keys[offset..offset + total_len].copy_from_slice(k);
    self.values[offset..offset + total_len].copy_from_slice(v);
    self.cur_len += n_tokens;
}
```

**`crates/llama/src/attention.rs`**

In `multi_head_attention_with_cache`, replace the per-token push loop:

```rust
// Before: for pos in 0..seq_len { kv_cache.push(...) }
// After:
kv_cache.push_batch(k, v, seq_len);
```

### Test strategy

- Unit test in `kv_cache.rs`:
  - `push_batch` 3 tokens, verify `cur_len == 3` and data accessible via `get()`
  - Verify overflow panic works correctly for batch push
  - Benchmark push_batch vs loop-of-push (in Step 8)

---

## Step 3: KV Cache Prefix Caching

**Goal**: Avoid recomputing KV for shared prefix across prompts.

**Rationale**: In chat applications, each new prompt shares a long prefix with previous prompts (system prompt, conversation history). Without prefix caching, every `generate()` call recomputes KV for the entire prompt.

### New types

**`crates/llama/src/kv_cache.rs`**

```rust
/// Strategy for managing the KV cache across inference calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Clear cache on every generation (current behavior).
    Full,
    /// Keep cache across calls. Incoming tokens are compared to cached prefix;
    /// matching prefix is reused, remaining tokens re-computed.
    Prefix,
}
```

Add to `KvCache`:

```rust
impl KvCache {
    /// Truncate cache to `new_len` tokens (used for prefix reuse).
    /// If `new_len > cur_len`, this is a no-op.
    pub fn truncate(&mut self, new_len: usize) {
        self.cur_len = self.cur_len.min(new_len);
    }
}
```

Add to `KvCacheManager`:

```rust
impl KvCacheManager {
    /// Truncate all layer caches to new_len.
    pub fn truncate_all(&mut self, new_len: usize) {
        for cache in &mut self.caches {
            cache.truncate(new_len);
        }
    }
    
    /// Get current sequence length (min across layers).
    pub fn cur_len(&self) -> usize {
        self.caches.first().map_or(0, |c| c.cur_len)
    }
}
```

### Integration in context.rs

**`crates/llama/src/context.rs`**

Modify `InferenceContext::generate()` to support prefix caching:

1. Get the prompt tokens
2. If `CacheStrategy::Prefix`:
   - Compare new prompt tokens against cached tokens (stored in a new field `cached_tokens: Vec<usize>` on `InferenceContext`)
   - Find the longest common prefix
   - Truncate KV cache to prefix length
   - Run prefill for the remaining non-cached tokens
   - Update `cached_tokens` with the full new prompt
3. If `CacheStrategy::Full`: reset KV cache (existing behavior)

New field on `InferenceContext`:

```rust
pub struct InferenceContext {
    pub model: Arc<Model>,
    pub config: ModelConfig,
    pub tokenizer: crate::SimpleTokenizer,
    pub sampling: SamplingConfig,
    /// Tokens currently in the KV cache (for prefix caching).
    cached_tokens: Vec<usize>,
}
```

### Modifications to forward_pass

The current `forward_pass` always processes a single token. For prefix cache warming during prefill, we need a `prefill` method that processes multiple tokens without generating output.

**`crates/llama/src/context.rs`** — Add:

```rust
/// Run forward pass for a batch of tokens (prefill phase).
/// Processes all tokens, stores KV in cache, returns final logits.
pub fn prefill(&self, tokens: &[usize]) -> anyhow::Result<Vec<f32>> {
    // ... iterate tokens, calling forward_pass logic for each
    // Returns the logits for the LAST token only (for next-token prediction)
}
```

### Add `cached_tokens` to InferenceContext

```rust
pub struct InferenceContext {
    pub model: Arc<Model>,
    pub config: ModelConfig,
    pub tokenizer: crate::SimpleTokenizer,
    pub sampling: SamplingConfig,
    cached_tokens: Vec<usize>,
}
```

Modify `InferenceContext::new()` to initialize `cached_tokens: Vec::new()`.

### Test strategy

- Unit test in `kv_cache.rs`:
  - Push 10 tokens, truncate to 5, verify cur_len == 5, get(4) works, get(5) panics
- Integration test in `tests/`:
  - Create model, generate with prefix caching
  - Make two calls with overlapping prompt prefixes
  - Verify second call produces same output but faster (via profiling)

---

## Step 4: Parallel Matmul Threshold

**Goal**: Skip thread spawning for small matmuls where overhead exceeds benefit.

**Rationale**: `matmul_f32` always splits rows across threads via `std::thread::scope`. For small matrices (e.g., QKV projections in tiny models or single-token inference), thread spawning overhead can exceed the compute time.

### Files to modify

**`crates/ggml-cpu/src/matmul.rs`**

Add a minimum rows threshold:

```rust
/// Minimum rows of A before parallel dispatch is worthwhile.
/// Below this, run single-threaded.
const MIN_PARALLEL_ROWS: usize = 128;
```

Modify `matmul_f32`:

```rust
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, n_threads: usize) {
    // ... existing assertions ...
    
    let n_threads = if n_threads == 0 { /* existing */ } else { n_threads };
    
    // Don't parallelize tiny matmuls
    if n_threads <= 1 || m < MIN_PARALLEL_ROWS {
        matmul_f32_block(a, b, c, n, k, 0, m, 0, n);
        return;
    }
    
    // ... existing parallel dispatch ...
}
```

Make `MIN_PARALLEL_ROWS` configurable:

**`crates/ggml-cpu/src/backend.rs`**

Add `parallel_min_rows` field to `CpuBackend`:

```rust
pub struct CpuBackend {
    n_threads: usize,
    /// Minimum rows before parallel dispatch. 0 = use default.
    parallel_min_rows: usize,
}

impl CpuBackend {
    pub fn new(n_threads: usize) -> Self { /* existing */ }
    
    pub fn with_parallel_threshold(n_threads: usize, parallel_min_rows: usize) -> Self {
        Self {
            n_threads: /* existing logic */,
            parallel_min_rows,
        }
    }
}
```

Pass `parallel_min_rows` through to `matmul_f32`:

```rust
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, n_threads: usize, min_parallel_rows: usize) {
```

### Test strategy

- Unit test in `matmul.rs`:
  - Verify `matmul_f32` with `m=16`, `n_threads=4`, `min_parallel_rows=128` runs single-threaded
  - Verify result matches multi-threaded for same computation
- Benchmark: compare parallel vs sequential for various sizes to find optimal threshold

---

## Step 5: Wire CpuBackend into Inference Pipeline

**Goal**: Replace direct `rayon` usage in `inference.rs` with `CpuBackend` for unified thread control.

**Rationale**: Currently `mat_vec`, `mul_vec`, `add_vec` use rayon directly while `CpuBackend` uses `std::thread::scope`. This creates two thread pools that can compete. Also, the inference pipeline in `context.rs` calls `mat_vec()` from `inference.rs` instead of using `CpuBackend::matmul()`.

### Files to modify

**`crates/ggml-cpu/src/backend.rs`**

Add parallelism helper methods to `CpuBackend`:

```rust
impl CpuBackend {
    /// Run a parallel operation across rows, dispatching if above threshold.
    pub fn parallel_for<T, F>(&self, items: &[T], f: F)
    where
        T: Sync,
        F: Fn(&T) + Sync,
    {
        if items.len() < self.parallel_min_rows || self.n_threads <= 1 {
            items.iter().for_each(f);
        } else {
            std::thread::scope(|s| {
                let chunk_size = items.len().div_ceil(self.n_threads);
                for chunk in items.chunks(chunk_size) {
                    s.spawn(|| chunk.iter().for_each(&f));
                }
            });
        }
    }
}
```

**`crates/llama/src/inference.rs`**

Modify `mat_vec`, `mul_vec`, `add_vec` to accept a `parallel_min_rows` parameter instead of using hardcoded constants:

```rust
pub fn mat_vec(mat: &[f32], rows: usize, cols: usize, vec: &[f32], 
               parallel_min_rows: usize) -> Vec<f32> {
    // Use parallel_min_rows instead of hardcoded 64
    if rows < parallel_min_rows {
        // sequential
    } else {
        // parallel via rayon
    }
}
```

**`crates/llama/src/context.rs`**

Update calls to `mat_vec`, `mul_vec`, `add_vec` to pass the configured threshold from `ModelConfig`.

### Add fields to ModelConfig

```rust
pub struct ModelConfig {
    pub n_threads: usize,
    pub use_cuda: bool,
    pub n_ctx: usize,
    pub n_batch: usize,
    /// Minimum rows for parallel mat-vec dispatch (0 = default 64).
    pub parallel_min_rows: usize,
    /// KV cache strategy.
    pub cache_strategy: CacheStrategy,
}
```

### Test strategy

- All existing tests pass (regression)
- Unit test for `parallel_for` with threshold
- Benchmark: verify same performance as before for existing config

---

## Step 6: Configuration-Driven Strategy Selection

**Goal**: Make KV cache strategy, parallelism tuning, and batch size configurable from CLI.

### Files to modify

**`crates/llama/src/context.rs`**

Update `ModelConfig`:

```rust
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_threads: usize,
    pub use_cuda: bool,
    pub n_ctx: usize,
    pub n_batch: usize,
    /// Minimum rows for parallel mat-vec dispatch (default: 64).
    pub parallel_min_rows: usize,
    /// KV cache strategy (default: Full).
    pub cache_strategy: CacheStrategy,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_threads: 4,
            use_cuda: false,
            n_ctx: 2048,
            n_batch: 512,
            parallel_min_rows: 64,
            cache_strategy: CacheStrategy::Full,
        }
    }
}
```

**`crates/common/src/lib.rs`** — Extend `CommonArgs`:

```rust
pub mod args {
    #[derive(Parser, Debug)]
    pub struct CommonArgs {
        pub model: String,
        pub threads: usize,
        pub ctx_size: usize,
        pub batch_size: usize,
        pub use_cuda: bool,
        
        /// KV cache strategy: "full" or "prefix" (default: full).
        #[arg(long, default_value = "full")]
        pub cache_strategy: String,
        
        /// Minimum rows for parallel dispatch (default: 64).
        #[arg(long, default_value_t = 64)]
        pub parallel_min_rows: usize,
    }
    
    impl CommonArgs {
        pub fn to_model_config(&self) -> ModelConfig {
            ModelConfig {
                n_threads: /* existing logic */,
                use_cuda: self.use_cuda,
                n_ctx: self.ctx_size,
                n_batch: self.batch_size,
                parallel_min_rows: self.parallel_min_rows,
                cache_strategy: match self.cache_strategy.as_str() {
                    "prefix" => CacheStrategy::Prefix,
                    _ => CacheStrategy::Full,
                },
            }
        }
    }
}
```

**`crates/llama-cli/src/main.rs`** and **`crates/llama-server/src/main.rs`**:

- Use `CommonArgs::to_model_config()` instead of manually constructing `ModelConfig`
- Fix `n_batch` to use the `batch_size` argument instead of `ctx_size`

### Test strategy

- Unit test for `ModelConfig` defaults
- Test argument parsing with new flags
- Verify CLI can start with `--cache-strategy prefix`

---

## Step 7: Prefill Batching

**Goal**: Process prompt tokens in batches during prefill phase for faster prompt ingestion.

**Rationale**: Currently `generate()` calls `forward_pass(last_token)` once per token for both prefill and decode. For the initial prompt, we can process all tokens together, storing KV for each.

### Files to modify

**`crates/llama/src/context.rs`**

Modify `generate()` to use batch prefill:

```rust
pub fn generate(&self, prompt: &str, n_predict: usize) -> anyhow::Result<Vec<usize>> {
    let mut toks = self.encode(prompt);
    
    if toks.len() > self.config.n_ctx {
        toks.truncate(self.config.n_ctx);
    }
    
    // PREFILL PHASE: Process all prompt tokens without generating output
    // Store KV cache for all prompt tokens in one pass
    if toks.len() > 1 {
        // Use config.n_batch to control prefill batch size
        for chunk in toks.chunks(self.config.n_batch) {
            // Process each batch of tokens through the model
            // Only keep KV cache, discard intermediate logits
            // Keep final logits for next-token prediction
            self.batch_forward(chunk)?;
        }
    }
    
    // DECODE PHASE: Generate one token at a time
    for _i in 0..n_predict {
        let last_token = *toks.last().unwrap_or(&0);
        match self.forward_pass(last_token) {
            Ok(logits) => {
                let next_token = sample_logits(&logits, &self.sampling);
                toks.push(next_token);
                if next_token == self.model.eos_token_id {
                    break;
                }
            }
            Err(_) => { toks.push(0); }
        }
    }
    
    Ok(toks)
}
```

Add `batch_forward` method:

```rust
/// Process a batch of tokens in a single forward pass.
/// Stores KV cache for all tokens.
/// Returns the logits for the last token.
fn batch_forward(&self, tokens: &[usize]) -> anyhow::Result<Vec<f32>> {
    // For each token in the batch, run the forward pass,
    // accumulating KV cache entries.
    // Only return the logits of the final token.
    let mut final_logits = None;
    for &token in tokens {
        final_logits = Some(self.forward_pass(token)?);
    }
    final_logits.ok_or_else(|| anyhow::anyhow!("empty batch"))
}
```

### Important constraint

`n_batch` should be independently configurable (not tied to `n_ctx`). Both binaries currently set `n_batch = args.ctx_size` which defeats the purpose. The `CommonArgs` already has `batch_size` — we just need to use it.

### Test strategy

- Existing `test_forward_pass_produces_logits` tests pass (regression)
- New test: generate with batch size < prompt length, verify output matches single-token generation
- Profile: verify prefill is faster with batching

---

## Step 8: New Benchmarks

### Files to create/modify

**`crates/llama/benches/kv_cache_bench.rs`** (new)

Benchmarks:
- `kv_cache_push_single` — push 1 token at a time, varying seq_len
- `kv_cache_push_batch` — push N tokens at once via `push_batch`
- `kv_cache_reset_old` vs `kv_cache_reset_new` — compare old zeroing vs O(1)
- `kv_cache_prefix_find` — measure longest-common-prefix computation

```rust
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llama::kv_cache::KvCache;

fn kv_cache_push_benchmark(c: &mut Criterion) {
    let head_dim = 128;
    let n_head_kv = 8;
    let max_seq = 4096;
    let mut cache = KvCache::new(max_seq, n_head_kv, head_dim);
    let token_len = n_head_kv * head_dim;
    let k = vec![0.1; token_len];
    let v = vec![0.2; token_len];
    
    c.bench_function("kv_cache_push_single", |b| {
        b.iter(|| {
            cache.reset();
            for _ in 0..1024 {
                cache.push(&k, &v);
            }
            black_box(&cache);
        })
    });
}

// ... more benchmarks ...
```

**`crates/llama/benches/attention_bench.rs`** (new)

Benchmarks:
- `flash_attention_seq_64` through `flash_attention_seq_4096` — varying seq_len
- `flash_attention_vs_legacy` — compare flash with materialized attention
- `attention_with_sliding_window` — measure sliding window impact

```rust
fn flash_attention_benchmark(c: &mut Criterion) {
    let head_dim = 128;
    let n_head_kv = 8;
    
    for &seq_len in &[64, 256, 1024, 4096] {
        let q = vec![0.1; head_dim];
        let keys = vec![0.1; seq_len * n_head_kv * head_dim];
        let values = vec![0.1; seq_len * n_head_kv * head_dim];
        
        c.bench_function(&format!("flash_attn_seq_{}", seq_len), |b| {
            b.iter(|| {
                flash_attention_head(&q, &keys, &values, seq_len, head_dim, n_head_kv, 0, None)
            })
        });
    }
}
```

**`crates/llama/Cargo.toml`** — Add new bench targets:

```toml
[[bench]]
name = "kv_cache_bench"
harness = false

[[bench]]
name = "attention_bench"
harness = false
```

**`crates/ggml-cpu/benches/cpu_bench.rs`** — Extend existing:

Add parallel threshold benchmarks:
- `matmul_16x128_parallel_vs_sequential` — show overhead at small sizes
- `matmul_512x512_threshold_scan` — find optimal threshold

### Test strategy

- All benchmarks compile and run without test model dependency
- `cargo bench` succeeds for all new targets
- Benchmark results are published to `target/criterion/` for comparison

---

## Summary: File Change Map

| File | Changes |
|------|---------|
| `crates/llama/src/kv_cache.rs` | O(1) reset, push_batch, truncate, CacheStrategy enum, PrefixCache |
| `crates/llama/src/attention.rs` | Use push_batch instead of push loop |
| `crates/llama/src/context.rs` | ModelConfig fields, prefill batching, prefix caching, cached_tokens |
| `crates/llama/src/lib.rs` | Re-export CacheStrategy |
| `crates/llama/src/inference.rs` | Parallel thresholds via parameter |
| `crates/ggml-cpu/src/matmul.rs` | MIN_PARALLEL_ROWS threshold |
| `crates/ggml-cpu/src/backend.rs` | parallel_min_rows, parallel_for helper |
| `crates/common/src/lib.rs` | CommonArgs: cache_strategy, parallel_min_rows |
| `crates/llama-cli/src/main.rs` | Use CommonArgs::to_model_config, fix n_batch |
| `crates/llama-server/src/main.rs` | Use CommonArgs::to_model_config, fix n_batch |
| `crates/llama/benches/kv_cache_bench.rs` | New: KV cache benchmarks |
| `crates/llama/benches/attention_bench.rs` | New: attention benchmarks |
| `crates/ggml-cpu/benches/cpu_bench.rs` | Extend: parallel threshold benchmarks |
| `crates/llama/Cargo.toml` | Add new bench targets |

---

## Execution Order

| Step | Description | Complexity | Risk |
|------|-------------|------------|------|
| 1 | O(1) KV cache reset | Low | Very low — pure optimization, no behavior change |
| 2 | Batch KV cache push | Low | Low — extends API, existing push unchanged |
| 3 | KV cache prefix caching | Medium | Medium — adds new code path, needs careful testing |
| 4 | Parallel matmul threshold | Low | Very low — min row check, no behavior change above threshold |
| 5 | Wire CpuBackend into inference | Medium | Medium — changes call sites across inference pipeline |
| 6 | Configuration-driven strategy | Low | Low — extended config struct, updated CLI args |
| 7 | Prefill batching | Medium | Medium — new batch_forward method, changes generate() flow |
| 8 | New benchmarks | Low | Low — new bench files, no production code changes |

---

## Verification

1. **Build**: `cargo build --workspace` must succeed with zero warnings
2. **Lint**: `cargo fmt --all -- --check` and `cargo clippy --workspace -- -D warnings` must pass
3. **Tests**: `cargo test --workspace` must pass (including new unit tests)
4. **Benchmarks**: `cargo bench --workspace` must compile and run
5. **CLI smoke test**: `cargo run -p llama-cli -- -m model.gguf -p "Hello" -n 8` produces output
6. **Backward compatibility**: All existing tests pass without changes — no test modifications needed for Steps 1, 2, 4, or 8
