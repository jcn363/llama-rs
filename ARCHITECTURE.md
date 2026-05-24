# llama-rs Architecture

## Overview

A Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine for LLaMA-family large language models. Targets **AMD Opteron 3280** (bdver1, 8 cores, no FMA) and **NVIDIA GTX 1050** (2GB VRAM, Compute 6.1). Uses GGUF v3 as the model format.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Rust, edition 2024, MSRV 1.85 |
| Parallelism | `rayon` (data-parallel), `std::thread::scope` (task-parallel), `std::sync::Arc` |
| SIMD | AVX (8-wide) → SSE4.2 (4-wide) → scalar fallback (no FMA, no AVX2) |
| GPU | `cudarc` bindings, cuBLAS matmul (enabled by default) |
| Hardware Abstraction | `Backend` trait + compile-time plugin registry (see [Plugin System](#9-plugin-system)) |
| Serialization | `byteorder` (LE binary), `memmap2` (memory-mapped I/O) |
| Async | `tokio` + `axum` + `futures` (server binary only) |
| CLI | `clap` v4 derive |
| Errors | `thiserror` (libs), `anyhow` (binaries) |
| Benchmarking | `criterion` with HTML reports |
| Testing | `#[cfg(test)] mod tests` + integration tests in `tests/` |

## Directory Structure

```
llama-rs/
├── Cargo.toml                  # Workspace root (9 members)
├── rustfmt.toml                # max_width=100, tab_spaces=4, reorder_imports
├── deny.toml                   # License policy (MIT, Apache-2.0, Unlicense)
├── .cargo/config.toml          # --target-cpu=bdver1
├── .github/workflows/ci.yml    # format → clippy → test → deny → doc
├── crates/
│   ├── gguf/                   # GGUF v3 file parser (no deps on other internal crates)
│   ├── ggml/                   # Core tensor types + computation graph + Backend trait (depends on nothing)
│   ├── ggml-cpu/               # CPU backend: implements Backend trait, SIMD matmul (depends on ggml)
│   ├── ggml-cuda/              # CUDA backend: implements Backend trait, cuBLAS (depends on ggml, requires CUDA toolkit)
│   ├── llama/                  # Inference engine: factory + dispatch via Arc<dyn Backend> (depends on gguf, ggml, ggml-cpu, ggml-cuda)
│   ├── common/                 # Shared utils: args, sampling config, chat templates
│   ├── llama-cli/              # CLI binary with --backend flag (depends on llama, common)
│   └── llama-server/           # HTTP server with --backend flag (depends on llama, common, axum)
├── test-models/                 # Test GGUF files (gitignored, downloaded separately)
├── docs/                        # Additional documentation
└── media/                       # Screenshots, diagrams
```

## Crate Dependency Graph

```
                    ┌──────────────────────┐
                    │        ggml          │  (Tensor, DType, Graph, Backend trait)
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ ggml-cpu │ │ggml-cuda │ │   gguf   │  (GGUF v3 parser)
        │ (Backend)│ │ (Backend)│ │          │
        └─────┬────┘ └─────┬────┘ └─────┬────┘
              │            │            │
              └──────┬─────┘            │
                     ▼                  ▼
              ┌─────────────────────────────────┐
              │     llama (create_backend       │  (Inference engine + backend factory)
              │     → Arc<dyn Backend>)         │
              └──────────────┬──────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  common  │  │llama-cli │  │llama-    │
        │          │  │ (--backend│  │server    │
        │          │  │  flag)   │  │(--backend │
        └──────────┘  └──────────┘  │  flag)   │
                                    └──────────┘
```

## Core Components

### 1. `gguf` — GGUF v3 Parser (`crates/gguf/src/`)

Parses the GGUF file format (v3). Minimal external dependencies: `memmap2`, `rayon`, `half`, `thiserror`.

| File | Responsibility |
|------|---------------|
| `lib.rs` | `GgufReader` struct definition, top-level re-exports |
| `loader.rs` | `GgufReader::from_file()`, `from_mmap()` — parses header, KV pairs, tensor info |
| `cursor.rs` | `CursorReader` — little-endian binary reader over `&[u8]` |
| `types.rs` | `GgufType` (13 value types), `GgmlType` (42 tensor types) |
| `value.rs` | `GgufValue` enum — typed metadata value representation |
| `tensor.rs` | `TensorInfo` (metadata) + `MmapTensor` (lazy mmap reference) |
| `reader.rs` | `GgufReader` method impls: `get_kv()`, `find_tensor()`, array accessors |
| `dequant.rs` | Dequantization functions: Q4_0..Q6_K, K-quants (Q2_K..Q6_K), Ollama quantizations (Q8_K, Q1_0) |
| `errors.rs` | `GgufError` enum, `GgufResult<T>` alias |
| `constants.rs` | `GGUF_MAGIC`, `GGUF_VERSION`, `GGUF_DEFAULT_ALIGNMENT` |

**Data flow:**
1. `from_file()` memory-maps the file → `CursorReader` parses header + KV pairs + tensor info → stores aligned data offset
2. Tensor data is accessed lazily via `MmapTensor::dequantize()`
3. `dequantize()` dispatches to SIMD-parallelized per-dtype dequant functions

### 2. `ggml` — Core Tensor Library (`crates/ggml/src/`)

Foundation types used by all compute backends. Also defines the [`Backend`] trait that hardware backends implement.

| File | Responsibility |
|------|---------------|
| `lib.rs` | Re-exports `Tensor`, `DType`, `Graph`; exposes `pub mod backend` |
| `backend.rs` | `Backend` trait + `BackendInfo` struct + CPU default fallback implementations |
| `tensor.rs` | `Tensor` struct — multi-dimensional array with `Arc<[u8]>` data |
| `dtype.rs` | `DType` enum — F32, F16, I8, U8, I32, I64 |
| `graph.rs` | `Graph` — simple DAG of tensor operations |

[`Backend`]: https://docs.rs/ggml/latest/ggml/backend/trait.Backend.html

The `Backend` trait is object-safe (`Send + Sync`) and defines four core operations:

| Method | Signature | Description |
|--------|-----------|-------------|
| `info()` | `fn info(&self) -> BackendInfo` | Returns name, availability, memory, parallelism |
| `mat_vec()` | `fn mat_vec(&self, weight, rows, cols, input) -> Vec<f32>` | Matrix-vector product |
| `add()` | `fn add(&self, a, b) -> Vec<f32>` | Element-wise addition |
| `mul()` | `fn mul(&self, a, b) -> Vec<f32>` | Element-wise multiplication |

The trait provides default CPU fallbacks for all operations, so implementors only
need to override the operations they accelerate.

### 3. `ggml-cpu` — CPU Backend (`crates/ggml-cpu/src/`)

Optimized for AMD Opteron 3280 (bdver1: SSE4.2 + AVX, no FMA/AVX2). Implements [`Backend`] trait.

| File | Responsibility |
|------|---------------|
| `lib.rs` | Public API: `dot_f32`, `matmul_f32`, `CpuBackend`, feature detection |
| `backend.rs` | `impl Backend for CpuBackend` — SIMD-accelerated `mat_vec` via `dot_f32` + `std::thread::scope` |
| `matmul.rs` | Block-tiled `matmul_f32()` (16x16 tiles, parallel via `std::thread::scope`) |
| `simd.rs` | `dot_f32()`: AVX 8-wide → SSE4.2 4-wide → scalar fallback |
| `cpu_features.rs` | Runtime detection: `has_sse4_2()`, `has_avx()`, `has_aes()`, `has_popcnt()` |

**Key design decisions:**
- Threadpool: uses `std::thread::scope` (not rayon) for raw pointer access to output matrix
- SIMD: 4 accumulators × SIMD width per iteration for instruction-level parallelism
- Small-matrix threshold: parallel below 64 rows is slower than sequential

### 4. `ggml-cuda` — CUDA Backend (`crates/ggml-cuda/src/`)

Enabled by default (requires NVIDIA GPU + CUDA toolkit). Targets GTX 1050 (sm_61, 2GB VRAM). Implements [`Backend`] trait.

| Responsibility | Details |
|---------------|---------|
| `CudaBackend` | Implements `Backend` trait; `mat_vec` uploads → cuBLAS gemm → downloads result |
| `DeviceTensor` | GPU-side tensor with `copy_to_device()` / `to_host()` |
| Error types | `CudaError::NotAvailable`, `OutOfMemory`, `RuntimeError` |
| Stub mode | When `cuda` feature disabled, returns `available=false` but doesn't crash |

The CUDA backend delegates cheap operations (`add`, `mul`) to the CPU fallback
and only accelerates `mat_vec` via GPU. If cuBLAS fails, it transparently falls
back to the default CPU implementation.

### 5. `llama` — Inference Engine (`crates/llama/src/`)

The core crate. Implements the full transformer forward pass.

| File | Lines | Responsibility |
|------|-------|---------------|
| `lib.rs` | 219 | `Model` struct definition, `TensorData` (lazy dequant), `InternedStrings`, `RoPEConfig`, `CacheStrategy`, public re-exports (incl. `Backend`, `BackendType`, `create_backend`) |
| `backend.rs` | 87 | `BackendType` enum (`Auto`/`Cpu`/`Cuda`), `create_backend()` factory — linear-priority chain: CUDA → CPU |
| `model.rs` | 335 | `Model::load_from_gguf()` — parses GGUF metadata, builds tensor map, RoPE scaling + QK-norm detection |
| `context.rs` | 429 | `InferenceContext` — ties model + tokenizer + sampling + `Arc<dyn Backend>`; `generate()`, `forward_pass()` dispatches to backend |
| `inference.rs` | 414 | Math ops: `rms_norm`, `silu`, `gelu`, `relu_squared`, `sample_logits`, top-k/p |
| `attention.rs` | 588 | `multi_head_attention_with_cache()`, `flash_attention_head()`, `apply_rope()`, `apply_rope_with_config()`, `multi_head_attention_prefill()` |
| `kv_cache.rs` | 198 | `KvCache` (per-layer) + `KvCacheManager` (multi-layer), `CacheStrategy`, `push_batch()`, `truncate()`, O(1) `reset()` |
| `tokenizer.rs` | 318 | `SimpleTokenizer` — greedy longest-match tokenizer from GGUF vocab |
| `profile.rs` | 63 | `ProfileResult` — per-layer timing data |

**Forward pass flow (`InferenceContext::forward_pass`):**
1. Embed lookup: `token_embd.weight[token_id]`
2. For each layer:
   a. RMSNorm → Q/K/V projections (via `self.backend.mat_vec()`) → QK-norm (if Gemma2) → RoPE with configurable scaling → KV cache store
   b. Flash attention (online softmax, O(N) memory; sliding window for Mistral)
   c. Attention output projection (via `self.backend.mat_vec()`) → residual add (via `self.backend.add()`)
   d. RMSNorm → SiLU-gated FFN (GELU for Gemma, ReLU² for Phi-3) → residual add
3. Final RMSNorm → output projection → logits

All tensor math operations (matrix-vector product, add, mul) are dispatched
through the `Arc<dyn Backend>` stored in `InferenceContext`, selected at startup
via the `--backend` CLI flag or `BackendType` configuration.

**Inference modes:**
- `generate()` — standard greedy/sampling generation
- `generate_with_profile()` — per-layer timing for benchmarking

**Cache strategies:**
- `Full` — standard KV cache (all tokens retained)
- `Prefix` — supports `truncate()` for repeated prompt reuse (prefix caching)
- O(1) `reset()` — zero-cost cache clear (no memory fill)

### 6. `common` — Shared Utilities (`crates/common/src/`)

| Module | Content |
|--------|---------|
| `args` | `CommonArgs` — clap args for model path, threads, ctx size, CUDA flag |
| `sampling` | `SamplingConfig` — temperature, top-k, top-p, repeat_penalty |

### 7. `llama-cli` — CLI Binary (`crates/llama-cli/src/main.rs`)

Single-file binary. Interactive mode (reads from stdin) or single-prompt mode (`-p`).
Supports `--backend auto|cpu|cuda` flag to select the hardware backend at startup.

### 8. `llama-server` — HTTP Server (`crates/llama-server/src/main.rs`)

Axum-based HTTP server with `--backend auto|cpu|cuda` CLI flag and two endpoints:
- `GET /health` → `{"status": "ok"}`
- `POST /completion` → JSON body with `prompt`, `max_tokens`, `stream`, `temperature`
  - Non-streaming: returns `CompletionResponse { content, model }`
  - Streaming: SSE with `StreamChunk { content, stop }` events

## Data Flow

```
GGUF file on disk
    │
    ▼
GgufReader::from_file()  ───►  mmap file, parse header/KV/tensors
    │
    ▼
Model::load_from_gguf()  ───►  Build tensor map (parallel dequant),
    │                           populate hyperparameters
    ▼
InferenceContext::new()  ───►  Create tokenizer, config
    │
    ▼
encode(prompt)           ───►  Tokenize input text → Vec<usize>
    │
    ▼
forward_pass(token_id)   ───►  Embed → N transformer layers → logits
    │                           (each layer: attn + FFN w/ residuals)
    ▼
sample_logits(logits)    ───►  temperature → top-k → top-p → categorical
    │
    ▼
decode(tokens)           ───►  Token IDs → text string
```

## 9. Plugin System — Hardware Backends

The hardware backend system decouples tensor math from the inference pipeline,
allowing CPU and GPU implementations to coexist and be selected at runtime.

### Architecture

```
                    ┌──────────────────────┐
                    │   ggml::Backend      │  ← trait defined in core crate
                    │  (mat_vec, add, mul) │
                    └──────────┬───────────┘
                               │ impl
                    ┌──────────┴──────────┐
                    ▼                     ▼
           ┌────────────────┐  ┌────────────────────┐
           │  CpuBackend    │  │    CudaBackend     │
           │ (ggml-cpu)     │  │ (ggml-cuda)        │
           │ SIMD + rayon   │  │ cuBLAS gemm        │
           │ std::thread    │  │ transparent CPU    │
           │                │  │ fallback on failure│
           └────────────────┘  └────────────────────┘
                    │                     │
                    └──────────┬──────────┘
                               │ Arc<dyn Backend>
                               ▼
                    ┌──────────────────────┐
                    │  llama::create_backend│  ← factory function
                    │  BackendType::Auto    │  ← priority: CUDA → CPU
                    └──────────────────────┘
```

### Backend Selection

`create_backend(&ModelConfig) -> Arc<dyn Backend>` uses a linear-priority chain:

| Priority | Backend     | Condition                    |
|----------|-------------|------------------------------|
| 1st      | CUDA        | `BackendType::Cuda` or `Auto` + `use_cuda` enabled, CUDA available |
| 2nd      | CPU         | Always available (fallback)  |

### CLI Usage

```bash
# Default: auto-select (CUDA if available, else CPU)
llama-cli -m model.gguf -p "Hello" -n 128

# Force CPU
llama-cli -m model.gguf --backend cpu -p "Hello" -n 128

# Force CUDA
llama-cli -m model.gguf --backend cuda -p "Hello" -n 128
```

### Extending

To add a new hardware backend:

1. Implement `ggml::backend::Backend` for your backend struct.
2. Add a variant to the selection logic in `crates/llama/src/backend.rs`.
3. The inference pipeline automatically dispatches through the `Arc<dyn Backend>`.

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Trait in `ggml` (core) | Both ggml-cpu and ggml-cuda depend on ggml; no circular deps |
| Object-safe trait (`&[f32]` / `Vec<f32>`) | Avoids generics in `InferenceContext`, keeps dispatch simple |
| Compile-time registry, not dynamic loading | Simpler, faster compile, no runtime linking |
| CUDA fallback on failure | If cuBLAS fails, transparently retries on CPU |
| `--backend` CLI flag | Explicit selection overrides auto-detection |

## External Integrations

| Integration | What it does |
|-------------|-------------|
| Filesystem | Reads GGUF model files via `memmap2` |
| CUDA (optional) | GPU matmul via `cudarc` + cuBLAS |
| HTTP | Axum server for inference API |
| Criterion | Benchmarking framework |

## Configuration

| File | Purpose |
|------|---------|
| `.cargo/config.toml` | `target-cpu=bdver1` for all builds |
| `rustfmt.toml` | max_width=100, tab_spaces=4, reorder_imports |
| `deny.toml` | License policy: MIT, Apache-2.0, Unlicense |
| `.github/workflows/ci.yml` | CI pipeline definition |

## Build & Deploy

```bash
# Debug build
cargo build --workspace --verbose

# Release build (optimized: -O3, LTO thin, codegen-units=1, strip debuginfo)
cargo build --release --workspace

# CUDA is included by default (requires CUDA toolkit)
# To build without CUDA:
#   cargo build --release --no-default-features -p ggml-cuda

# Test
cargo test --workspace --verbose
cargo test --workspace --doc

# CI pipeline (5 steps — `cargo test` compiles, no separate build step)
cargo fmt --all -- --check          # 1. Format check
cargo clippy --workspace -- -D warnings  # 2. Lint (warnings as errors)
cargo test --workspace --verbose    # 3. Unit + integration tests (compiles first)
cargo deny check licenses           # 4. License audit (uses EmbarkStudios/cargo-deny-action)
cargo doc --no-deps --document-private-items  # 5. Documentation build

# Benchmarks
cargo bench -p ggml-cpu --bench cpu_bench      # Matmul, dot product, parallel threshold
cargo bench -p llama --bench profiling          # End-to-end forward pass
cargo bench -p llama --bench kv_cache           # KV cache push, push_batch, reset, truncate
cargo bench -p llama --bench attention          # RoPE scaling, flash attention (full + window)
```
