# llama‑rs Usage Guide

## Overview
`llama-rs` provides a Rust implementation of LLaMA‑style language models with optional CUDA acceleration. The workspace is a Cargo workspace with 17 crates:

**Foundation Layer:**
- `error` – unified error handling
- `config` – configuration management
- `gguf` – GGUF v3 file parser with memory-mapped I/O
- `ggml` – core tensor types, computation graph, `Backend` trait
- `ggml-cpu` – CPU backend (AVX/SSE4.2 SIMD matmul)
- `ggml-cuda` – CUDA backend (cuBLAS matmul, requires CUDA toolkit)
- `llama-core` – core inference traits and shared types (`Model`, `InferenceContext`, `KvCache`)

**Inference Layer:**
- `llama` – inference engine (transformer forward pass, attention, KV cache, sampling)
- `common` – shared utilities (sampling config, chat templates, argument parsing)

**Application Layer:**
- `llama-cli` – interactive CLI binary
- `llama-server` – HTTP REST API server (Axum, SSE streaming)
- `llama-ui-core` – shared UI types, theme, error types
- `llama-ui-models` – model discovery, manifest, GGUF metadata extraction
- `llama-ui-session` – chat history, session persistence, export (JSON/MD/plain)
- `llama-ui-sandbox-client` – sandboxed server spawning with resource limits
- `llama-ui` – desktop GUI (Iced 0.13, multi-pane chat)

## Building
```bash
# Debug build (all crates)
cargo build --workspace

# Release build (optimised binaries)
cargo build --release --workspace
```
CUDA support is enabled by default. To build without CUDA (e.g., on CI without a GPU):
```bash
cargo build --release --no-default-features -p ggml-cuda
```

## Running the CLI
```bash
./target/release/llama-cli -m model.gguf -p "Your prompt here" -n 128
```
- `-m` – path to a GGUF model file.
- `-p` – prompt string.
- `-n` – number of tokens to generate.

### Batch inference
The `Model` struct exposes `run_batch(self, prompts: &[&str]) -> Vec<Vec<usize>>` which runs each prompt sequentially using `InferenceContext::generate()`. This consumes the model and returns token ID sequences (not decoded strings). For streaming or repeated generation, use `InferenceContext` directly:

```rust
let model = Arc::new(Model::load_from_gguf("model.gguf", false)?);
let mut ctx = InferenceContext::new(model, ModelConfig::default());
let tokens = ctx.generate("Your prompt", 128)?;           // Returns Vec<usize>
let text = ctx.decode(&tokens);                           // Decode to String

// Or generate from pre-encoded tokens (for prefix caching):
let prompt_tokens = ctx.encode("Your prompt");
let new_tokens = ctx.generate_from_tokens(&prompt_tokens, 128)?;
```

## Server mode
```bash
./target/release/llama-server -m model.gguf --host 0.0.0.0 --port 8080
```
The server accepts JSON POST requests:
```json
{ "prompt": "Hello", "max_tokens": 64 }
```
and returns a JSON response with the generated text.

## Profiling & JSON export
`ProfileResult` now implements `serde::Serialize`/`Deserialize` and provides a `to_json(&self) -> String` helper. The profiling benchmark (`crates/llama/benches/profiling.rs`) checks that the JSON round‑trip works.

## KV‑Cache strategies
`KvCacheManager` supports four strategies via the `CacheStrategy` enum:
- `Full` – store the entire context (default).
- `Prefix` – prefix caching: trim during generation for long contexts, keeping common prefix.
- `SlidingWindow { size: usize }` – keep only the most recent `size` tokens.
- `PrefixOnly` – keep only the initial prompt prefix, discarding generated tokens.

Create a manager with a custom strategy (requires all cache dimensions):
```rust
let manager = KvCacheManager::with_strategy(
    n_layers,      // number of transformer layers
    max_seq,       // maximum sequence length
    n_head_kv,     // number of key/value heads (GQA/MQA)
    head_dim,      // dimension per head
    CacheStrategy::SlidingWindow { size: 1024 },
);
```
The strategy is configured via `ModelConfig::cache_strategy` and applied automatically during `InferenceContext::generate()`.

## Benchmarks
- **CPU backend**: `crates/ggml-cpu/benches/cpu_bench.rs`
- **KV‑cache**: `crates/llama/benches/kv_cache_bench.rs` (new)
Run all benchmarks with:
```bash
cargo bench --workspace
```

## Testing
```bash
cargo test --workspace          # unit & integration tests
cargo test --workspace --doc    # doctests
```
All tests must pass `cargo fmt --all -- --check` and `cargo clippy --workspace -- -D warnings` before merging.

## CI
GitHub Actions workflows for macOS and Windows are provided under `.github/workflows/`. They run the full build, lint, and test matrix.

---
*For more details, see the individual crate READMEs and the `ARCHITECTURE.md` file.*
