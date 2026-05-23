# llama‑rs Usage Guide

## Overview
`llama-rs` provides a Rust implementation of LLaMA‑style language models with optional CUDA acceleration. The workspace is a Cargo workspace with several crates:
- `ggml` – core tensor library (CPU only)
- `ggml-cpu` – CPU‑only backend
- `ggml-cuda` – CUDA‑accelerated backend
- `llama` – high‑level model API (inference, profiling, KV‑cache)
- `llama-cli` – command‑line interface
- `llama-server` – HTTP server exposing the model

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
The `Model` struct now exposes `run_batch(prompts: &[&str]) -> Vec<String>` which runs each prompt sequentially using the existing `infer` implementation. This is a placeholder for true batched inference; it provides a convenient API for callers that need to process many prompts.

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
`KvCacheManager` supports three strategies:
- `Full` – store the entire context (default).
- `SlidingWindow { size: usize }` – keep only the most recent `size` tokens.
- `PrefixOnly` – keep only the prompt prefix, discarding generated tokens.
Create a manager with a custom strategy:
```rust
let manager = KvCacheManager::with_strategy(CacheStrategy::SlidingWindow { size: 1024 });
```
The manager automatically enforces the strategy after each push.

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
