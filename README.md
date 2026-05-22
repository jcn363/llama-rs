# llama.rs — LLaMA inference in Rust

![llama](media/llama1-banner.png)

A Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp), optimized for **AMD Opteron 3280** (bdver1) + **NVIDIA GTX 1050** (2GB VRAM).

## Project Structure & Module Organization

The workspace splits concerns across 8 domain crates, each handling a specific subsystem:

```
llama-rs/
├── crates/
│   ├── gguf/          # GGUF v3 parser — file format, tensor info, dequantization
│   ├── ggml/          # Core tensor library — Tensor, DType, computation graphs
│   ├── ggml-cpu/      # CPU backend — AVX/SSE4.2 SIMD matmul, block-tiling
│   ├── ggml-cuda/     # CUDA backend — cuBLAS matmul, VRAM tracking (requires CUDA toolkit)
│   ├── llama/         # Inference engine — transformer forward pass, attention, KV cache
│   ├── common/        # Shared utilities — argument parsing, sampling config
│   ├── llama-cli/     # CLI binary for interactive text generation
│   └── llama-server/  # HTTP server with /completion and /health endpoints
├── .cargo/            # cargo config (target-cpu=bdver1)
├── .github/workflows/ # CI: format → clippy → test → deny → doc
├── test-models/       # Test GGUF files (downloaded separately, gitignored)
├── media/             # Visual identity system and design assets
├── Cargo.toml         # Workspace root (9 members)
├── rustfmt.toml       # Formatting: max_width=100, 4-space indent
└── deny.toml          # License policy (MIT, Apache-2.0, Unlicense)
```

The architecture enforces **strict separation of concerns** — the GGUF parser (`gguf`) depends on zero internal crates; compute backends (`ggml-cpu`, `ggml-cuda`) depend only on `ggml`; the inference engine (`llama`) consumes all lower layers but never reaches into binaries.

### Documentation & Design

A complete visual identity system is available in the `media/` directory, showcasing the project's branding and design language:

- **media/index.html** — Interactive design system demonstrating the Ethereal Glass aesthetic
- **Media assets** — Logos, banners, and diagrams illustrating the architecture and performance
- **Design principles** — Fluid motion, micro-interactions, and a vibrant color palette for each of the 8 crates

The design features:
- Distinct accent colors for each crate (gguf: purple, ggml: blue, ggml-cpu: emerald, ggml-cuda: pink, llama: amber, common: slate, llama-cli: orange, llama-server: indigo)
- Asymmetrical Bento layout with Double-Bezel architecture
- Fluid Island navigation with morphing hamburger icon
- Scroll-triggered reveal animations with custom cubic-bezier curves
- Performance-guardrailed animations using only transform and opacity

## Hardware Target

| Component | Specs |
|-----------|-------|
| **CPU** | AMD Opteron 3280 (Bulldozer bdver1) — 8 cores, 32GB RAM |
| **SIMD** | SSE4.2, AVX (NO FMA, AVX2, AVX512) |
| **GPU** | NVIDIA GTX 1050 — 640 CUDA cores, 2GB VRAM, Compute 6.1 |

## Performance

| Operation | Size | Single Thread | Parallel (8 cores) | Speedup |
|-----------|------|---------------|-------------------|---------|
| Matmul | 64×64 | 86µs | 429µs | 0.2x (overhead) |
| Matmul | 128×128 | 637µs | 562µs | 1.1x |
| Matmul | 256×256 | 4.2ms | 1.7ms | 2.5x |
| Matmul | 512×512 | 36.5ms | 11.4ms | **3.2x** |
| Dot product | 4096 | 1.1µs | - | - |
| Forward pass (13M params) | 1 token | ~18µs | - | - |
| Token generation (13M params) | 5 tokens | ~84µs | - | - |

## Quick Start

```bash
# Build (debug)
cargo build --workspace

# Build (release, optimized)
cargo build --release --workspace

# Run CLI
./target/release/llama-cli -m model.gguf -p "Hello, world!" -n 128

# Run server
./target/release/llama-server -m model.gguf --host 0.0.0.0 --port 8080

# Test
cargo test --workspace

# Benchmark
cargo bench -p ggml-cpu --bench cpu_bench
```

## Features

- **GGUF v3 parser**: Full support for 13 metadata types, 42 tensor types, memory-mapped I/O, and all quantization types used by Ollama (Q8_K, Q1_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1, IQ_XXS, IQ_XS, IQ_S, IQ_M)
- **SIMD matmul**: AVX 8-wide (32 floats/iter) → SSE4.2 4-wide (16 floats/iter) → scalar fallback
- **CUDA backend**: cuBLAS matmul, VRAM tracking (enabled by default, requires CUDA toolkit)
- **Inference engine**: RMSNorm, RoPE, multi-head attention with GQA, SwiGLU FFN, KV cache, flash attention
- **Sampling**: Greedy, temperature, top-k, top-p (nucleus)
- **Multi-architecture support**: Llama, Mistral, Phi2/3, Gemma/Gemma2, Qwen2, StableLM
- **Profiling**: Per-layer timing benchmarks with `generate_with_profile()` method
- **Memory-mapped tensors**: Lazy loading via `MmapTensor` for reduced memory footprint
- **CLI**: Interactive mode, single prompt, streaming token output
- **Server**: POST `/completion`, GET `/health`, JSON API, SSE streaming

## CLI Commands

**llama-cli** — Interactive text generation:
```
llama-cli -m model.gguf [-p "prompt"] [-n 128] [-t 0] [-c 512]
```
- `-m, --model` — Path to GGUF model file (required)
- `-p, --prompt` — Input prompt (empty for interactive mode)
- `-n, --n-predict` — Maximum tokens to generate (default: 128)
- `-t, --threads` — Worker threads (0 = auto-detect)
- `-c, --ctx-size` — Context window size (default: 512)
- `--verbose` — Enable debug logging

**llama-server** — HTTP inference API:
```
llama-server -m model.gguf [--host 127.0.0.1] [--port 8080]
```
- `-m, --model` — Path to GGUF model file (required)
- `--host` — Bind address (default: 127.0.0.1)
- `-p, --port` — Listen port (default: 8080)
- `-t, --threads` — Worker threads (0 = auto-detect)
- `-c, --ctx-size` — Context window size (default: 512)

**Endpoints:**
- `GET /health` — Health check, returns `{"status": "ok"}`
- `POST /completion` — Generate text (`prompt`, `max_tokens`, `stream`, `temperature`)

## Development

### Build & Test

| Command | Purpose |
|---------|---------|
| `cargo build --workspace` | Debug build |
| `cargo build --release --workspace` | Release build (LTO thin, codegen-units=1) |
| `cargo check` | Type-check without producing binaries |
| `cargo test --workspace` | Run all unit + integration tests |
| `cargo test --workspace --doc` | Run doctests |
| `cargo test [test_name] -- --nocapture` | Run a single test with output |
| `cargo bench -p ggml-cpu --bench cpu_bench` | Run CPU benchmarks |
| `cargo bench -p llama --bench profiling` | Run profiling benchmarks |

### Linting & Formatting

| Command | Purpose |
|---------|---------|
| `cargo fmt --all -- --check` | Verify formatting (CI enforces this) |
| `cargo fmt --all` | Auto-format all code |
| `cargo clippy --workspace -- -D warnings` | Lint with warnings-as-errors |
| `cargo deny check licenses` | Audit dependency licenses |

**CI pipeline** runs: format check → clippy (warnings as errors) → test → license audit → doc build.

### CUDA Build

CUDA is enabled by default (requires NVIDIA GPU + CUDA toolkit). To build without CUDA:

```bash
cargo build --release --no-default-features -p ggml-cuda
```

## Build Configuration

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=bdver1"]
```

## Code Style & Best Practices

See [`CODE_STYLE.md`](./CODE_STYLE.md) for full project conventions. For a deeper reference on Rust best practices (borrowing, error handling, testing, unsafe, async, workspace management), see [`docs/RBP.md`](./docs/RBP.md). For contribution workflow, see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

Key rules from `CODE_STYLE.md`:

- **Naming**: `snake_case` for files/functions/variables, `PascalCase` for types/enums
- **Errors**: `thiserror` for libraries, `anyhow` for binaries; never `.unwrap()` outside tests
- **Concurrency**: `Arc` for shared ownership, `RwLock` for read-mostly state, `rayon` for data parallelism
- **SIMD**: AVX → SSE4.2 → scalar fallback (no FMA, no AVX2 — bdver1 target)
- **Unsafe**: Every `unsafe` block must have a `// SAFETY:` comment
- **Clippy**: The workspace allows pedantic lints globally; individual crates opt in with `#![deny(clippy::pedantic)]` and explicit allow-list

## Testing Guidelines

Tests are organized as:
- **Unit tests**: Inline in source files under `#[cfg(test)] mod tests { ... }`
- **Integration tests**: `crates/<name>/tests/<name>_test.rs`
- **Benchmarks**: `crates/<name>/benches/<name>.rs` (criterion)
- **Doctests**: In doc comments (`/// ```no_run ...`)

Naming convention: `describe_should_expected_behavior` — e.g., `dot_f32_should_compute_correct_result`.

Tests that require external model files skip gracefully when the file is absent:
```rust
if !model_path.exists() {
    println!("Skipping: test model not found");
    return;
}
```

## Commit Guidelines

- Format: `phase [N]: [description]` for feature phases, plain titles for fixes/refactors
- Keep commits focused on a single logical concern
- Messages should be imperative and descriptive enough to understand the change without reading the diff
- All commits must pass `cargo fmt --all -- --check` and `cargo clippy --workspace -- -D warnings`

## Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1.1 | ✅ | Workspace setup (8 crates) |
| 1.2 | ✅ | GGUF v3 parser |
| 2 | ✅ | SIMD matmul (AVX + SSE4.2) |
| 3 | ✅ | CUDA backend (cuBLAS) |
| 4 | ✅ | Inference engine (transformer) |
| 5 | ✅ | CLI and server binaries |
| 6 | ✅ | CI/CD pipeline |
| 7 | ✅ | Benchmarks |
| 8 | ✅ | Multi-architecture support (Llama, Mistral, Phi, Gemma, Qwen2) |
| 9 | ✅ | Flash attention and memory-mapped tensors |
| 10 | ✅ | Profiling and per-layer timing benchmarks |

**46 tests pass** across all crates (43 unit/integration + 3 doctests).

## License

MIT (same as llama.cpp)
