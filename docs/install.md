# Installing llama-rs

llama-rs is a Rust port of llama.cpp for LLM inference. This guide covers building from source and installing the Debian package.

## Prerequisites

- **Rust toolchain** ≥ 1.85 (stable) — install via [rustup.rs](https://rustup.rs/)
- **Required components**: `rustfmt` and `clippy`
  ```bash
  rustup component add rustfmt clippy
  ```
- **Optional**: CUDA toolkit (for GPU acceleration via `ggml-cuda` backend)

## Build from Source

### Debug Build
```bash
cargo build --workspace --verbose
```

### Release Build (Optimized)
```bash
cargo build --release --workspace
```

### Build Without CUDA
CUDA is enabled by default. To build without it:
```bash
cargo build --release --no-default-features -p ggml-cuda
```

### Run the Binaries
```bash
# CLI — interactive text generation
./target/release/llama-cli -m model.gguf -p "Hello, world!" -n 128

# Server — HTTP API with /completion and /health endpoints
./target/release/llama-server -m model.gguf --host 0.0.0.0 --port 8080

# UI — Desktop GUI (multi-pane chat interface)
./target/release/llama-ui
```

## Debian/Ubuntu Package

A `.deb` package is available for easy installation on Debian-based systems.

```bash
# Install the package (adjust filename as needed)
sudo dpkg -i llama-rs_0.1.0-1_amd64.deb

# Verify installation
llama-cli --version
llama-server --version
llama-ui --version

# Uninstall if needed
sudo dpkg -r llama-rs
```

The package installs three binaries to `/usr/bin/`:
- `llama-cli` — Command-line interface for interactive text generation
- `llama-server` — HTTP server with `/completion` and `/health` endpoints
- `llama-ui` — Desktop GUI for interactive LLM inference

See [README.md](../README.md#debian-package) for the exact `.deb` filename.

## Testing

Run the full test suite (unit tests + integration tests + doctests):

```bash
# Unit and integration tests
cargo test --workspace --verbose

# Doctests
cargo test --workspace --doc
```

Run benchmarks (requires `criterion`):

```bash
# CPU benchmarks
cargo bench -p ggml-cpu --bench cpu_bench

# Inference profiling benchmarks
cargo bench -p llama --bench profiling
cargo bench -p llama --bench kv_cache
cargo bench -p llama --bench attention
```

## Linting & Formatting

```bash
# Check formatting (CI enforces this)
cargo fmt --all -- --check

# Auto-format
cargo fmt --all

# Lint with clippy (warnings as errors)
cargo clippy --workspace -- -D warnings

# License audit
cargo deny check licenses
```

## Hardware Targets

| Component | Specs |
|-----------|-------|
| **CPU** | AMD Opteron 3280 (Bulldozer bdver1) — 8 cores |
| **SIMD** | SSE4.2, AVX (no FMA, AVX2, AVX512) |
| **GPU** | NVIDIA GTX 1050 — 2GB VRAM, Compute 6.1 |

The build uses `-C target-cpu=bdver1` via `.cargo/config.toml` for optimal SIMD code generation.