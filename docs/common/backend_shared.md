# Backend Shared Documentation

This file was a shared reference for the upstream llama.cpp backend docs (CUDA‑FEDORA, OPENCL, BLIS, VirtGPU), which have been removed. See [ARCHITECTURE.md](../../ARCHITECTURE.md) for the llama-rs backend model.

## llama-rs Backend Configuration

llama-rs uses a compile-time plugin registry with two backends:

- **CPU backend** (`ggml-cpu`): AVX / SSE4.2 SIMD matmul with block-tiling, Rayon for data parallelism. Target: AMD Opteron 3280 (bdver1, 8 cores, no FMA, no AVX2).
- **CUDA backend** (`ggml-cuda`): cuBLAS matmul, VRAM tracking. Enabled by default; requires CUDA toolkit at build time.

### Build Flags

```bash
# Default build (CUDA enabled, requires CUDA toolkit)
cargo build --release --workspace

# Disable CUDA backend
cargo build --release --no-default-features -p ggml-cuda

# CPU-only build (all crates)
cargo build --release --no-default-features --workspace
```

### Environment Variables

- `RUSTFLAGS` – Additional compiler flags, e.g., `-C target-cpu=bdver1` (set in CI and `.cargo/config.toml`).
- `CUDA_PATH` / `CUDA_HOME` – Standard CUDA toolkit paths (used by `cudarc`/`pkg-config` to locate CUDA). No `LLAMA_CUDA_PATH` variable.

### Performance Notes

- **GPU backend** (CUDA): Best for large batch sizes and high throughput on NVIDIA GPUs (tested on GTX 1050, Compute 6.1, 2GB VRAM).
- **CPU backend**: Best for small, latency-critical inference. Uses AVX (8-wide) → SSE4.2 (4-wide) → scalar fallback.
- Thread pool size: Controlled via `rayon` (default = logical cores). No `LLAMA_THREAD_COUNT` env var.

### Key Differences from llama.cpp

| Feature | llama.cpp | llama-rs |
|---------|-----------|----------|
| CUDA toggle | `-DLLAMA_CUDA=ON/OFF` | `--no-default-features -p ggml-cuda` |
| OpenCL backend | Yes | No |
| BLIS backend | Yes | No |
| VirtGPU backend | Yes | No |
| AVX2/FMA kernels | Yes | No (targets bdver1: AVX only, no FMA) |
| CUDA path env | `LLAMA_CUDA_PATH` | Standard `CUDA_PATH` / `CUDA_HOME` |

For the full crate dependency graph and data flow, see [ARCHITECTURE.md](../../ARCHITECTURE.md).