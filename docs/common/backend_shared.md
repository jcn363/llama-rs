# Backend Shared Documentation

This file aggregates the common snippets that appeared duplicated across several backend docs (CUDA‑FEDORA, OPENCL, BLIS, VirtGPU). It provides a concise reference for developers working on hardware acceleration layers.

## Common Build Flags

```bash
# Enable OpenCL support
cargo build --features opencl
# Enable CUDA support (requires CUDA toolkit)
cargo build --features cuda
# Enable AVX2/FMA optimizations (CPU backend)
cargo build --features avx2,fma
```

## Environment Variables

- `LLAMA_CUDA_PATH` – Path to CUDA installation.
- `OPENCL_LIB_DIR` – Directory containing OpenCL libraries.
- `RUSTFLAGS` – Additional compiler flags, e.g., `-C target-cpu=bdver1`.

## Performance Tips

- Prefer the **GPU backend** for large batch sizes; CPU backend works best for small, latency‑critical inference.
- Tune the **thread pool size** via `LLAMA_THREAD_COUNT`.
- Use **profile mode** (`cargo build --profile`) for accurate benchmarking.

## Known Issues

- Some older GPUs may not support the latest **CUDA kernels**; fall back to **OpenCL** or **CPU**.
- AVX2/FMA kernels require a CPU with the corresponding instruction set; otherwise the binary will panic at startup.

For detailed per‑backend documentation, see the original files:
- `docs/backend/CUDA-FEDORA.md`
- `docs/backend/OPENCL.md`
- `docs/backend/BLIS.md`
- `docs/backend/VirtGPU.md`
