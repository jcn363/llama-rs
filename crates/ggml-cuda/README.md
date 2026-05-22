# GGML‑CUDA Crate

CUDA‑accelerated backend for GGML, enabling GPU inference.

## Key Public APIs

- `ggml_cuda::CudaBackend` – GPU backend implementation.
- `ggml_cuda::ops::*` – CUDA kernels.

## Build Instructions

```bash
cd crates/ggml-cuda
# CUDA is enabled by default (requires CUDA toolkit)
cargo build --release
```
