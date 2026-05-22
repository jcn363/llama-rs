# GGML‑CUDA Crate

CUDA‑accelerated backend for GGML, enabling GPU inference.

## Key Public APIs

- `ggml_cuda::CudaBackend` – GPU backend implementation.
- `ggml_cuda::ops::*` – CUDA kernels.

## Build Instructions

```bash
cd crates/ggml-cuda
# Enable CUDA feature
cargo build --release -p ggml-cuda --features cuda
```
