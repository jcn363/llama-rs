# GGML‑CUDA Crate

CUDA‑accelerated backend for GGML, enabling GPU inference.

## Key Public APIs

- `ggml_cuda::CudaBackend` – implements `ggml::backend::Backend` trait for GPU acceleration.
- `ggml_cuda::DeviceTensor` – GPU-side tensor with `copy_to_device()` / `to_host()`.
- `ggml_cuda::CudaError` / `CudaResult<T>` – error types for CUDA operations.

## Build Instructions

```bash
cd crates/ggml-cuda
# CUDA is opt-in via --features cuda (requires CUDA toolkit)
cargo build --release --features cuda
```
