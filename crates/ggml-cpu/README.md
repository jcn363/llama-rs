# GGML‑CPU Crate

CPU‑only backend for the GGML library, exposing optimized kernels.

## Key Public APIs

- `ggml_cpu::CpuBackend` – implements `ggml::backend::Backend` trait for CPU execution.
- `ggml_cpu::dot_f32` – SIMD dot product (AVX → SSE4.2 → scalar).
- `ggml_cpu::matmul_f32` – block-tiled matrix multiplication (16×16, parallel via `std::thread::scope`).

## Build Instructions

```bash
cd crates/ggml-cpu
cargo build --workspace
```
