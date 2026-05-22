# GGML‑CPU Crate

CPU‑only backend for the GGML library, exposing optimized kernels.

## Key Public APIs

- `ggml_cpu::cpu_backend::CpuBackend` – entry point for CPU execution.
- `ggml_cpu::ops::*` – CPU‑specific implementations.

## Build Instructions

```bash
cd crates/ggml-cpu
cargo build --workspace
```
