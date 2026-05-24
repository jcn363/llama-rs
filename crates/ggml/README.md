# GGML Crate

Low‑level linear algebra library powering the Llama inference engine.

## Key Public APIs

- `ggml::tensor::Tensor` – core multi-dimensional array type.
- `ggml::backend::Backend` – object-safe trait for hardware-accelerated tensor math.
- `ggml::backend::BackendInfo` – metadata struct (name, availability, memory, parallelism).

## Build Instructions

```bash
cd crates/ggml
cargo build --workspace
```
