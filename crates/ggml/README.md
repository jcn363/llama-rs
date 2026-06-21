# GGML Crate

Low‑level linear algebra library powering the Llama inference engine.

## Key Public APIs

- `ggml::tensor::Tensor` – core multi-dimensional array type.
- `ggml::backend::Backend` – object-safe trait for hardware-accelerated tensor math.
- `ggml::backend::BackendInfo` – metadata struct (name, availability, memory, parallelism).

> **Note:** `Backend` and `BackendInfo` are in the `ggml::backend` module, not re-exported at the crate root.

## Build Instructions

```bash
cd crates/ggml
cargo build --workspace
```
