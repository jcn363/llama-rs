# GGML Crate

Low‑level linear algebra library powering the Llama inference engine.

## Key Public APIs

- `ggml::tensor::Tensor` – core tensor type.
- `ggml::ops::*` – matrix multiplication, softmax, etc.

## Build Instructions

```bash
cd crates/ggml
cargo build --workspace
```
