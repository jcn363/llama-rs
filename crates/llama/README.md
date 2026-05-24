# Llama Crate

Core inference engine implementing the Llama architecture.

## Key Public APIs

- `llama::Model` – model loading with hyper-parameters, tensors, and KV cache.
- `llama::InferenceContext` – inference session (ties model + backend + tokenizer + sampling).
- `llama::ModelConfig` – configuration for inference (threads, context size, backend type, CUDA flag).
- `llama::BackendType` – enum to select hardware backend: `Auto`, `Cpu`, `Cuda`.
- `llama::create_backend` – factory function: `(&ModelConfig) -> Arc<dyn Backend>`.
- Re-exports `ggml::backend::{Backend, BackendInfo}` for convenience.

## Build Instructions

```bash
cd crates/llama
cargo build --workspace
```
