# Llama Crate

Core inference engine implementing the Llama architecture.

## Key Public APIs

- `llama::Model` – model loading with hyper-parameters, tensors, and KV cache.
- `llama::InferenceContext` – inference session (ties model + backend + tokenizer + sampling).
- `llama::ModelConfig` – configuration for inference (threads, context size, backend type, CUDA flag).
- `llama::BackendType` – enum to select hardware backend: `Auto`, `Cpu`, `Cuda`.
- `llama::create_backend` – factory function: `(&ModelConfig) -> Arc<dyn Backend>`.
- Re-exports `ggml::backend::{Backend, BackendInfo}` for convenience.

### Inference & Sampling

- `llama::inference::SamplingConfig` – sampling parameters (temperature, top-k, top-p, repeat penalty, seed).

### KV Cache

- `llama::kv_cache::KvCache` – key-value cache for transformer attention.
- `llama::kv_cache::CacheStrategy` – cache eviction strategy (e.g., `PrefixCaching`, `PushBatch`).
- `llama::kv_cache::KvCacheManager` – manages KV cache lifecycle and strategies.

### Profiling

- `llama::profile::ProfileResult` – per-layer timing results from `generate_with_profile()`.

### Tokenizer

- `llama::tokenizer::SimpleTokenizer` – basic BPE tokenizer for Llama models.

### Architecture Types (re-exported at crate root)

- `llama::RoPEConfig` – RoPE configuration (theta, scaling type, factor).
- `llama::NormType` – normalization type enum (`RMSNorm`, `LayerNorm`).
- `llama::RopeScaleType` – RoPE scaling type enum (`None`, `Linear`, `Ntk`, `DynamicNtk`).

### Internal Types (re-exported at crate root)

- `llama::TensorData` – tensor data container for GGUF loading.
- `llama::InternedStrings` – string interning for metadata.

## Build Instructions

```bash
cd crates/llama
cargo build --workspace
```
