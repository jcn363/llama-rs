# Llama Crate

Core inference engine implementing the Llama architecture.

## Key Public APIs

- `llama::model::Model` – model loading and configuration.
- `llama::inference::InferenceEngine` – run inference.
- `llama::tokenizer::Tokenizer` – tokenization utilities.

## Build Instructions

```bash
cd crates/llama
cargo build --workspace
```
