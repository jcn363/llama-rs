# GGUF Crate

Handles the GGUF model file format used for Llama models.

## Key Public APIs

- `gguf::GgufReader` – parse GGUF files with memory-mapped I/O.
- `gguf::TensorInfo`, `gguf::MmapTensor` – tensor metadata and access.
- `gguf::GgufValue`, `gguf::GgmlType`, `gguf::GgufType` – metadata and type enums.
- `gguf::GgufError`, `gguf::GgufResult` – error handling.

## Features

- Full support for GGUF v3 specification
- Support for all quantization types used by Ollama (Q8_K, Q1_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1, IQ_XXS, IQ_XS, IQ_S, IQ_M)
- Memory-mapped I/O for efficient loading of large models
- Comprehensive metadata handling (13 types)
- Support for 42 tensor types

## Build Instructions

```bash
cd crates/gguf
cargo build --workspace
```
