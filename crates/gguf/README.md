# GGUF Crate

Handles the GGUF model file format used for Llama models.

## Key Public APIs

- `gguf::reader::Reader` – parse GGUF files.
- `gguf::writer::Writer` – write GGUF files.

## Build Instructions

```bash
cd crates/gguf
cargo build --workspace
```
