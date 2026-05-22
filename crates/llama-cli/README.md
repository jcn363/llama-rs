# Llama‑CLI Crate

Command‑line interface for running Llama models.

## Key Public APIs

- `llama_cli::main` – entry point for the binary.
- Uses the `llama` crate for model loading and inference.

## Build Instructions

```bash
cd crates/llama-cli
cargo build --release --workspace
```
