# Llama‑Server Crate

Provides an HTTP server exposing Llama inference via a REST API.

## Key Public APIs

- `llama_server::run` – starts the server.
- API endpoints are defined in `src/main.rs`.
- Accepts `--backend auto|cpu|cuda` CLI flag to select hardware backend.

## Build Instructions

```bash
cd crates/llama-server
cargo build --release --workspace
```
