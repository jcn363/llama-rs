# Llama-Server Crate

HTTP server exposing LLM inference via a REST API. Built with **axum 0.8** and **tower-http**.

## Usage

```bash
llama-server -m model.gguf --host 0.0.0.0 --port 8080
```

## Source

- **Entry point:** `src/main.rs` (407 lines)
- **Framework:** axum 0.8 with tower-http CORS
- **Dependencies:** `llama` (inference), `common` (shared args, sampling), `tokio` (async runtime)

## CLI Arguments

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | `String` | — | **Required.** Path to GGUF model file. |
| `--host` | — | `String` | `127.0.0.1` | Host address to bind to. |
| `--port` | `-p` | `u16` | `8080` | Port to listen on. |

Plus all `CommonArgs` flags (see [common/README.md](../common/README.md)).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `GET` | `/v1/models` | List models (OpenAI-compatible) |
| `GET` | `/samplers` | Get default sampling config |
| `POST` | `/tokenize` | Tokenize text → `{tokens, count}` |
| `POST` | `/completion` | Generate text (streaming or non-streaming) |

### POST /completion

**Request:**
```json
{
  "prompt": "The capital of France is",
  "max_tokens": 128,
  "stream": false,
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.95,
  "repeat_penalty": 1.1,
  "seed": null
}
```

**Non-streaming response:**
```json
{
  "content": "Paris is the capital of France.",
  "model": "Llama 7B Q4_K_M",
  "tokens_per_sec": 42.5
}
```

**Streaming response** (`stream: true`): Returns `text/event-stream` with SSE chunks:
```
data: {"content": "Paris", "stop": false}
data: {"content": "", "stop": true}
```

## Build

```bash
cargo build -p llama-server --release
# Binary: target/release/llama-server
```

## Architecture

1. On startup: parse args → load model → create `ServerState` (model + config in `Arc`) → bind TCP
2. Each request creates its own `InferenceContext` (independent KV cache, tokenizer)
3. Graceful shutdown on SIGINT/SIGTERM via `tokio::select!`
4. CORS is fully permissive (all origins)

## Related

- [Server Help (comprehensive)](../../docs/SERVER_HELP.md)
- [llama crate](../llama/README.md) — Core inference engine
- [common crate](../common/README.md) — Shared argument definitions
