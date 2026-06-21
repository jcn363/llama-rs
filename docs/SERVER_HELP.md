# llama-server — HTTP Server Help

`llama-server` is an HTTP server that exposes LLM inference via a REST API. It supports text generation (streaming and non-streaming), tokenization, health checks, and model information queries. Built with **axum 0.8**.

## Quick Start

```bash
# Start the server with a model
./target/release/llama-server -m model.gguf

# Bind to all interfaces on port 9090
./target/release/llama-server -m model.gguf --host 0.0.0.0 --port 9090

# Test with curl
curl http://localhost:8080/health
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 64}'
```

## Building

```bash
# Release build (recommended)
cargo build -p llama-server --release

# Debug build
cargo build -p llama-server

# The binary is at: target/release/llama-server
```

## Command-Line Flags

### Server Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | `String` | — | **Required.** Path to the GGUF model file. |
| `--host` | — | `String` | `127.0.0.1` | Host address to bind to. Use `0.0.0.0` for all interfaces. |
| `--port` | `-p` | `u16` | `8080` | Port to listen on. |

### Inference Configuration Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--threads` | `usize` | `0` | Number of CPU threads. `0` = auto-detect. |
| `--ctx-size` | `usize` | `4096` | Context window size in tokens. |
| `--batch-size` | `usize` | `512` | Batch size for prompt processing. |
| `--cache-strategy` | `String` | `"full"` | KV cache strategy: `full`, `prefix`, or `prefix_only`. |
| `--backend` | `String` | `"auto"` | Compute backend: `auto`, `cpu`, or `cuda`. |
| `--offload-ffn` | `bool` | `false` | Offload FFN weights to RAM to save VRAM. |
| `--memory-pool-size` | `usize` | `0` | Thread-local memory pool size in bytes. `0` = disabled. |

### Sampling Flags (Default Values)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--temperature` | `f32` | `0.8` | Default sampling temperature. |
| `--top-k` | `usize` | `40` | Default top-k sampling. |
| `--top-p` | `f32` | `0.95` | Default top-p nucleus sampling. |
| `--repeat-penalty` | `f32` | `1.1` | Default repeat penalty. |
| `--seed` | `Option<u64>` | `None` | Default random seed. |

## API Endpoints

### `GET /health`

Health check endpoint. Returns server status.

**Response:** `200 OK`
```json
{ "status": "ok" }
```

**Use cases:**
- Monitoring server availability
- Load balancer health checks
- Client connection verification

---

### `GET /v1/models`

List available models. OpenAI-compatible format.

**Response:** `200 OK`
```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 0,
      "owned_by": "llama-rs",
      "description": "Llama 7B Q4_K_M — 7B parameters, 4096 context"
    }
  ]
}
```

The `description` field contains the model's summary (architecture, parameter count, context size, quantization).

---

### `GET /samplers`

Returns the default sampling configuration.

**Response:** `200 OK`
```json
{
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.95,
  "repeat_penalty": 1.1,
  "seed": null
}
```

---

### `POST /tokenize`

Tokenize a text string using the model's tokenizer.

**Request body:**
```json
{ "text": "Hello, world!" }
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | `string` | yes | Must not be empty. |

Uses `serde(deny_unknown_fields)` — extra fields are rejected.

**Success response:** `200 OK`
```json
{
  "tokens": [1, 2345, 678, 2],
  "count": 4
}
```

**Error response:** `400 BAD REQUEST`
```json
{ "error": "text must not be empty" }
```

---

### `POST /completion`

Generate text from a prompt. Supports both streaming (SSE) and non-streaming modes.

**Request body:**
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

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `prompt` | `string` | yes | — | Must not be empty. |
| `max_tokens` | `usize` | no | `128` | Hard-capped at `4096` server-side. |
| `stream` | `bool` | no | `false` | If `true`, returns SSE stream. |
| `temperature` | `f32` | no | `0.8` | `0.0` = greedy decoding. |
| `top_k` | `usize` | no | `40` | `0` = disabled. |
| `top_p` | `f32` | no | `0.95` | `1.0` = disabled. |
| `repeat_penalty` | `f32` | no | `1.1` | `1.0` = no penalty. |
| `seed` | `u64?` | no | `null` | Random if absent. |

Uses `serde(deny_unknown_fields)` — extra fields are rejected.

#### Non-Streaming Response (`stream: false`)

**Response:** `200 OK`
```json
{
  "content": "Paris is the capital of France.",
  "model": "Llama 7B Q4_K_M — 7B parameters, 4096 context",
  "tokens_per_sec": 42.5
}
```

| Field | Type | Notes |
|-------|------|-------|
| `content` | `string` | Decoded text of all generated tokens. |
| `model` | `string?` | Model summary string. |
| `tokens_per_sec` | `f64?` | Generated tokens per second. `null` if generation took 0 time. |

#### Streaming Response (`stream: true`)

**Response:** `200 OK` — `text/event-stream` (SSE)

Each event's `data` field contains a JSON object:

```
data: {"content": "Paris", "stop": false}
data: {"content": " is", "stop": false}
data: {"content": " the", "stop": false}
...
data: {"content": "", "stop": true}
```

| Field | Type | Notes |
|-------|------|-------|
| `content` | `string` | A single decoded token. Empty string on the final stop chunk. |
| `stop` | `bool` | `false` for content chunks, `true` for the final termination chunk. |

**Implementation note:** The server generates all tokens in a spawn_blocking task, collects them into a Vec, then wraps the result as an SSE event stream. Clients receive all tokens at once when generation completes.

---

## Error Responses

All error responses use the format:

```json
{ "error": "<error message>" }
```

| Endpoint | Status | Condition |
|----------|--------|-----------|
| `/completion` | `400` | Empty `prompt` |
| `/completion` | `500` | Inference error (forward pass failure) |
| `/tokenize` | `400` | Empty `text` |

## CORS

CORS is fully permissive (`CorsLayer::permissive()`) — all origins, methods, and headers are allowed. This is suitable for local development. For production, configure a restrictive CORS policy.

## Server Lifecycle

1. **Startup:** Parse args → init tracing → load GGUF model → build `ModelConfig` → create `ServerState` (model + config in `Arc`) → bind TCP listener.
2. **Request handling:** Each request creates its own `InferenceContext` (own KV cache, bump allocator, tokenizer). The model is shared across all requests.
3. **Shutdown:** Listens for `Ctrl+C` (SIGINT) or `SIGTERM`. Logs and exits cleanly.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Controls log verbosity. Default: `info`. |

## Examples

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Get default samplers
curl http://localhost:8080/samplers

# Tokenize text
curl -X POST http://localhost:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# Non-streaming completion
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "temperature": 0.7
  }'

# Streaming completion
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "prompt": "Write a haiku about Rust:",
    "max_tokens": 128,
    "stream": true,
    "temperature": 0.9
  }'
```

### Python Example

```python
import requests

# Non-streaming
response = requests.post("http://localhost:8080/completion", json={
    "prompt": "Explain recursion:",
    "max_tokens": 256,
    "temperature": 0.7
})
print(response.json()["content"])

# Streaming
response = requests.post("http://localhost:8080/completion",
    json={"prompt": "Tell me a joke:", "max_tokens": 128, "stream": True},
    stream=True
)
for line in response.iter_lines():
    if line and line.startswith(b"data: "):
        import json
        chunk = json.loads(line[6:])
        if chunk["stop"]:
            break
        print(chunk["content"], end="", flush=True)
```

## Limitations

- **No authentication** — The server has no auth mechanism. Do not expose to untrusted networks.
- **No rate limiting** — All requests are processed immediately.
- **No session management** — Each request is independent; no conversation history is maintained server-side.
- **Max tokens cap** — `max_tokens` is hard-capped at 4096 regardless of client request.
- **Streaming is batch-then-stream** — Tokens are collected in a blocking thread before streaming to the client.

## Troubleshooting

### "Address already in use"
Another process is using the port. Use `--port` to specify a different port, or find and stop the conflicting process.

### "Model file not found"
Ensure the path to the GGUF file is correct. Use absolute paths.

### Connection refused
Ensure the server is running and bound to the correct interface. Use `--host 0.0.0.0` to accept connections from other machines.

### Slow generation
- Use a smaller model or lower quantization.
- Enable CUDA backend: `--backend cuda`.
- Increase threads: `--threads 8`.

## Related

- [llama-cli help](CLI_HELP.md) — Command-line text generation
- [llama-ui help](UI_HELP.md) — Desktop GUI application
- [Project README](../README.md) — Build instructions and overview
- [Architecture](../ARCHITECTURE.md) — Crate dependency graph and data flow
