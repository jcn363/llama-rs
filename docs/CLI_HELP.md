# llama-cli — Command-Line Interface Help

`llama-cli` is the interactive command-line tool for text generation using GGUF models. It supports single-prompt generation, interactive mode, streaming output, and configurable sampling parameters.

## Quick Start

```bash
# Interactive mode (reads from stdin)
./target/release/llama-cli -m model.gguf

# Single prompt
./target/release/llama-cli -m model.gguf -p "The capital of France is" -n 128

# With custom sampling and backend
./target/release/llama-cli -m model.gguf -p "Write a haiku about Rust" \
  --temperature 0.5 --top-k 20 --top-p 0.9 --repeat-penalty 1.2 \
  --seed 42 --backend cpu --threads 8

# Large context with prefix caching
./target/release/llama-cli -m model.gguf -p "Summarize this document:" \
  --ctx-size 16384 --batch-size 1024 --cache-strategy prefix
```

## Building

```bash
# Release build (recommended)
cargo build -p llama-cli --release

# Debug build
cargo build -p llama-cli

# The binary is at: target/release/llama-cli
```

## Command-Line Flags

### Required Flags

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--model` | `-m` | `String` | Path to the GGUF model file. No default — this flag is required. |

### Generation Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--prompt` | `-p` | `String` | `""` (empty) | Prompt text. If empty, enters **interactive mode** (reads from stdin). |
| `--n-predict` | `-n` | `usize` | `128` | Maximum number of tokens to generate. |
| `--verbose` | — | `bool` | `false` | Enable verbose debug logging to stderr. |

### Inference Configuration Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--threads` | `usize` | `0` | Number of CPU threads. `0` = auto-detect (uses all logical cores). |
| `--ctx-size` | `usize` | `4096` | Context window size in tokens. Larger values use more memory. |
| `--batch-size` | `usize` | `512` | Batch size for prompt processing (tokens processed in parallel). |
| `--cache-strategy` | `String` | `"full"` | KV cache strategy: `full`, `prefix`, or `prefix_only`. |
| `--backend` | `String` | `"auto"` | Compute backend: `auto` (CUDA if available, else CPU), `cpu`, or `cuda`. |
| `--offload-ffn` | `bool` | `false` | Offload FFN weights to RAM to save VRAM (load on demand). |
| `--memory-pool-size` | `usize` | `0` | Thread-local bump memory pool size in bytes. `0` = disabled. |

### Sampling Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--temperature` | `f32` | `0.8` | Sampling temperature. `0.0` = greedy decoding (deterministic). |
| `--top-k` | `usize` | `40` | Top-k sampling. `0` = disabled. |
| `--top-p` | `f32` | `0.95` | Top-p (nucleus) sampling. `1.0` = disabled. |
| `--repeat-penalty` | `f32` | `1.1` | Repeat penalty. Values `> 1.0` penalize repeated tokens. `1.0` = no penalty. |
| `--seed` | `Option<u64>` | `None` | Random seed. Omit for random (non-reproducible) output. |

## Interactive Mode

When `--prompt` is empty (the default), `llama-cli` enters **interactive mode**:

1. The model is loaded and an `InferenceContext` is created.
2. A prompt is read from **stdin** (one line).
3. Tokens are generated and printed to **stdout**.
4. Generation stats (token count, elapsed time, tokens/second) are printed to **stderr**.

```bash
# Example interactive session
$ ./target/release/llama-cli -m model.gguf
> What is the meaning of life?
[generated text appears here...]
```

## Output Format

- **Generated tokens** are printed to `stdout` as decoded text (streamed token-by-token).
- **Statistics** are printed to `stderr` after generation completes:
  ```
  Generated 42 tokens in 1.2s (35.0 tok/s)
  ```

## Behavior Details

### Backend Resolution

The `--backend` flag resolves as follows:
- `"auto"` → Uses CUDA if available, falls back to CPU
- `"cpu"` → CPU-only inference
- `"cuda"` → GPU inference (requires CUDA toolkit)
- Any other value → Falls back to `auto`

### Cache Strategies

| Strategy | Description |
|----------|-------------|
| `full` | Keeps the entire KV cache in memory. Default. |
| `prefix` | Caches prompt prefixes for reuse across generations. |
| `prefix_only` | Only caches prefixes; does not cache generated tokens. |

### Thread Count

When `--threads 0` (default), the CPU backend auto-detects the number of logical cores and uses all of them. Set explicitly to limit CPU usage.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Controls log verbosity (e.g., `debug`, `info`, `warn`, `error`). Default: `info`. |
| `LLAMA_MODEL_PATH` | Default model path if `--model` is omitted. |
| `LLAMA_NUM_THREADS` | Override thread count. |
| `LLAMA_BACKEND` | Default backend (`cpu` or `cuda`). |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (model load failure, invalid arguments, inference error) |

## Troubleshooting

### "model file not found"
Ensure the path to the GGUF file is correct and the file exists. Use absolute paths for reliability.

### "CUDA not available"
Build with CUDA support: `cargo build -p llama-cli --release`. Requires NVIDIA GPU + CUDA toolkit installed.

### Low tokens/second
- Use a smaller model or lower quantization (Q4_K_M is faster than Q8_0).
- Increase `--threads` if not using all CPU cores.
- Enable `--backend cuda` if you have a compatible GPU.
- Reduce `--ctx-size` if context window is unnecessarily large.

### Out of memory
- Reduce `--ctx-size` (each token uses ~4 bytes per context slot per layer).
- Enable `--offload-ffn` to offload FFN weights to RAM.
- Use a smaller model or lower quantization.

## Examples

```bash
# Basic generation
./target/release/llama-cli -m models/llama-7b-q4_k_m.gguf \
  -p "Write a Python function to sort a list:" -n 256

# Deterministic output with fixed seed
./target/release/llama-cli -m models/llama-7b-q4_k_m.gguf \
  -p "Explain quantum computing in one paragraph" \
  --seed 12345 --temperature 0.0

# Creative writing with high temperature
./target/release/llama-cli -m models/llama-7b-q4_k_m.gguf \
  -p "Write a poem about the ocean" \
  --temperature 1.2 --top-k 100 --top-p 0.95

# Force CPU backend with specific thread count
./target/release/llama-cli -m models/llama-7b-q4_k_m.gguf \
  -p "Hello" --backend cpu --threads 4

# Large context for document analysis
./target/release/llama-cli -m models/llama-7b-q4_k_m.gguf \
  -p "$(cat long_document.txt)" \
  --ctx-size 32768 --batch-size 2048 --cache-strategy prefix
```

## Related

- [llama-server help](SERVER_HELP.md) — HTTP server with REST API
- [llama-ui help](UI_HELP.md) — Desktop GUI application
- [Project README](../README.md) — Build instructions and overview
- [Architecture](../ARCHITECTURE.md) — Crate dependency graph and data flow
