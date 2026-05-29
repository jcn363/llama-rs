# Llama-CLI Crate

Command-line interface for interactive text generation using GGUF models.

## Usage

```bash
# Interactive mode (reads prompt from stdin)
llama-cli -m model.gguf

# Single prompt
llama-cli -m model.gguf -p "Hello, world!" -n 128

# With custom sampling
llama-cli -m model.gguf -p "Write a haiku" --temperature 0.5 --top-k 20 --seed 42
```

## Source

- **Entry point:** `src/main.rs` (128 lines)
- **Framework:** `clap` (derive API) for argument parsing
- **Dependencies:** `llama` (inference), `common` (shared args, sampling)

## CLI Arguments

The binary flattens `common::args::CommonArgs` and adds three CLI-specific flags:

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | `String` | — | **Required.** Path to GGUF model file. |
| `--prompt` | `-p` | `String` | `""` | Prompt text. Empty = interactive mode. |
| `--n-predict` | `-n` | `usize` | `128` | Max tokens to generate. |
| `--verbose` | — | `bool` | `false` | Enable debug logging. |

Plus all `CommonArgs` flags (see [common/README.md](../common/README.md)).

## Build

```bash
cargo build -p llama-cli --release
# Binary: target/release/llama-cli
```

## Behavior

1. Parses args via `clap::Parser::parse()`
2. Loads GGUF model via `Model::load_from_gguf()`
3. Creates `InferenceContext` with `SamplingConfig`
4. If `--prompt` is empty → reads from stdin (interactive mode)
5. Generates tokens via `ctx.generate(prompt, n_predict)`
6. Prints decoded text to stdout, stats to stderr

## Related

- [CLI Help (comprehensive)](../../docs/CLI_HELP.md)
- [llama crate](../llama/README.md) — Core inference engine
- [common crate](../common/README.md) — Shared argument definitions
