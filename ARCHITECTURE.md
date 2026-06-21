# Architecture — llama-rs

## Overview

llama-rs is a modular Rust workspace for LLM inference with a native desktop GUI. It is a Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine for LLaMA-family large language models. The architecture targets **AMD Opteron 3280** (bdver1, 8 cores, no FMA) and **NVIDIA GTX 1050** (2GB VRAM, Compute 6.1), and uses GGUF v3 as the model format.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Rust, edition 2024, MSRV 1.85 |
| Parallelism | `rayon` (data-parallel), `std::thread::scope` (task-parallel), `std::sync::Arc` |
| SIMD | AVX (8-wide) → SSE4.2 (4-wide) → scalar fallback (no FMA, no AVX2) |
| GPU | `cudarc` bindings, cuBLAS matmul (enabled by default) |
| Hardware Abstraction | `Backend` trait + compile-time plugin registry (see [Plugin System](#9-plugin-system)) |
| Serialization | `byteorder` (LE binary), `memmap2` (memory-mapped I/O) |
| Async | `tokio` + `axum` + `futures` (server binary only) |
| CLI | `clap` v4 derive |
| Errors | `thiserror` (libs), `anyhow` (binaries) |
| Benchmarking | `criterion` with HTML reports |
| Testing | `#[cfg(test)] mod tests` + integration tests in `tests/` |

## Directory Structure

```
llama-rs/
├── Cargo.toml                  # Workspace root (16 members)
├── rustfmt.toml                # max_width=100, tab_spaces=4, reorder_imports
├── deny.toml                   # License policy (MIT, Apache-2.0, Unlicense)
├── .cargo/config.toml          # --target-cpu=bdver1
├── .github/workflows/ci.yml    # format → clippy → test → deny → doc
├── crates/
│   ├── error/                  # Unified error handling — Error enum, Result<T> alias
│   ├── config/                 # Configuration management — Config struct, env-based loading
│   ├── gguf/                   # GGUF v3 file parser with memory-mapped I/O
│   ├── ggml/                   # Core tensor types + computation graph + Backend trait (depends on nothing)
│   ├── ggml-cpu/               # CPU backend: implements Backend trait, SIMD matmul (depends on ggml)
│   ├── ggml-cuda/              # CUDA backend: implements Backend trait, cuBLAS (depends on ggml, requires CUDA toolkit)
│   ├── llama-core/             # Core inference traits and shared types
│   ├── llama/                  # Inference engine: factory + dispatch via Arc<dyn Backend> (depends on gguf, ggml, ggml-cpu, ggml-cuda)
│   ├── common/                 # Shared utils: args, sampling config, chat templates
│   ├── llama-cli/              # CLI binary with --backend flag (depends on llama, common)
│   ├── llama-server/           # HTTP server with --backend flag (depends on llama, common, axum)
│   ├── llama-ui-core/          # Shared UI types, theme, and error types
│   ├── llama-ui-models/        # Model discovery, manifest, GGUF metadata extraction
│   ├── llama-ui-session/       # Chat history, session persistence, export (JSON/MD/plain)
│   ├── llama-ui-sandbox-client/ # Sandbox server spawning with resource limits
│   └── llama-ui/               # Desktop GUI (iced 0.13) — multi-pane chat interface
├── test-models/                 # Test GGUF files (gitignored, downloaded separately)
├── docs/                        # Additional documentation
├── debian/                      # Debian packaging files
└── media/                       # Screenshots, diagrams
```

The architecture enforces strict separation of concerns across 16 crates, organized into three layers:

1. **Foundation** — GGUF parsing, tensor operations, compute backends, core inference traits
2. **Inference** — LLM inference engine, sampling, KV cache management
3. **Application** — CLI, HTTP server, desktop GUI, session management, shared UI types

## Crate Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
├─────────────────────────────────────────────────────────────────┤
│ llama-ui (iced GUI)                                             │
│   ├─ llama-ui-models (model discovery)                          │
│   ├─ llama-ui-session (chat history)                            │
│   ├─ llama-ui-sandbox-client (server spawning)                  │
│   └─ common (sampling config, chat templates)                   │
│                                                                 │
│ llama-cli (interactive CLI)                                     │
│   └─ llama (inference engine)                                   │
│                                                                 │
│ llama-server (HTTP REST API)                                    │
│   ├─ llama (inference engine)                                   │
│   └─ common (sampling config)                                   │
├─────────────────────────────────────────────────────────────────┤
│ Inference Layer                                                 │
├─────────────────────────────────────────────────────────────────┤
│ llama (transformer forward pass, attention, KV cache)           │
│   ├─ ggml (tensor operations)                                   │
│   ├─ ggml-cpu (CPU backend)                                     │
│   ├─ ggml-cuda (CUDA backend)                                   │
│   ├─ gguf (model loading)                                       │
│   └─ common (sampling, error handling)                          │
├─────────────────────────────────────────────────────────────────┤
│ Foundation Layer                                                │
├─────────────────────────────────────────────────────────────────┤
│ gguf (GGUF v3 parser)                                           │
│   └─ error (error types)                                        │
│                                                                 │
│ ggml (tensor library)                                           │
│   └─ error (error types)                                        │
│                                                                 │
│ ggml-cpu (CPU backend)                                          │
│   ├─ ggml (tensor operations)                                   │
│   └─ error (error types)                                        │
│                                                                 │
│ ggml-cuda (CUDA backend)                                        │
│   ├─ ggml (tensor operations)                                   │
│   └─ error (error types)                                        │
│                                                                 │
│ common (shared utilities)                                       │
│   ├─ error (error types)                                        │
│   └─ config (configuration)                                     │
│                                                                 │
│ config (configuration management)                               │
│   └─ error (error types)                                        │
│                                                                 │
│ error (unified error handling)                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Inference Pipeline (CLI / Server)

```
User Input
    ↓
[llama-cli / llama-server]
    ↓
Load Model (gguf parser)
    ↓
Create InferenceContext
    ↓
Tokenize Prompt (llama::encode)
    ↓
Generate Tokens Loop:
    ├─ Forward Pass (llama::forward_pass)
    │   ├─ Embedding lookup
    │   ├─ Transformer blocks (attention + FFN)
    │   └─ Output projection
    ├─ Sample Next Token (llama::sample_logits)
    │   ├─ Apply temperature
    │   ├─ Apply top-k / top-p
    │   ├─ Apply repeat penalty
    │   └─ Sample from distribution
    └─ Append to KV cache
    ↓
Output Token Stream
```

### Desktop GUI Pipeline (llama-ui)

```
User Opens llama-ui
    ↓
[llama-ui] Load Config (UiConfig)
    ↓
Scan Models (llama-ui-models)
    ├─ Discover GGUF files
    ├─ Extract metadata (architecture, context length)
    └─ Build manifest
    ↓
Display Model List
    ↓
User Selects Model + Pane
    ↓
[llama-ui-sandbox-client] Spawn Server
    ├─ Resolve llama-server binary
    ├─ Apply resource limits (cgroup/systemd-run)
    └─ Wait for /health endpoint
    ↓
User Types Message
    ↓
[llama-ui] Send /completion Request
    ├─ Tokenize prompt via /tokenize
    ├─ Check context overflow (80% warning, 95% alert)
    ├─ Stream tokens via SSE (streaming mode)
    └─ Or non-streaming block completion
    ↓
[llama-ui] Render Tokens in Real-Time
    ├─ Update message in active pane
    ├─ Auto-scroll to bottom
    ├─ Update context usage progress bar
    └─ Update token count
    ↓
User Saves Session
    ↓
[llama-ui-session] Export
    ├─ JSON (full metadata)
    ├─ Markdown (formatted chat)
    └─ Plain text (messages only)
```

## Per-Crate Breakdown

### Foundation Layer

#### `gguf` (GGUF v3 Parser)
- **Responsibility**: Parse GGUF file format, extract tensors, metadata
- **Key Types**: `GgufFile`, `Tensor`, `MetadataValue`
- **Dependencies**: `error`
- **Tests**: Unit tests for parsing, dequantization, metadata extraction

#### `ggml` (Tensor Library)
- **Responsibility**: Core tensor operations, computation graphs, dtype support
- **Key Types**: `Tensor`, `DType`, `ComputeGraph`
- **Dependencies**: `error`
- **Tests**: Unit tests for tensor operations, dtype conversions

#### `ggml-cpu` (CPU Backend)
- **Responsibility**: AVX/SSE4.2 SIMD matmul, block-tiling, CPU inference
- **Key Functions**: `matmul_f32`, `matmul_q4`, `dot_product`
- **Dependencies**: `ggml`, `error`
- **Tests**: Benchmarks, correctness tests, SIMD verification

#### `ggml-cuda` (CUDA Backend)
- **Responsibility**: cuBLAS matmul, VRAM tracking, GPU inference
- **Key Functions**: `matmul_cuda`, `vram_available`
- **Dependencies**: `ggml`, `error`, `cudarc`
- **Tests**: GPU-only tests (skipped if CUDA unavailable)

#### `error` (Unified Error Handling)
- **Responsibility**: Define error types, implement `Display` and `Error` traits
- **Key Types**: `Error` enum with variants: `Io`, `Config`, `Gguf`, `Network`, `Parse`, `Template`, `GgufMeta`, `Other`
- **Dependencies**: None (foundational)
- **Tests**: Unit tests for error formatting

#### `llama-core` (Core Inference Traits)
- **Responsibility**: Core inference traits and shared types used across inference and application layers
- **Key Types**: `Model`, `InferenceContext`, `KvCache`, `SamplingConfig` (re-exported)
- **Dependencies**: `error`
- **Tests**: Unit tests for trait definitions and shared types

#### `config` (Configuration Management)
- **Responsibility**: Load/save configuration from environment and TOML
- **Key Types**: `Config`, `UiConfig`
- **Dependencies**: `error`
- **Tests**: Unit tests for config loading, TOML serialization

#### `common` (Shared Utilities)
- **Responsibility**: Sampling config, chat templates, argument parsing
- **Key Types**: `SamplingConfig`, `ChatTemplate`
- **Key Functions**: `render_chat_template`, `get_builtin_template`
- **Dependencies**: `error`, `config`, `minijinja`
- **Tests**: Unit tests for sampling, template rendering

### Inference Layer

#### `llama` (Inference Engine)
- **Responsibility**: Transformer forward pass, attention, KV cache, token generation
- **Key Types**: `Model`, `InferenceContext`, `KvCache`
- **Key Functions**: `generate`, `forward_pass`, `sample_logits`, `encode`
- **Dependencies**: `gguf`, `ggml`, `ggml-cpu`, `ggml-cuda`, `common`, `error`
- **Tests**: Unit tests for inference, attention, KV cache; integration tests with tiny models

### Application Layer

#### `llama-cli` (Interactive CLI)
- **Responsibility**: Command-line interface for text generation
- **Key Functions**: `main`, argument parsing, prompt loop
- **Dependencies**: `llama`, `common`, `error`
- **Tests**: Integration tests with tiny models

#### `llama-server` (HTTP REST API)
- **Responsibility**: HTTP server with `/completion` (SSE + non-streaming), `/health`, `/tokenize`, `/samplers`
- **Key Endpoints**:
  - `POST /completion` — Generate tokens (streaming or non-streaming)
  - `GET /health` — Server health check
  - `POST /tokenize` — Tokenize prompt
  - `POST /samplers` — Update sampling parameters
  - `GET /v1/models` — List available models
- **Dependencies**: `llama`, `common`, `error`, `tokio`, `axum`, `tower`
- **Tests**: Integration tests with HTTP client

#### `llama-ui` (Desktop GUI)
- **Responsibility**: Multi-pane chat interface, model management, session persistence
- **Key Components**:
  - `app.rs` — Main application state machine, update/view/subscription logic
  - `theme.rs` — Custom button style module (delegates to `llama_ui_core::theme`)
- **Key Features**:
  - Streaming & non-streaming modes per pane
  - Context usage progress bar with color-coded warnings
  - Clear chat, session export/import, model browsing
  - Per-pane backend selection (auto/cpu/cuda)
  - Per-pane resource limits (memory/CPU)
- **Dependencies**: `llama-ui-models`, `llama-ui-session`, `llama-ui-sandbox-client`, `common`, `error`, `iced`, `tokio`
- **Tests**: Unit tests for model picker rendering

#### `llama-ui-core` (Shared UI Types)
- **Responsibility**: Shared UI types, theme, and error types for the desktop GUI
- **Key Types**: Theme configuration, UI error types, shared component types
- **Dependencies**: `error`
- **Tests**: Unit tests for theme and type definitions

#### `llama-ui-models` (Model Discovery)
- **Responsibility**: Scan for GGUF files, extract metadata, build manifest
- **Key Types**: `ModelEntry`, `Manifest`
- **Key Functions**: `scan_models`, `load_manifest`, `extract_metadata`
- **Dependencies**: `gguf`, `error`
- **Tests**: Unit tests for model scanning, metadata extraction

#### `llama-ui-session` (Chat History & Export)
- **Responsibility**: Manage chat messages, export to JSON/Markdown/plain text
- **Key Types**: `Session`, `ChatMessage`, `Role`
- **Key Functions**: `add_message`, `export_json`, `export_markdown`, `export_plain`
- **Dependencies**: `common`, `error`
- **Tests**: Unit tests for session management, export formats

#### `llama-ui-sandbox-client` (Sandbox Server Spawning)
- **Responsibility**: Spawn llama-server subprocess with resource limits
- **Key Types**: `SandboxClient`, `ResourceLimits`
- **Key Functions**: `spawn`, `wait_for_ready`, `health_check`
- **Dependencies**: `error`, `tokio`, `reqwest`, `nix`
- **Tests**: Unit tests for spawn logic, resource limit application

## Key Design Decisions

### 1. Strict Layering
- Foundation crates (`gguf`, `ggml`, `error`) have zero internal dependencies
- Inference layer (`llama`) depends only on foundation
- Application layer can depend on anything, but not vice versa
- **Benefit**: Easy to test, reuse, and replace components

### 2. Unified Error Handling
- All crates use `error::Error` enum
- Libraries return `Result<T, error::Error>`
- Binaries use `anyhow::Result` for convenience
- **Benefit**: Consistent error propagation, easy debugging

### 3. Sampling Configuration
- `common::SamplingConfig` is the canonical type
- Shared across CLI, server, and GUI
- Serializable for session persistence
- **Benefit**: Consistent sampling behavior across all interfaces

### 4. Chat Templates
- `common::render_chat_template` supports multiple formats (ChatML, Llama, Gemma, StableLM)
- Uses `minijinja` for template rendering
- Shared across CLI, server, and GUI
- **Benefit**: Consistent prompt formatting, easy to add new templates

### 5. Sandbox Isolation
- `llama-ui-sandbox-client` spawns `llama-server` as a subprocess
- Optional cgroup resource limits (memory, CPU)
- Graceful fallback if cgroups unavailable
- **Benefit**: Safe multi-model inference, resource protection

### 6. Subscription-Based Streaming
- `llama-ui` uses `iced::Subscription` for SSE streaming
- Non-blocking token rendering
- Keyboard input via `iced::keyboard::on_key_press`
- **Benefit**: Responsive UI, no blocking I/O

## Testing Strategy

### Unit Tests
- Inline in source files under `#[cfg(test)] mod tests { ... }`
- Test individual functions, types, and error cases
- Run with `cargo test --lib`

### Integration Tests
- Located in `crates/<name>/tests/<name>_test.rs`
- Test cross-crate interactions, end-to-end workflows
- Run with `cargo test --test <name>`

### Benchmarks
- Located in `crates/<name>/benches/<name>.rs`
- Use `criterion` crate for statistical analysis
- Run with `cargo bench -p <crate>`

### Doctests
- In doc comments (`/// ```no_run ...`)
- Test API examples, document behavior
- Run with `cargo test --doc`

### CI Pipeline
- Format check: `cargo fmt --all -- --check`
- Lint: `cargo clippy --workspace -- -D warnings`
- Tests: `cargo test --workspace --verbose`
- License audit: `cargo deny check licenses`
- Documentation: `cargo doc --no-deps --document-private-items`

## Performance Considerations

### CPU Backend
- AVX/SSE4.2 SIMD matmul with block-tiling
- Rayon for data parallelism (8 cores)
- Typical speedup: 2.5–3.2x for large matrices

### CUDA Backend
- cuBLAS matmul for GPU acceleration
- VRAM tracking to prevent OOM
- Typical speedup: 10–50x for large matrices (GTX 1050)

### KV Cache
- Prefix caching for repeated prompts
- O(1) reset for new conversations
- Push-batch strategy for efficient memory layout

### Inference
- Token generation: ~84µs per token including sampling (13M params, CPU)
- Streaming: Real-time token rendering in GUI

## Future Improvements

1. **Quantization** — Support more quantization formats (GGIQ, IQ2_XXS)
2. **Multi-GPU** — Distribute inference across multiple GPUs
3. **Speculative Decoding** — Use smaller model to predict next tokens
4. **Continuous Batching** — Batch multiple requests for higher throughput
5. **Model Merging** — Combine multiple models for ensemble inference
