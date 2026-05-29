# Common Crate

Shared utilities and types used across the llama-rs workspace.

## Public Modules

| Module | Description |
|--------|-------------|
| `args` | Shared CLI argument parsing (`CommonArgs` struct, flattened by both `llama-cli` and `llama-server`) |
| `sampling` | `SamplingConfig` — canonical sampling parameters (temperature, top_k, top_p, repeat_penalty, seed) |
| `chat_templates` | Chat template rendering using `minijinja` (ChatML, Llama, Gemma, StableLM formats) |

## Re-exports

```rust
pub use error::Error;   // Unified error type
pub use error::Result;  // Result<T, Error> alias
```

## Key Types

```rust
// CommonArgs — shared between CLI and server
pub struct CommonArgs {
    pub model: String,
    pub threads: usize,          // 0 = auto-detect
    pub ctx_size: usize,         // default: 4096
    pub batch_size: usize,       // default: 512
    pub temperature: f32,        // default: 0.8
    pub top_k: usize,            // default: 40
    pub top_p: f32,              // default: 0.95
    pub repeat_penalty: f32,     // default: 1.1
    pub seed: Option<u64>,
    pub backend: String,         // "auto", "cpu", "cuda"
    pub cache_strategy: String,  // "full", "prefix", "prefix_only"
    pub offload_ffn: bool,
    pub memory_pool_size: usize,
}

// SamplingConfig — used by inference engine
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub seed: Option<u64>,
}
```

## Usage from Other Crates

```rust
use common::args::CommonArgs;
use common::sampling::SamplingConfig;
use common::chat_templates;
use clap::Parser;

#[derive(Parser)]
struct MyArgs {
    #[clap(flatten)]
    common: CommonArgs,
}
```

## Dependencies

- `error` crate (unified error types)
- `config` crate (configuration management)
- `clap` (argument parsing)
- `minijinja` (template rendering)
- `serde` / `serde_json` (serialization)
