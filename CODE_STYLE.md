# llama-rs Code Style Guide

This documents the conventions and patterns used throughout the codebase. Follow these when contributing.

## Naming Conventions

| Category | Convention | Example |
|----------|-----------|---------|
| **Files** | `snake_case.rs` | `kv_cache.rs`, `cpu_features.rs` |
| **Directories** | `snake_case` | `ggml-cpu/`, `llama-server/` |
| **Types/Structs** | `PascalCase` | `InferenceContext`, `KvCacheManager`, `MmapTensor` |
| **Traits** | `PascalCase` | (none currently defined in codebase) |
| **Enums & Variants** | `PascalCase` | `GgmlType::F32`, `CudaError::OutOfMemory` |
| **Functions** | `snake_case` | `multi_head_attention_with_cache()`, `sample_logits()` |
| **Methods** | `snake_case` | `tensor.byte_size()`, `model.summary()` |
| **Variables** | `snake_case` | `n_layers`, `head_dim`, `max_seq_len` |
| **Constants** | `SCREAMING_SNAKE_CASE` | `GGUF_MAGIC`, `GGUF_DEFAULT_ALIGNMENT`, `BLOCK_M` |
| **Module-level** | `snake_case` | `pub mod tensor;` |
| **Private items** | Leading `_` for intentionally unused | `_j_start`, `_layer_total` |

## File Organization

### Crate Structure

Each crate follows a consistent pattern in `lib.rs`:
1. Doc comment (`//!`) — crate-level documentation
2. Module declarations (`mod xxx;`)
3. Re-exports (`pub use ...;`)
4. Main struct definitions
5. Helper functions
6. Tests (`#[cfg(test)] mod tests { ... }`)

```rust
//! Crate-level doc comment

// ─── Module declarations ─────────────────────────────────────────────
mod module_a;
mod module_b;

// ─── Re-exports ─────────────────────────────────────────────────────
pub use module_a::ItemA;
pub use module_b::ItemB;

// ─── Main struct ────────────────────────────────────────────────────
pub struct MainStruct { ... }

// ─── Implementation ─────────────────────────────────────────────────
impl MainStruct { ... }

// ─── Helpers ────────────────────────────────────────────────────────
fn helper() { ... }

// ─── Tests ──────────────────────────────────────────────────────────
#[cfg(test)]
mod tests { ... }
```

### Section Separators

Use ASCII section comments with em-dashes (`───`) to visually separate sections:

```rust
// ─── Imports ─────────────────────────────────────────────────────────
// ─── Constants ───────────────────────────────────────────────────────
// ─── Public API ──────────────────────────────────────────────────────
// ─── Tests ──────────────────────────────────────────────────────────
```

## Import Style

- Imports are **reordered** automatically by rustfmt (`reorder_imports = true`)
- Group standard library first, then external crates, then internal crates
- Use `use` over `extern crate`
- Prefer nested imports over long import paths:

```rust
// ✅ Good
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use gguf::{GgufError, GgufReader, GgufValue};
use rayon::prelude::*;

use crate::kv_cache::KvCacheManager;
use crate::{InternedStrings, Model, TensorData};

// ❌ Avoid
use std::sync::Arc;
use std::sync::RwLock;
```

## Code Patterns

### Struct Definition + `impl` Block

Define the struct in `lib.rs`, place `impl` blocks in dedicated files:

- `lib.rs` — struct definition + top-level re-exports
- `model.rs` — `impl Model { ... }`
- `reader.rs` — `impl GgufReader { ... }`

```rust
// lib.rs
pub struct Model {
    pub tensors: HashMap<usize, TensorData>,
    pub n_embd: usize,
    // ...
}

// model.rs
impl Model {
    pub fn load_from_gguf(path: impl AsRef<Path>) -> Result<Self, GgufError> { ... }
    pub fn summary(&self) -> String { ... }
}
```

### Error Handling

- **Libraries**: use `thiserror` for `#[derive(Error)]` enums with `GgufResult<T>` type aliases:

  ```rust
  #[derive(Debug, Error)]
  pub enum GgufError {
      #[error("IO error: {0}")]
      Io(#[from] io::Error),
      #[error("invalid GGUF magic number")]
      InvalidMagic,
      #[error("decode error: {0}")]
      DecodeError(String),
  }
  pub type GgufResult<T> = Result<T, GgufError>;
  ```

- **Binaries**: use `anyhow::Result` / `anyhow::Error` in `main()` and handlers
- Use `.map_err(|e| ...)` for error conversion, not `.unwrap()` or `.expect()` in library code
- `unwrap()` / `expect()` only in tests or where failure is unrecoverable (e.g., lock poisoning)

### Lock Poisoning

Always handle `RwLock` / `Mutex` poisoning explicitly with `expect("lock poisoned")`:

```rust
let data = *self.data.read().expect("lock poisoned");
```

### Thread Safety

- Use `Arc<Model>` to share across threads
- Use `RwLock` for read-mostly state (KV cache, tensor data cache)
- Use `Mutex` for exclusive access (interned strings during parallel loading)
- Use `rayon::prelude::*` for data-parallel operations (dequantization, tensor loading)
- Use `std::thread::scope` for scoped thread pools (matmul)

### Parallelism Thresholds

```rust
// Small matrices: sequential is faster
if rows < 64 { /* sequential */ } else { /* parallel */ }

// Vector operations: parallel above 1024 elements
if len < 1024 { /* sequential */ } else { /* parallel */ }

// Dequantization: parallel above 64K elements
if num_elements > 65536 { /* parallel */ } else { /* sequential */ }
```

### Type Aliases

```rust
pub type CudaResult<T> = Result<T, CudaError>;
pub type GgufResult<T> = Result<T, GgufError>;
```

### Builder Pattern (Limited)

```rust
pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
    self.sampling = sampling;
    self
}
```

### Default Implementations

```rust
impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
        }
    }
}
```

### `#[must_use]` Annotation

Used on functions that return values where discarding the result is likely an error:

```rust
#[must_use]
pub fn is_available(&self) -> bool { ... }

#[must_use]
pub fn total_vram(&self) -> usize { ... }
```

## Error Handling Patterns

| Pattern | Where Used | Example File |
|---------|-----------|-------------|
| `#[derive(Error)]` enum | Library crates | `gguf/src/errors.rs` |
| `anyhow::Result` | Binaries, `main()` | `llama-cli/src/main.rs` |
| `#[from]` attribute | Wrapping std errors | `gguf/src/errors.rs` |
| `expect("lock poisoned")` | RwLock/Mutex access | `crates/llama/src/lib.rs` |
| Unwrap only in tests | Test code | All `#[cfg(test)] mod tests` |
| `#[should_panic]` | Testing panics | `ggml-cpu/src/lib.rs` |

## Logging

- Uses the `tracing` crate (not `log`)
- Initialized in binaries with `tracing_subscriber`
- Info-level logging for major events (model loading, request handling)
- Format: `tracing::info!("Loading model from: {}", args.model);`
- Server uses `EnvFilter` for runtime log level control

```rust
tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::new("info"))
    .init();

tracing::info!("Completion request: prompt_len={}", request.prompt.len());
```

## Testing

### Test Organization

- **Unit tests**: Inline in source files under `#[cfg(test)] mod tests { }`
- **Integration tests**: `crates/<name>/tests/` directory
- **Doctests**: In doc comments (`/// ```no_run ...`)

### Test Patterns

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name() {
        // Arrange
        let input = ...;
        let expected = ...;

        // Act
        let result = function(input);

        // Assert
        assert!(condition);
        assert_eq!(result, expected);
        assert!((result - expected).abs() < 0.001);  // Float comparison
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match")]
    fn test_invalid_input() { ... }
}
```

### Test File Naming

- Integration test files: `<name>_test.rs` in `tests/` directory
- Benchmarks: `<name>.rs` in `benches/` directory

### Benchmark Patterns

```rust
use criterion::{Criterion, criterion_group, criterion_main};

fn my_benchmark(c: &mut Criterion) {
    c.bench_function("operation_name", |b| {
        b.iter(|| {
            // code to benchmark
        })
    });
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
```

### Conditional Test Execution

For tests that require external resources (model files):

```rust
#[test]
fn test_needs_model_file() {
    let model_path = ...;
    if !model_path.exists() {
        println!("Skipping: test model not found");
        return;  // Skip gracefully, not panic
    }
    // ... actual test
}
```

## Feature Gates

CUDA code is conditionally compiled:

```rust
#[cfg(feature = "cuda")]
{
    // Real CUDA implementation
}

#[cfg(not(feature = "cuda"))]
{
    // Stub that returns available=false
}
```

## Do's and Don'ts

### ✅ Do
- Use `thiserror` for library error enums
- Use `anyhow` for binary error handling
- Put struct definitions in `lib.rs`, impls in named files
- Use `Arc` for shared ownership across threads
- Use `RwLock` for read-mostly concurrent access
- Add `#[must_use]` to pure accessors
- Put tests in `#[cfg(test)] mod tests` blocks
- Use `#[expect(dead_code)]` for intentionally unused items (not `#[allow(dead_code)]`)

### ❌ Don't
- Don't `unwrap()` in library code (use `?` + `map_err`)
- Don't use `#[allow(...)]` without a comment explaining why (use `#[expect(...)]` if possible)
- Don't mix `tracing` and `log` — use `tracing` consistently
- Don't add dependencies without checking `deny.toml` license policy
- Don't use `unsafe` without `// Safety: ...` comment
- Don't hardcode architecture-specific constants — detect at runtime or use `#[cfg(...)]`

## Formatting Rules

Enforced by `cargo fmt` via `rustfmt.toml`:
- `max_width = 100`
- `tab_spaces = 4` (tab width in spaces — rustfmt uses spaces for indentation, not tabs)
- `newline_style = "Unix"` (LF)
- `reorder_imports = true`
- `reorder_modules = true`

## Clippy Lints

The workspace sets pedantic lints to `allow` by default in `Cargo.toml`:
```toml
[workspace.lints.clippy]
pedantic = { level = "allow", priority = -1 }
```

Individual crates enable strict linting with:
```rust
#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(
    clippy::many_single_char_names,
    clippy::wildcard_imports,
    // ... other allowed lints
)]
```

CI enforces `cargo clippy --workspace -- -D warnings` — treat all warnings as errors.
