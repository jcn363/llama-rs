# Code Style & Conventions — llama-rs

This document defines the coding standards for the llama-rs workspace. All code must follow these conventions to pass CI checks.

## Formatting

### Rustfmt Configuration
- **Max width**: 100 characters
- **Indent**: 4 spaces (no tabs)
- **Newline style**: Unix (LF)
- **Edition**: 2024

Configuration file: `rustfmt.toml`

```bash
# Check formatting (CI step)
cargo fmt --all -- --check

# Auto-format
cargo fmt --all
```

## Naming Conventions

| Category | Convention | Example |
|----------|-----------|---------|
| Files | `snake_case` | `inference.rs`, `kv_cache.rs` |
| Modules | `snake_case` | `mod inference;` |
| Functions | `snake_case` | `fn forward_pass()` |
| Variables | `snake_case` | `let token_count = 0;` |
| Constants | `SCREAMING_SNAKE_CASE` | `const MAX_TOKENS: usize = 2048;` |
| Types | `PascalCase` | `struct InferenceContext` |
| Enums | `PascalCase` | `enum Role { User, Assistant }` |
| Traits | `PascalCase` | `trait Sampler` |
| Lifetimes | `'lowercase` | `fn borrow<'a>(&'a self)` |
| Type parameters | `PascalCase` | `fn generic<T: Clone>()` |

## Error Handling

### Libraries
- Use `thiserror` for error types
- Return `Result<T, error::Error>` from public functions
- Never use `.unwrap()` or `.expect()` in library code (except tests)
- Provide context with `?` operator

```rust
// ✅ Good
pub fn load_model(path: &Path) -> Result<Model, error::Error> {
    let file = std::fs::read(path)?;
    let model = parse_gguf(&file)?;
    Ok(model)
}

// ❌ Bad
pub fn load_model(path: &Path) -> Model {
    let file = std::fs::read(path).unwrap();  // Panics!
    parse_gguf(&file).unwrap()
}
```

### Binaries
- Use `anyhow::Result` for convenience
- `.unwrap()` is acceptable in main() for fatal errors
- Provide user-friendly error messages

```rust
// ✅ Good
fn main() -> anyhow::Result<()> {
    let model = Model::load("model.gguf")?;
    println!("Model loaded successfully");
    Ok(())
}
```

### Error Types
All crates use the unified `error::Error` enum:

```rust
pub enum Error {
    Io(std::io::Error),
    Config(String),
    Gguf(String),
    Network(String),
    Parse(String),
    Template(String),
    GgufMeta(String),
    Other(String),
}
```

## Concurrency

### Shared Ownership
- Use `Arc<T>` for shared ownership across threads
- Use `Arc<RwLock<T>>` for shared mutable state (read-mostly)
- Use `Arc<Mutex<T>>` for exclusive access

```rust
// ✅ Good — read-mostly state
let cache = Arc::new(RwLock::new(KvCache::new()));
let cache_clone = Arc::clone(&cache);
std::thread::spawn(move || {
    let mut cache = cache_clone.write().unwrap();
    cache.update(...);
});

// ❌ Bad — unnecessary mutex
let cache = Arc::new(Mutex::new(KvCache::new()));
```

### Data Parallelism
- Use `rayon` for CPU-bound parallelism
- Use `tokio` for async I/O

```rust
// ✅ Good — rayon for matmul
let result: Vec<f32> = (0..n_rows)
    .into_par_iter()
    .map(|i| compute_row(i))
    .collect();

// ✅ Good — tokio for HTTP
#[tokio::main]
async fn main() {
    let response = reqwest::get("http://...").await?;
}
```

## SIMD & Unsafe Code

### SIMD Targets
- **Primary**: AVX (bdver1 Bulldozer)
- **Fallback**: SSE4.2
- **Scalar**: Pure Rust

```rust
// ✅ Good — feature-gated SIMD
#[cfg(target_feature = "avx")]
fn matmul_avx(...) { ... }

#[cfg(target_feature = "sse4.2")]
fn matmul_sse(...) { ... }

fn matmul_scalar(...) { ... }
```

### Unsafe Code
- Every `unsafe` block must have a `// SAFETY:` comment
- Explain why the code is safe
- Minimize scope of `unsafe` blocks

```rust
// ✅ Good
// SAFETY: ptr is valid and aligned; we've verified bounds above
let value = unsafe { *ptr };

// ❌ Bad
unsafe { *ptr = value; }  // No explanation!
```

## Testing

### Test Organization
- **Unit tests**: Inline in source files under `#[cfg(test)] mod tests { ... }`
- **Integration tests**: `crates/<name>/tests/<name>_test.rs`
- **Benchmarks**: `crates/<name>/benches/<name>.rs`
- **Doctests**: In doc comments

### Test Naming
Convention: `describe_should_expected_behavior`

```rust
#[test]
fn dot_f32_should_compute_correct_result() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = dot_f32(&a, &b);
    assert_eq!(result, 32.0);
}
```

### Skipping Tests
- Skip gracefully when external resources are missing
- Don't use `#[ignore]` for CI-skipped tests

```rust
#[test]
fn inference_should_generate_tokens() {
    let model_path = "test-models/tiny-llm.gguf";
    if !Path::new(model_path).exists() {
        println!("Skipping: test model not found");
        return;
    }
    // ... test code
}
```

## Clippy & Linting

### Clippy Configuration
- No workspace-level lint config in `[workspace.lints]`
- Only `ggml-cuda` enables `#![deny(clippy::pedantic)]` at the crate level (verified: `crates/ggml-cuda/src/lib.rs` line 24)
- Individual crates may opt in or allow specific lints
- CI runs `cargo clippy --workspace -- -D warnings` which treats all warnings as errors

### CI Check
```bash
cargo clippy --workspace -- -D warnings
```

## Documentation

### Doc Comments
- Use `///` for public items
- Include examples in doc comments
- Use `no_run` for examples that require external resources

```rust
/// Compute dot product of two vectors.
///
/// # Arguments
/// * `a` — First vector
/// * `b` — Second vector
///
/// # Returns
/// Dot product as f32
///
/// # Example
/// ```no_run
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = dot_f32(&a, &b);
/// assert_eq!(result, 32.0);
/// ```
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // ...
}
```

### Module Documentation
- Document module purpose at the top of the file
- Explain key types and functions

```rust
//! KV cache management for transformer inference.
//!
//! This module provides efficient storage and retrieval of key-value pairs
//! during token generation. Supports prefix caching and O(1) reset.

pub struct KvCache { ... }
```

## GUI Code (llama-ui)

### Iced 0.13.1 Patterns

#### Application Structure
- Use function-based API (`iced::application()` builder)
- Define `Message` enum for all user actions
- Implement `update()` for state transitions
- Implement `view()` for rendering

```rust
#[derive(Debug, Clone)]
pub enum Message {
    SendMessage(String),
    ToggleSettings,
    SelectModel(String),
}

pub struct App {
    active_pane: usize,
    panes: Vec<ChatPane>,
    settings_open: bool,
}

impl App {
    fn update(&mut self, message: Message) {
        match message {
            Message::SendMessage(text) => { /* ... */ }
            Message::ToggleSettings => self.settings_open = !self.settings_open,
            Message::SelectModel(model) => { /* ... */ }
        }
    }

    fn view(&self) -> Element<Message> {
        // Render UI
    }
}
```

#### Subscriptions
- Use `iced::Subscription::run_with_id` for SSE streaming
- Use `iced::stream::channel` for async tasks
- Use `iced::keyboard::on_key_press` for keyboard input

```rust
fn subscription(&self) -> Subscription<Message> {
    // SSE streaming
    iced::Subscription::run_with_id(
        "completion_stream",
        stream_completion(self.active_pane),
    )
    .map(Message::CompletionToken)
}

fn stream_completion(pane_id: usize) -> impl Stream<Item = String> {
    // Yield tokens as they arrive
}
```

#### Tasks
- Use `Task::perform` for non-streaming async work
- Use `Task::batch` to combine multiple tasks
- Handle errors gracefully

```rust
fn load_model(&self, model_path: String) -> Task<Message> {
    Task::perform(
        async move {
            Model::load(&model_path).await
        },
        |result| match result {
            Ok(model) => Message::ModelLoaded(model),
            Err(e) => Message::Error(e.to_string()),
        },
    )
}
```

#### Keyboard Shortcuts
- Use `iced::keyboard::on_key_press` for global shortcuts
- Return `Option<Message>` to indicate if handled

```rust
fn handle_key_press(key: keyboard::Key, modifiers: keyboard::Modifiers) -> Option<Message> {
    match key {
        keyboard::Key::Named(keyboard::key::Named::Escape) => {
            Some(Message::CloseSettings)
        }
        keyboard::Key::Named(keyboard::key::Named::F11) => {
            Some(Message::ToggleFullscreen)
        }
        _ => None,
    }
}
```

## Commit Messages

### Format
- Feature phases: `phase [N]: [description]`
- Fixes/refactors: Plain title
- Keep commits focused on a single logical concern

### Examples
```
phase 1: Workspace setup with 8 crates
phase 2: Implement GGUF parser
fix: Handle edge case in KV cache reset
refactor: Extract matmul into separate module
docs: Update ARCHITECTURE.md with llama-ui section
```

### CI Requirements
All commits must pass:
- `cargo fmt --all -- --check`
- `cargo clippy --workspace -- -D warnings`
- `cargo test --workspace`
- `cargo deny check licenses`

## License

All code must comply with the license policy in `deny.toml`:
- **Allowed**: MIT, Apache-2.0, Unlicense
- **Forbidden**: GPL, AGPL, SSPL

Run `cargo deny check licenses` to audit dependencies.
