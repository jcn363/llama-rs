# Contributing to llama-rs

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/jcn363/llama-rs.git
   cd llama-rs
   ```

2. **Install Rust** (stable toolchain, version >= 1.85). Follow the instructions at <https://rustup.rs/>.
3. **Ensure the required toolchain components are installed**

   ```bash
   rustup component add rustfmt clippy
   ```

## Building

The workspace contains multiple crates. To build everything in debug mode:

```bash
cargo build --workspace --verbose
```

For an optimized release build:

```bash
cargo build --release --workspace
```

CUDA support is enabled by default (requires CUDA toolkit). To build without CUDA:

```bash
cargo build --release --no-default-features -p ggml-cuda
```

## Testing

Run the full test suite (unit tests and doctests):

```bash
cargo test --workspace --verbose
cargo test --workspace --doc
```

Benchmarks are defined using the `criterion` crate and can be executed with:

```bash
cargo bench -p ggml-cpu --bench cpu_bench
```

## Linting

The project enforces formatting and linting via CI. To check locally:

```bash
# Formatting check (fails if code is not formatted)
cargo fmt --all -- --check

# Lint with clippy (treats warnings as errors)
cargo clippy --workspace -- -D warnings
```

## Code Conventions

Before writing code, familiarize yourself with the project's conventions:

- **[`CODE_STYLE.md`](./CODE_STYLE.md)** — code organization, naming, error handling, unsafe rules, testing patterns
- **[`docs/RBP.md`](./docs/RBP.md)** — broader Rust best practices (borrowing, error handling, async, workspace management, API guidelines)
- **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — crate dependency graph, data flow, per-crate file breakdown

### Commit Messages

See `README.md#Commit-Guidelines` for the project's commit message format. In short:

- Format: `phase [N]: [description]` for feature phases, plain titles for fixes/refactors
- Keep commits focused on a single logical concern
- Messages must be imperative and descriptive
- All commits must pass `cargo fmt --all -- --check` and `cargo clippy --workspace -- -D warnings`

## License Policy

License compliance is managed with **deny**. The policy is defined in `deny.toml`. To audit licenses locally:

```bash
cargo deny check licenses
```

Only dependencies that satisfy the allowances in `deny.toml` may be used.

---

Thank you for contributing! Please make sure your changes pass all checks before opening a pull request.
