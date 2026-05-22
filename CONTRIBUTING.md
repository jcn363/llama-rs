# Contributing to llama-rs

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/llama-rs.git
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

## License Policy

License compliance is managed with **deny**. The policy is defined in `deny.toml`. To audit licenses locally:

```bash
cargo deny check licenses
```

Only dependencies that satisfy the allowances in `deny.toml` may be used.

---

Thank you for contributing! Please make sure your changes pass all checks before opening a pull request.
