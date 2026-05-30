# AGENTS.md

## Workspace
- Workspace members (from `Cargo.toml`): `crates/gguf`, `crates/ggml`, `crates/ggml-cpu`, `crates/ggml-cuda`, `crates/llama-core`, `crates/llama`, `crates/common`, `crates/config`, `crates/error`, `crates/llama-cli`, `crates/llama-server`, `crates/llama-ui-models`, `crates/llama-ui-session`, `crates/llama-ui-sandbox-client`, `crates/llama-ui`, `crates/llama-ui-core`.

## Build & Test Commands
- `cargo fmt --all -- --check` – format check (CI step).
- `cargo clippy --workspace -- -D warnings` – lint with warnings treated as errors.
- `cargo build --workspace --verbose` – compile all crates (debug).
- `cargo build --release --workspace` – release build for binaries.
- `cargo test --workspace --verbose` – run all unit tests.
- `cargo test --workspace --doc` – run doctests.
- `cargo bench -p ggml-cpu --bench cpu_bench` – run CPU benchmark.
- CUDA backend is enabled by default (requires CUDA toolkit). Disable with: `--no-default-features -p ggml-cuda`.

## Binaries / Entry Points
- CLI: `./target/release/llama-cli -m model.gguf -p "prompt" -n 128`
- Server: `./target/release/llama-server -m model.gguf --host 0.0.0.0 --port 8080`
- UI: `./target/release/llama-ui`

## CI / Environment
- Runs on Ubuntu with Rust stable toolchain; installs `clippy` and `rustfmt`.
- Sets `RUSTFLAGS="-C target-cpu=bdver1"` and `CARGO_TERM_COLOR=always`.
- Caches `~/.cargo/registry`, `~/.cargo/git`, and `target` between runs.
- CI steps: format check → clippy (warnings as errors) → test → license audit → doc build.

## Formatting
- `rustfmt.toml`: `max_width = 100`, `tab_spaces = 4`, `newline_style = "Unix"`.

## Rust Version / Edition
- Workspace edition: 2024.
- Minimum Rust version: 1.85.

## Benchmarking
- Uses `criterion` crate; benchmarks under `crates/ggml-cpu/benches` and `crates/ggml/benches`.

## Persistence Tool
- `uncensored-persistence` CLI (see `.uncensored/rust_persistence/README.md`):
  - `save --name <session>`
  - `load --name <session>`
  - `list`
  - `validate --name <name>`

## Documentation References
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) – crate dependency graph, data flow, per-crate breakdown
- [`CODE_STYLE.md`](./CODE_STYLE.md) – naming, error handling, unsafe rules, testing conventions
- [`docs/RBP.md`](./docs/RBP.md) – broader Rust best practices reference
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) – build, test, lint commands; contribution workflow

## Common Gotchas
- CUDA backend is enabled by default; requires CUDA toolkit at build time.
- CI enforces `cargo fmt --all -- --check` and `cargo clippy -- -D warnings`; code must be both formatted and lint‑clean.
- Release builds require `--release` for performance‑critical runs.
