# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- `config` crate with `Config` struct and `from_env()` for env-based configuration.
- `error` crate with `Error` enum and `Result<T>` alias for unified error handling.
- `common::args::CommonArgs` struct for DRY argument parsing across binaries.
- Property-based tests (`proptest`) for config parsing and error formatting.
- Integration tests for `config` and `error` crates under `tests/` directories.
- CI badges in README (Linux, macOS, Windows workflow status).
- GUI: Streaming/non-streaming mode toggle per pane.
- GUI: Clear chat button to reset conversation history.
- GUI: Browse for GGUF file button in model picker.
- GUI: Context usage progress bar with color-coded indicators.
- GUI: Version display in settings panel.
- GUI: Enhanced settings panel with feature reference and keyboard shortcuts.
- GUI: Improved loading and error screens with branding.

### Changed
- Workspace `Cargo.toml`: added `[workspace]` section, `[workspace.package]`, and `[workspace.dependencies]`.
- `llama-cli` and `llama-server` now use `common::args::CommonArgs` via `#[clap(flatten)]`.
- Both binaries depend on `config` and `error` crates.
- `CpuBackend` implements `Backend` trait with proper `matmul` computation.
- `ci.yml`: replaced `actions-rs/toolchain@v1` with `dtolnay/rust-toolchain@stable`.
- `ci-macos.yml` and `ci-windows.yml`: same dtolnay migration + bumped `actions/checkout` to v4.
- `ggml` doc comments fixed to avoid unresolved-link warnings.
- `ggml-cpu` and `gguf` doc comments fixed for math expressions (backtick-wrapped).
- `#![deny(missing_docs)]` added to `config` and `error` crates.
- `IMPRO.md` updated with accurate status of all phases.
- README updated with new crate descriptions, usage examples, and CI badges.
- GUI: StartChat now uses the selected model instead of always model[0].
- GUI: Improved model picker with model selection highlighting and browse button.
- GUI: Enhanced pane header showing model name and controls.
- GUI: Send button shows streaming mode indicator.
- GUI: Improved chat message display with "Generating..." placeholder for in-progress responses.
- Documentation: Fixed workspace member count (16 crates).
- Documentation: Fixed ARCHITECTURE.md directory structure to list all crates.
- Documentation: Fixed CONTRIBUTING.md with correct repository URL.

### Fixed
- `crates/ggml/src/lib.rs` — added `pub mod backend;` to expose `Backend` trait.
- `crates/ggml-cpu/src/backend.rs` — fixed `matmul` to compute with `matmul_f32` instead of returning zeros.
- `crates/ggml-cuda/Cargo.toml` — updated `cudarc` to v0.19.7 with correct features.
- `crates/config/src/lib.rs` — fixed duplicate `pub use Config` re-exports.
- `crates/error/src/lib.rs` — fixed duplicate `pub use Error` re-exports.
- `crates/ggml/src/improvements.rs` — removed duplicate nested function definitions, fixed clippy lints.
- `crates/llama/tests/inference_context_test.rs` — `test_load_real_gguf_model` now uses correct path.
- `crates/llama-ui-models/src/lib.rs` — removed duplicate `llama.context_length` key in metadata extraction.
- `Cargo.toml` — fixed inconsistent indentation on `llama-ui-core` workspace dependency.
- `.cargo/config.toml` — created missing build config for bdver1 target CPU.
- `crates/ggml-cpu/benches/cpu_bench.rs` — added missing SAFETY comment on unsafe block.
- `crates/ggml-cpu/src/matmul.rs` — fixed SAFETY comment to use standard format.
- `crates/ggml-cuda/src/lib.rs` — added missing SAFETY comment on cuMemGetInfo_v2 call.
- `.gitignore` — un-ignored `.cargo/config.toml` so it can be tracked.
- `crates/common/src/lib.rs` — exported `chat_templates` module.

## [0.1.0] - 2026-05-25
- Initial release of `llama-rs` workspace with core crates and binaries.
