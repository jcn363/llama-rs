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

### Fixed
- `crates/ggml/src/lib.rs` — added `pub mod backend;` to expose `Backend` trait.
- `crates/ggml-cpu/src/backend.rs` — fixed `matmul` to compute with `matmul_f32` instead of returning zeros.
- `crates/ggml-cuda/Cargo.toml` — updated `cudarc` to v0.19.7 with correct features.
- `crates/config/src/lib.rs` — fixed duplicate `pub use Config` re-exports.
- `crates/error/src/lib.rs` — fixed duplicate `pub use Error` re-exports.
- `crates/ggml/src/improvements.rs` — removed duplicate nested function definitions, fixed clippy lints.
- `crates/llama/tests/inference_context_test.rs` — `test_load_real_gguf_model` now uses correct path.

## [0.1.0] - 2026-05-25
- Initial release of `llama-rs` workspace with core crates and binaries.
