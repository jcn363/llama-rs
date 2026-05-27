# Rust MCP Filesystem Shared Documentation

This file contains the unified documentation for the Rust MCP filesystem setup and installation summary. The content was extracted from the duplicated sections in:
- `docs/others/RUST_MCP_FILESYSTEM_SETUP.md`
- `docs/others/RUST_MCP_FILESYSTEM_INSTALLATION_SUMMARY.md`

## Overview

The Rust MCP filesystem provides a virtualized file system layer that enables fast, memory‑mapped access to model files. It integrates seamlessly with the OpenAI‑style MCP server and supports both read‑only and mutable modes.

## Features
- Zero‑copy mapping of GGUF/GGML files
- Automatic cache eviction based on LRU policy
- Optional TTL (time‑to‑live) for temporary files
- Cross‑platform support (Linux, macOS, Windows)

## Setup Instructions
1. Add the `rust-mcp-filesystem` crate to your `Cargo.toml`.
2. Enable the `filesystem` feature when building the server.
3. Call `Mcp::register_filesystem()` during initialization.
4. Configure the mount point via the `LLAMA_FS_ROOT` environment variable.

## Common Pitfalls
- **TTL never cleaned** – ensure you call `Mcp::run_gc()` periodically.
- **Path collisions** – mount points must be unique across MCP instances.
- **Permission errors** – the process needs read access to the target directory.

For detailed API reference, see the crate documentation in `crates/gguf/src/reader.rs`.
