# Project Help Overview

This file serves as a single entry point for all **helpful documentation** within the `llama-rs` repository. It links to the most relevant guides, troubleshooting resources, and agent-specific instructions.

---

## Binary Help (Comprehensive)

| Binary | Help Document | Description |
|--------|---------------|-------------|
| **llama-cli** | [CLI_HELP.md](CLI_HELP.md) | Command-line text generation — flags, interactive mode, sampling |
| **llama-server** | [SERVER_HELP.md](SERVER_HELP.md) | HTTP server — REST API endpoints, SSE streaming, cURL/Python examples |
| **llama-ui** | [UI_HELP.md](UI_HELP.md) | Desktop GUI — screens, controls, session management, theme |

---

## General Project Docs

- **[README.md](../README.md)** — High-level overview, build & install instructions, UI features.
- **[CODE_STYLE.md](../CODE_STYLE.md)** — Formatting, naming conventions, and linting requirements.
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — How to contribute, set up the development environment, and run CI checks.
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** — Detailed architectural diagram and crate dependency graph.
- **[SECURITY.md](../SECURITY.md)** — Security policy and vulnerability reporting.
- **[CHANGELOG.md](../CHANGELOG.md)** — Release history.

## Development Guides

- [docs/build.md](build.md) — Build system details and cross-platform notes.
- [docs/USAGE.md](USAGE.md) — Running the CLI, server, and UI.
- [docs/development/parsing.md](development/parsing.md) — Extending the GGUF parser.
- [docs/development/debugging-tests.md](development/debugging-tests.md) — Debugging strategies and test patterns.
- [docs/development/HOWTO-add-model.md](development/HOWTO-add-model.md) — Adding new GGUF models to the UI.
- [docs/development/token_generation_performance_tips.md](development/token_generation_performance_tips.md) — Performance tuning.

## Backend & Performance

- [docs/backend/](backend/) — Backend-specific configuration (CUDA, OpenCL, VirtGPU, etc.).
- [docs/common/backend_shared.md](common/backend_shared.md) — Shared backend documentation.
- [docs/multimodal/](multimodal/) — Multimodal model support guides.

## Architecture & Reference

- [docs/architecture.md](architecture.md) — High-level architecture with Mermaid diagrams.
- [docs/architecture_and_crates.md](architecture_and_crates.md) — Crate dependency graph and rationale.
- [docs/RBP.md](RBP.md) — Rust best practices (borrowing, error handling, async, testing).
- [docs/ICED_API.md](ICED_API.md) — Iced 0.13 API reference for the UI.
- [docs/RSPLAN.md](RSPLAN.md) — Full llama-ui implementation plan (M0–M14).
- [docs/MARKET.md](MARKET.md) — Market analysis and competitive positioning.

## Tools & Integration

- [docs/install.md](install.md) — Pre-built packages (Winget, Homebrew, Nix, Debian).
- [docs/docker.md](docker.md) — Docker images and build instructions.
- [docs/common/octocode.md](common/octocode.md) — OctoCode integration.
- [docs/common/rust_mcp_filesystem.md](common/rust_mcp_filesystem.md) — Rust MCP filesystem integration.

---

## How to Use This Help

1. **Start here** — Open `HELP.md` to locate the topic you need.
2. **For binary-specific help** — Click the links in the "Binary Help" table above for comprehensive flag references, API docs, and examples.
3. **For development** — Follow the Development Guides for build, debug, and contribution workflows.
4. **For architecture questions** — See the Architecture & Reference section for crate graphs and design decisions.
5. **When contributing** — Always keep this file up-to-date: add new sections for any new major documentation.

---

*Last updated: 2026-05-29*
