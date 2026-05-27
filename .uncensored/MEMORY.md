# MEMORY.md - Long-term Project Knowledge

## Project: llama-rs

A high-performance Rust implementation of Llama models with a focus on GGUF support and a desktop UI.

## Key Facts (from state.json)

- **RSPLAN.md:** 566-line implementation plan for llama-ui desktop app.
- **M1 (Scaffolding):** ~90% done. 4 crates exist: `llama-ui`, `llama-ui-core`, `llama-ui-models`, `llama-ui-session`.
- **M2 (Codebase Fixes):** 100% done. Unified `SamplingConfig`, fixed O(n²) streaming, implemented `repeat_penalty`.
- **M3 (Model Management):** 100% done. Download, manifest, GGUF metadata implemented.
- **M4 (Session + Templates):** 100% done. `ChatMessage`/`Session` serde, `chat_templates` module.
- **M5 (GUI Skeleton):** PARTIAL. Code exists but fails to compile due to `iced 0.13` API changes.
- **M5a (Sandbox Client):** 100% done. Spawn, health, ports, crash detection, graceful shutdown.
- **M6 (Streaming):** PARTIAL. Server streaming works, but `/tokenize` endpoint is missing and GUI is not wired.
- **M7-M14:** Not started.

## Critical Decisions

- **UI Framework:** Using `iced 0.13`.
- **Persistence:** Using `.uncensored/` directory for agent state and memory.
- **Architecture:** Modular crate structure for UI to separate core logic from models and session management.

## Learnings

- `iced 0.13` removed `iced::Command` module path and `Appearance` type, and changed the `Application` trait.
- RSPLAN needs reconciliation with the actual implementation state.
