# Architecture & Major Crates

| Component | Crate / Library | Rationale |
|-----------|----------------|-----------|
| Core LLM inference | `crates/llama` | Existing — GGUF loader, sampler, inference loop. |
| HTTP/REST sandbox IPC | `crates/llama-server` (extended in M2) | Already has `/completion` (SSE + non‑streaming), `/health`, `/v1/models`, CORS, graceful shutdown, `tokens_per_sec` metrics, structured error responses, CLI sampler params, `CacheStrategy` support. Needs fixes (§1 issues #2, #4, #5). |
| GUI framework | **`iced`** (recommended, confirmed in M0) | Cross‑platform Rust; built‑in `Subscription` for SSE streaming, async commands for background work. |
| Backend detection & selection | `crates/llama-ui::backend` (single module) | Probe `nvidia-smi`, `BackendType::Cuda`; pass `--backend` to `llama-server`. Detection + dropdown combined. |
| Model management | **New** `crates/llama-ui-models` | Download, manifest, scanning, GGUF metadata extraction. |
| Session persistence & export | **New** `crates/llama-ui-session` | Chat history, prompt templates, serialisation (JSON/MD/plain). |
| Chat template rendering | **Extended** `crates/common` | Already has `minijinja` dep. Add `chat_templates` module. Shared across CLI, server, GUI. |
| UI preferences | **Extended** `crates/config` (add `UiConfig`) | Avoids a whole new crate. The existing `crates/config` gets a `UiConfig` struct for TOML‑based UI prefs alongside the existing env‑var `Config`. |
| Sandbox client | **New** `crates/llama-ui-sandbox-client` | Manages `llama-server` process lifecycle: spawn, health‑check, restart, port allocation, crash detection. Extracted from GUI crate for testability. |
| Logging | **`tracing` + `tracing-subscriber`** | Already in workspace deps. **Do not** introduce `env_logger`. |
| HTTP client | **`reqwest`** (add to workspace) | Download models; stream SSE from `llama-server`. |
| Serialisation | **`serde` + `serde_json`** | Already in workspace deps. |
| Async runtime | **`tokio`** | Already in workspace deps (`features = "full"`). |
| Model metadata store | **`models.json`** at `$XDG_DATA_HOME/llama-ui/models.json` | Simple JSON, no migrations. |
| App preferences store | **`prefs.toml`** at `$XDG_CONFIG_HOME/llama-ui/prefs.toml` | Human‑editable TOML. |
| Model storage | `$XDG_DATA_HOME/llama-ui/models/` | XDG‑compliant. |
| Model download source | HuggingFace (default, configurable) | Pre‑converted `.gguf` files (single files, not archives). |
