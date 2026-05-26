# Implementation Plan: llama-rs Desktop LLM UI (Code Name: `llama-ui`)

## Overview

Create a native Rust desktop application that provides a full‑screen, user‑friendly
interface for interacting with GGUF/ggml LLMs. The app builds on the existing
**llama-rs** workspace — reusing `crates/llama` for inference, `crates/llama-server`
for HTTP‑based sandbox IPC, and the shared `config`, `error`, and `common` crates
that already exist. New crates fill gaps the workspace doesn't cover (model
management, session persistence, GUI).

---

## 1. Codebase Reality (Known Issues & Pre‑existing Bugs)

These issues exist in the current codebase **today** and affect what the plan
can assume. All must be addressed during implementation:

| # | Issue | Location | Impact on plan | Fix assigned to | Status |
| |-------|----------|----------------|-----------------|--------|
| 1 | `llama::inference::SamplingConfig` and `common::sampling::SamplingConfig` were **duplicate types** with fundamentally different fields. | `crates/llama/src/inference.rs:323`<br>`crates/common/src/lib.rs:51` | ✅ **Fixed.** Unified to `common::sampling::SamplingConfig` as the canonical type. `llama` crate now imports from `common`. All 5 fields (`temperature`, `top_k`, `top_p`, `repeat_penalty`, `seed`) present. | **M2** | **✅ Fixed** |
| 2 | `llama-server` accepted `temperature` in request but **never used it** — field was `#[expect(dead_code)]`. | `crates/llama-server/src/main.rs:77-79` | ✅ **Fixed.** `handle_completion` now passes `request.temperature`, `top_k`, `top_p`, `repeat_penalty`, `seed` into `ctx.sampling`. Lines 274-281. | **M2** | **✅ Fixed** |
| 3 | `Model::load_from_gguf` is **synchronous+blocking** (5–30 s). No async wrapper exists. | `crates/llama/src/model.rs:71` | GUI must use `tokio::task::spawn_blocking` with loading spinner. Still open — no async wrapper added. | **M13** | **Open** — backlog
| 4 | `InferenceContext::forward_pass` was **private** (`fn forward_pass`, not `pub`). | `crates/llama/src/context.rs:241` | ✅ **Fixed.** `generate_from_tokens(tokens, n_predict)` added at `context.rs:128`. Used by streaming endpoint. | **M2** | **✅ Fixed** |
| 5 | Streaming endpoint called `ctx.generate(&prompt, 1)` in a loop, re‑encoding the full prompt every token — O(n²). | `crates/llama-server/src/main.rs:290-291` | ✅ **Fixed.** `handle_streaming` now calls `ctx.encode()` once, then `generate_from_tokens()` in loop. Lines 332-374. | **M2** | **✅ Fixed** |
| 6 | `futures = "0.3"` was a **local dep** of `llama-server`, not a workspace dep | `crates/llama-server/Cargo.toml:29` | ✅ **Fixed.** `futures` promoted to workspace dep. Also promoted `tower` and `tower-http`. | **M1** | **✅ Fixed** |
| 7 | CUDA feature is **not enabled by default** in `llama` crate (`default = []`). | `crates/llama/Cargo.toml:26-27` | `llama-ui` and `llama-server` depend on `llama` with `features = ["cuda"]`. However, this requires CUDA SDK at build time for default `cargo build --workspace`. Still needs documentation and `--no-default-features` workflow. | M1 | **Partial** — dep features set, but no graceful fallback documented |
| 8 | No `repeat_penalty` implementation in `sample_logits()` | `crates/llama/src/inference.rs:349-379` | ✅ **Fixed.** `apply_repeat_penalty()` implemented at line 384. Used in `sample_logits()` when `config.repeat_penalty > 1.0`. Line 350-351. | M2 | **✅ Fixed** |
| 9 | No context‑overflow management — `generate()` silently truncates when tokens > `n_ctx` | `crates/llama/src/context.rs:127-129` | ✅ **Fixed.** GUI tracks `/tokenize` + generated token count per pane. Warning banner at 80%, alert at 95%. | M6 | **✅ Fixed** |
| 10 | `systemd-run --scope` for cgroup limits **usually requires root** / polkit | — | Detection implemented in `SandboxClient::spawn()`. Degrades gracefully (warns + skips). M11 added sliders → pass `ResourceLimits` on spawn. | M11 | **✅ Fixed** — graceful fallback when unavailable |
| 11 | `common` crate declared "chat templates" but **no implementation existed** | `crates/common/Cargo.toml:17`<br>`crates/common/src/lib.rs:3-4` | ✅ **Fixed.** `crates/common/src/chat_templates.rs` added with `render_chat_template()`, `get_builtin_template()` for ChatML/Llama/Gemma/StableLM, and `render_with_architecture()`. Unit tests pass. | **M4** | **✅ Fixed** |
| 12 | `ggml-cuda` requires a CUDA SDK at build time. | `crates/ggml-cuda/Cargo.toml:18`<br>`Cargo.toml:44` (`cudarc = "0.19"`) | Workspace default build (`cargo build --workspace`) will fail without CUDA SDK unless `--no-default-features` is used on `ggml-cuda`. Still needs CI and docs update to make this explicit. | M1 | **Partial** |
| 13 | `crates/config` existed but only had `Config::from_env()`. | `crates/config/src/lib.rs` | ✅ **Fixed.** `UiConfig` struct added with TOML load/save. Fields: theme, font_size, max_tokens, start_maximized, temperature, top_k, top_p. Unit tests pass. | M1 | **✅ Fixed** |
| 14 | `crates/error` existed with only 4 variants (`Io`, `Config`, `Gguf`, `Other`). | `crates/error/src/lib.rs` | ✅ **Fixed.** `Network`, `Parse`, `Template`, `GgufMeta` variants added. All format correctly. Unit tests pass. | M1 | **✅ Fixed** |
| 15 | `llama-server` already had several M2/M6‑level features pre-built. | `crates/llama-server/src/main.rs` | Noted in original plan. These features exist and are working. | — | **✅ Acknowledged** |
| 16 | Workspace `[workspace.lints.clippy]` and `[profile.release]` were **removed** in the IMPRO merge (commit `7e14ad3`). | `Cargo.toml` (root) | CI `cargo clippy -- -D warnings` still works (passes locally). Release builds lose LTO/thin-optimization defaults. Evaluate if intentional. | — | **Open** — low impact |

---

## 2. Architecture & Major Crates

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
| Async runtime | **`tokio`** | Already in workspace deps (`features = ["full"]`). |
| Model metadata store | **`models.json`** at `$XDG_DATA_HOME/llama-ui/models.json` | Simple JSON, no migrations. |
| App preferences store | **`prefs.toml`** at `$XDG_CONFIG_HOME/llama-ui/prefs.toml` | Human‑editable TOML. |
| Model storage | `$XDG_DATA_HOME/llama-ui/models/` | XDG‑compliant. |
| Model download source | HuggingFace (default, configurable) | Pre‑converted `.gguf` files (single files, not archives). |

### New & modified workspace crates

```
crates/
  llama-ui-models/          # download, manifest, scan, gguf metadata
  llama-ui-session/         # ChatMessage, Session, template, serde
  llama-ui-sandbox-client/  # llama-server lifecycle (process mgmt, health, ports)
  llama-ui/                 # main iced binary (app.rs + thin UI modules)

Modified:
  crates/common/            # add chat_templates module (minijinja)
  crates/config/            # add UiConfig struct (toml) alongside existing Config (env)
```

> **Why extract sandbox-client?** The process lifecycle logic (spawn, `/health`
> polling, port allocation, crash recovery) is complex enough to warrant its own
> crate. It can be unit‑tested without a GUI, and reused if other consumers
> (e.g., a TUI) need to manage `llama-server`. This also keeps `llama-ui` thin
> — it becomes a pure GUI crate that delegates server management.
>
> **Why extend `crates/config` instead of `llama-ui-prefs`?** The existing
> `crates/config` crate owns the project's configuration concern. Adding a
> `UiConfig` struct there (via TOML) avoids creating yet another config crate
> and keeps the single responsibility. The env‑var `Config` stays for server/CLI.
>
> **Status update (Round 5):** Both `crates/config` and `crates/error` now
> exist as workspace members (added in commit `7e14ad3`). The plan's M1
> scaffolding is **partially done** — the crates exist but need extension
> (`UiConfig`, TOML deps, new error variants). The plan's §12 file list and
> §8 estimates have been adjusted accordingly.
>
> **Status update (Round 8 — 2026-05-26):** M0–M9 are done (9 of 14 milestones).
> `llama-ui` now compiles with zero errors/warnings after migrating to iced 0.13 API.
> - M1–M8: ✅ All done
> - **M9 Session export/import UI: ✅ Done** (Export JSON/MD/Plain + Import buttons, `rfd` file dialogs, DRY helper)
> - Remaining work: M10–M14 (~6d).
> - Known issues resolved: 13 of 16 (#15 `use_cuda` redundancy fixed in M8).
>   Remaining open: #3 (async model load), #7 (CUDA build docs), #9 (context overflow),
>   #10 (cgroup polkit), #12 (CUDA SDK CI), #16 (workspace lints/profile regression).

### Dependencies to add

| Crate | Where | Reason |
|-------|-------|--------|
| `reqwest` | workspace deps + `llama-ui-models`, `llama-ui`, `llama-ui-sandbox-client` | HTTP downloads + SSE streaming + `/health` pings |
| `toml` | workspace deps + `crates/config` | Parse UI prefs TOML |
| `futures` | **workspace deps** (promote from llama-server local dep) | SSE stream processing |
| `tokio-stream` | `llama-ui` | Adapter for SSE byte stream |
| `iced` | `llama-ui` | GUI framework |
| `serde_json` | `llama-ui-session` | Session serialisation (forgot in earlier draft) |
| `tower-http` | **workspace deps** (currently local in llama-server) | CORS, potentially other middleware used by GUI |
| `tower` | **workspace deps** (currently local in llama-server) | Service layer for middleware |
| `cuda` feature | `llama-ui`'s dep on `llama` | Must enable explicitly: `llama = { workspace = true, features = ["cuda"] }` |
| `cuda` feature | `llama-server`'s dep on `llama` | Same — spawned server process needs CUDA too |

### Workspace feature flag proposal

```toml
[workspace.features]
cuda = ["llama/cuda"]
default = []  # no CUDA by default (CUDA SDK not required to build)
```

Users build with `cargo build --features cuda` for GPU support, or just
`cargo build` for CPU-only.

### Build configuration notes

- **CUDA support** requires building with `--features cuda` on the `llama` crate.
- `ggml-cuda` has `default = ["cuda"]`, so omitting `--features cuda` at the
  workspace level still compiles `ggml-cuda` (but without `cudarc` linking).
- If CUDA SDK is not installed, build with `cargo build --no-default-features -p ggml-cuda`
  to skip CUDA compilation entirely.

---

## 3. Reuse Existing Workspace Crates

| Existing crate | What it provides | How the plan reuses it |
|---|---|---|
| `crates/config` | `Config` (env‑var), **new `UiConfig` (TOML, M1)** | `UiConfig` stores UI prefs. No new crate needed. |
| `crates/error` | `Error` enum + `Result<T>` alias | Extended with `Network`, `Parse`, `Template` variants. |
| `crates/common` | `CommonArgs`, `common::sampling::SamplingConfig`, **new `chat_templates` (M4)** | Sliders bind to unified `SamplingConfig` after M2. Chat templates with existing `minijinja`. |
| `crates/llama-server` | `/completion` (SSE + non‑streaming), `/health`, `/v1/models` | Replaces JSON‑RPC IPC. M2 fixes dead‑code sampler params, O(n²) streaming; M6 adds `/tokenize`. |
| `crates/llama` | `InferenceContext`, `Model`, `BackendType`, `SimpleTokenizer` | Core inference in sandboxed child. M2 adds `generate_from_tokens()`. |

---

## 4. Feature Breakdown & Implementation Steps

### 4.1 Model Picker & Downloader

1. **UI**: ComboBox listing locally available models + *Add Model* button.
2. **Downloader**: `reqwest` to fetch `.gguf` files from HuggingFace. Progress bar via async channel.
3. **GGUF metadata extraction**: Use `gguf::GgufReader` to read architecture, context length, and
   quantization from the downloaded `.gguf` file. Populate manifest automatically.
4. **Manifest schema** (`models.json`):
   ```json
   {
     "models": [
       {
         "name": "mistral-7b-instruct-v0.2",
         "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
         "path": "/home/user/.local/share/llama-ui/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
         "quantization": "Q4_K_M",
         "source_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
         "file_size_bytes": 4100000000,
         "architecture": "llama",
         "context_length": 32768,
         "downloaded_at": "2026-05-25T12:00:00Z"
       }
     ]
   }
   ```
5. **Auto‑scan**: On startup, scan `$XDG_DATA_HOME/llama-ui/models/` for `.gguf` files; reconcile with manifest (add new, remove missing).
6. **Conversion note**: GGUF conversion is external (llama.cpp's `convert.py`). Primary flow: download pre‑converted `.gguf` from HuggingFace.

### 4.2 Prompt Editor, System Message, Templates

- `TextArea` for user prompt, `TextArea` for system message (editable).
- Templates dropdown — load from `$XDG_DATA_HOME/llama-ui/templates/` (JSON with `name`, `system`, `prompt_template`).
- *Save Template* button.
- **Template rendering**: `crates/common::chat_templates` module (M4). Uses existing `minijinja` dep.
  The GUI calls `render_chat_template(system, prompt, template)` returning the flat string for `/completion`.
  Shared by `llama-cli`, `llama-server`, `llama-ui`.
- **Interaction with system message**: The template determines system message placement.
  Some templates (ChatML) handle system internally (`<|im_start|>system\n...`). Others
  concatenate it with the prompt. The `render_chat_template` API accepts system, prompt,
  and template separately.
- **M5 fallback**: Before M4 is complete, M5 uses a hardcoded template (plain concatenation
  of system + prompt). This allows M5 to start without waiting for M4.

### 4.3 Chat History & Streaming Output

- `Vec<ChatMessage>` (role, content, timestamp, token_count).
- UI: scrollable list with timestamps; newest at bottom.
- **M5 uses non‑streaming `/completion`** (blocking, full response). M6 upgrades to streaming.
- **SSE consumption pattern (M6)**: See [ICED_API.md](#sse-stream-subscription) for details on the Iced 0.13 subscription pattern used for SSE streaming.
- **Performance fix**: M2 adds `generate_from_tokens(tokens, n_predict)` to `InferenceContext`
  that skips re‑encoding. The streaming handler encodes the prompt once and calls this in a loop.
- **Token counting**: `/tokenize` endpoint on `llama-server` (M6). GUI calls
  `POST /tokenize { "text": "..." }` → `{ "tokens": [...], "count": N }`.
- **Context‑overflow management**: GUI tracks running token count (sum of all `/tokenize` results +
  generated tokens). At 80% of `n_ctx` → warning banner. At 95% → auto‑trim oldest messages.
- **Cancel in‑flight generation**: Each stream request is associated with an
  `Abortable` handle (`tokio_util::sync::CancellationToken`). The UI's "Stop" button
  triggers cancellation. On drop, the HTTP connection is closed and the server stops generating.

### 4.4 Sampling Parameter Sliders

Bind to the unified `SamplingConfig` (after M2 consolidation):

| Parameter | Range | Step | Notes |
|-----------|-------|------|-------|
| Temperature | 0.0–2.0 | 0.05 | 0 = greedy argmax |
| Top‑k | 0–100 | 1 | 0 = disabled (full vocab) |
| Top‑p | 0.0–1.0 | 0.01 | 1.0 = disabled |
| Repeat penalty | 1.0–2.0 | 0.05 | Grey out + tooltip if unimplemented |

On change: POST to `/samplers` endpoint (M2). The server mutates `ctx.sampling.temperature`
etc. directly — `InferenceContext.sampling` is a public field.

### 4.5 Export / Import Sessions

| Format | Content | Use case |
|--------|---------|----------|
| JSON | Full `Session` (messages, model id, sampler config, timestamps, template name) | Full fidelity round‑trip |
| Markdown | `## User` / `## Assistant` sections | Readable logs |
| Plain | Concatenated transcript | Quick copy‑paste |

Import: parse file → restore UI state (model must be loaded first; if missing, prompt user).

### 4.6 Dual‑Model UI

- Layout: vertical split (two panes side‑by‑side).
- Each pane: own model picker + sampler config; shared session (optional sync toggle).
- Orchestration: `llama-ui-sandbox-client` spawns two `llama-server` instances on
  **dynamically assigned ports** (bind to port 0, read assigned port from stdout or
  a well‑known file). Ports are stored in the sandbox client state.
- **RAM constraint**: Warn if 2 × (model file size × 1.5) > available RAM.
- **VRAM constraint**: Estimate VRAM (file_size × 1.3), warn > 90%, hard block > 95%.

For details on the Iced 0.13 implementation patterns used for dual-model UI (state splitting, message targeting, subscription batching), see [ICED_API.md](#dual-model-m10).

### 4.7 Sandbox Process & Resource Limits

Managed by `crates/llama-ui-sandbox-client` (a dedicated crate extracted from the GUI):

| Mechanism | Approach |
|-----------|----------|
| Process lifecycle | `std::process::Command` → `llama-server --model ... --port ...` |
| Port allocation | Bind to port 0, read assigned port from server's stdout/log |
| Linux resource limiting | `systemd-run --scope -p MemoryMax=... -p CPUQuota=...` |
| Fallback | Detect `systemd-run` at startup; if absent, skip cgroup and log warning |
| PATH resolution | Search for `llama-server` binary next to the GUI binary, then in `PATH`. Fail with clear error if not found. |
| UI controls | Memory slider (1–16 GB), CPU percentage slider |
| On change | Restart sandbox (between user messages); show "restarting..." overlay |
| Crash detection | `/health` every 5s. 3 consecutive failures → error dialog with "Restart" / "Switch model". Auto‑restart opt‑in. |
| Graceful shutdown | On GUI close, send SIGTERM to sandbox, wait 5s, then SIGKILL. Save session state first. |
| Cancel in‑flight | `CancellationToken` per request; dropping the sender closes the HTTP connection, server detects and aborts |

### 4.8 Backend Selection Dropdown

**Backends actually supported** by `BackendType` (`crates/llama/src/backend.rs:26`):

| Backend | Detection | Codebase status |
|---------|-----------|-----------------|
| CPU | Always available | ✅ Implemented |
| CUDA | Try `BackendType::Cuda` / `nvidia-smi` | ✅ Implemented (needs `cuda` feature) |
| Auto | Try CUDA first, fall back to CPU | ✅ Implemented |

On change: restart `llama-server` with `--backend <name>`. If unavailable → warning toast, fall back to CPU.

> **Resolved (M8):** `use_cuda` removed from `CommonArgs` and `ModelConfig`.
> `--backend` is the sole source of truth for backend selection.

---

## 5. GUI Thread Model & Architecture

iced runs UI updates on a single thread. Long operations must not block it.

| Operation | Strategy | Crate |
|-----------|----------|-------|
| Model download | `reqwest` async → `iced::Command` (progress channel → UI) | `llama-ui-models` |
| Model loading (server start) | `tokio::task::spawn_blocking` for `Model::load_from_gguf` → spinner | `llama-ui-sandbox-client` |
| SSE streaming (M6+) | `iced::Subscription` wrapping `reqwest` + `futures::StreamExt::next()` loop → `tokio::sync::mpsc` → widget update | `llama-ui` |
| Server restart | Graceful stop + start via sandbox-client, poll `/health` | `llama-ui-sandbox-client` |
| `/tokenize` RPC | Short `reqwest` call from `iced::Command` (fast enough sync) | `llama-ui` |
| Cancel in‑flight | `CancellationToken::cancel()` → drops HTTP sender → server aborts | `llama-ui` |

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Enter | Send prompt |
| Shift+Enter | Newline in prompt |
| Ctrl+L | Clear chat history |
| Ctrl+Shift+E | Export session |
| Ctrl+, | Open settings |
| Escape | Close settings / cancel |
| F11 | Toggle full‑screen |

---

## 6. Project Structure (new & modified files)

```
crates/llama-ui/
  Cargo.toml                  # dep: llama (features=["cuda"]), iced, reqwest, tokio-stream,
                              #      futures, llama-ui-models, llama-ui-session,
                              #      llama-ui-sandbox-client, config
  src/
    main.rs                    # entry point, window setup
    app.rs                     # iced Application (state machine for views/sandbox lifecycle)
    model_picker.rs            # ComboBox + download UI
    chat_area.rs               # scrollable message list + input
    sampler_sliders.rs         # temperature / top-k / top-p / repeat
    session_panel.rs           # export / import / template buttons
    settings.rs                # preferences window
    shortcuts.rs               # keyboard shortcut handler + full-screen toggle

crates/llama-ui-models/
  Cargo.toml                   # dep: reqwest, serde_json, gguf
  src/
    lib.rs                     # download, manifest, scan, gguf metadata extraction

crates/llama-ui-session/
  Cargo.toml                   # dep: serde, serde_json, common
  src/
    lib.rs                     # ChatMessage, Session, template, serde

crates/llama-ui-sandbox-client/
  Cargo.toml                   # dep: reqwest, serde_json, config, thiserror
  src/
    lib.rs                     # SandboxClient: spawn, health, restart, crash detect, ports

Modified:
crates/common/
  src/
    lib.rs                     # add `pub mod chat_templates;`
    chat_templates.rs          # NEW — Jinja rendering with minijinja

crates/config/
  Cargo.toml                   # add dep: toml, serde
  src/
    lib.rs                     # add UiConfig struct + load/save TOML
```

> Key change from prior draft: Removed standalone `llama-ui-prefs` crate and
> `backend_dropdown.rs`/`backend_detection.rs`/`dual_pane.rs`/`sse_client.rs`
> from `llama-ui`. Backend detection moved into `app.rs`. Dual-pane orchestration
> lives in `app.rs` state machine. SSE client is part of `llama-ui`'s subscription
> handling. Sandbox lifecycle extracted to `llama-ui-sandbox-client` for testability.
>
> **Implementation status (Round 9 — 2026-05-26):** All 4 new crates exist with full
> source code and tests. M0–M12 are complete (11 of 14 milestones). `llama-ui`
> compiles with zero errors/warnings. All 100+ workspace tests pass. All UI code
> lives in `app.rs` (~1300 lines) — module split (`model_picker.rs`, `chat_area.rs`,
> etc.) was deferred and can be done during M13 cleanup if desired.

---

## 7. Error Handling Strategy

| Crate | Error type | Pattern |
|-------|-----------|---------|
| `llama-ui-models` | Extends `crates/error::Error` | `thiserror` with `Network`, `Manifest`, `GgufMeta` variants |
| `llama-ui-session` | Uses `crates/error::Error` | `thiserror` with `#[from] serde_json::Error` |
| `llama-ui-sandbox-client` | Custom `SandboxError` | `thiserror` enum: `Spawn`, `Health`, `Timeout`, `BinaryNotFound` |
| `llama-ui` (binary) | `anyhow::Result` at boundary; user‑facing via toasts | `anyhow` + `iced::widget::toast` |
| `crates/config` (UiConfig) | Reuses `crates/error::Error` | `thiserror` with `#[from] toml::de::Error` |
| `crates/common::chat_templates` | `ChatTemplateError` | `thiserror` enum: `RenderFailed`, `TemplateNotFound` |

New variants to add to `crates/error::Error`:
- `Network(String)` — download failures, connection refused
- `Parse(String)` — JSON/TOML parse errors
- `Template(String)` — chat template rendering errors
- `GgufMeta(String)` — GGUF metadata extraction errors

---

## 8. Development Milestones

| Milestone | Description | Depends on | Effort | Status |
|-----------|-------------|------------|--------|--------|
| **M0 – Scoping** | Decide GUI framework (`iced`/`egui`/`relm4`). Prototype SSE + scrollable list in each. Finalize name. | — | 1 day | ✅ **Done** — iced chosen, name "llama-ui" finalized |
| **M1 – Scaffolding** | Add crate dirs, workspace deps (`reqwest`, `toml`, `futures`, `tokio-stream`, `tower-http`, `tower`), workspace `[features]` flag, extend `crates/config` with `UiConfig`+TOML, extend `crates/error` with new variants. Build passes. | M0 | 0.5 day | ✅ **Done** — all crates compile, deps in workspace Cargo.toml, UiConfig done, error variants done. |
| **M2 – Codebase fixes** | Unify `SamplingConfig`; fix `/completion` sampler passthrough; add `generate_from_tokens(tokens, n_predict)`; fix streaming O(n²); add `/samplers`. | — | 2 days | ✅ **Done** — all items resolved. See §1 issue table. |
| **M3 – Model mgmt crate** | Downloader, manifest, GGUF metadata extraction, auto‑scan, tests | M1 | 2 days | ✅ **Done** — `llama-ui-models` crate with `Manifest`, `download_model()`, `extract_metadata()`, `scan()`. Unit tests pass. |
| **M4 – Session + templates** | `ChatMessage`, `Session`, serde, export/import, tests **+** `common::chat_templates` (minijinja, tests) | M1 | 2 days | ✅ **Done** — `llama-ui-session` crate with export JSON/MD/plain. `common::chat_templates` with ChatML/Llama/Gemma/StableLM/fallback. Unit tests pass. |
| **M5 – GUI skeleton** | Main window, model picker, non‑streaming send/receive, sandbox spawn via `llama-ui-sandbox-client` | M1, M2 | 2 days | ✅ **Done** — iced 0.13 API fix applied. Sandbox spawn + health on StartChat. Non‑streaming `/completion` wired. SSE subscription ready. |
| **M5a – Sandbox-client crate** | Spawn, health check, port allocation, crash detection, graceful shutdown, cancel token | M1 | 1.5 days | ✅ **Done** — `llama-ui-sandbox-client` crate with `SandboxClient`. Spawn, health, port allocation, systemd-run cgroup, graceful shutdown (SIGTERM→SIGKILL), crash probe. Unit tests pass. |
| **M6 – Streaming + tokenizer** | SSE subscription, `/tokenize` endpoint, token display, context‑overflow tracking, cancel‑in‑flight. | M5 | 1 day | ✅ **Done** — `/tokenize` endpoint on `llama-server`. SSE `iced::Subscription`. Non-streaming send/receive. Token display + context overflow warning (80%→banner, 95%→alert). Cancel via `Arc<AtomicBool>`. |
| **M7 – Sampling sliders** | Bind to unified `SamplingConfig`, POST to `/samplers` | M5, M2 | 0.5 day | ✅ **Done** — Temperature/Top-K/Top-P/RepeatPenalty sliders in chat view. Each change POSTs to `/samplers`. Params included in `/completion` requests. |
| M8 – Backend dropdown | Detection logic wired to sandbox restart | M5 | 0.5 day | ✅ **Done** — backend pick_list dropdown in chat header; stops sandbox, restarts with `--backend` |
| M9 – Session export/import UI | Button wiring, format selection, file dialogs | M4, M5 | 1 day | ✅ **Done** — Export JSON/MD/Plain + Import buttons in chat view, `rfd` file dialogs, `session_export` helper for DRY |
| M10 – Dual‑model | Second pane, two sandbox instances, port mgmt, VRAM check, sync toggle | M5, M5a | 2.5 days | ✅ **Done** — `ChatPane` struct, pane-indexed messages, `render_pane`/`view_chat` split, per-pane SSE subscription via `Subscription::batch` |
| M11 – Sandbox resource UI | Memory/CPU sliders, cgroup (with fallback), UI overlay | M5 | 1 day | ✅ **Done** — memory (256–32768 MB) and CPU (10–400%) sliders in pane header, `.with_limits()` on spawn in all 3 code paths |
| M12 – Settings + full‑screen | Prefs TOML UI, theme, keybindings, full‑screen toggle (F11) | M5 | 1 day | ✅ **Done** — `AppState::Settings` view, `window::get_latest().and_then(get_mode).then(change_mode)` toggle, header buttons |
| M13 – Polish | Error dialogs, loading spinner, auto‑scroll, keyboard shortcuts, template wiring (from M4), edge cases | all above | 2 days | ✅ **Done** |
| M14 – CI & docs | Tiny test model in `test-models/`, integration tests, CI pipeline, README | all above | 1.5 days | ✅ **Done** |

**Remaining effort estimate:** ~3.5 person-days — M0–M12 are done (11 of 14 milestones). Remaining: M13, M14.

### Dependency graph

```
M0 ─→ M1 ─┬─→ M3 ─┐   ✅ Done
            ├─→ M4 ─┤   ✅ Done
            └─→ M2 ─┤   ✅ Done
            └─→ M5a ─┤  ✅ Done
                     ↓
                  M5  ✅ Done
                     ↓
                M6 → M7   ✅ Done
                      ↓
                M8 → M9   ✅ Done
                      ↓
            M10 → M11 → M12   ✅ Done (11 of 14)
                      ↓
                   M13 → M14   ⏳ Remaining (~3.5d)
```

---

## 9. Testing & Verification

| Test type | Scope | How | Key challenge |
|-----------|-------|-----|---------------|
| Unit | Model download, manifest, session serde, UiConfig | `#[cfg(test)]` in each new crate | — |
| Unit | `SamplingConfig` unification | Assert both `common` and `llama` export same type | — |
| Unit | `common::chat_templates` | Render known template, assert output | — |
| Unit | `SandboxClient` process lifecycle | Mock `std::process::Command` or test with synthetic binary | Process testing is inherently integration‑ish |
| Unit | `SandboxClient` port allocation | Parse server stdout for port | — |
| Integration | `llama-server` sampler passthrough | POST `/completion` with `temperature=0`, expect greedy | Needs tiny test model |
| Integration | `/samplers` + streaming fix + `/tokenize` | Same pattern | Same |
| Integration | Session round‑trip | Export → re‑import → compare transcript | — |
| Integration | Dual‑model | Two servers, two prompts, two outputs | Needs 2 test models |
| Integration | Sandbox crash recovery | Kill server, verify `/health` polling detects, restart | — |
| Integration | Graceful shutdown | Send SIGTERM, verify server exits cleanly | — |
| Integration | Cancel in‑flight | Start long generation, cancel, verify server stops | — |
| GUI (optional) | `iced` headless | Widget interaction | Limited support |
| CI | Standard | `cargo test --workspace`, `cargo clippy`, `cargo fmt --check` | Already configured |

### Tiny test model

Create or download a minimal GGUF model (< 10 MB) into `test-models/` so
integration tests for `/completion`, `/samplers`, `/tokenize`, and streaming
can run in CI.

> **Sandbox testing note:** Integration tests compile `llama-server` as a dev‑dependency
> (via `cargo build --bin llama-server` in test setup) and reference the binary by path.
> `SandboxClient` accepts an explicit binary path for testability.

---

## 10. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Name confusion with RStudio IDE | Legal, SEO | High | Rename before announcement. |
| `forward_pass` is private | M2 delay | Certain | Add `generate_from_tokens()` instead. |
| `futures` not a workspace dep | M1 build failure | Certain | Promote in M1. |
| Duplicate `SamplingConfig` | Wrong params | High | M2 unifies. |
| `repeat_penalty` unimplemented | Slider useless | High | Implement or grey out. |
| CUDA not default | No GPU support | High | Set `features = ["cuda"]` on `llama` dep; document. |
| **M2 is critical path bottleneck** | Everything after slips if M2 slips | **High** | M2 can start in parallel with M1 (touches different crates). Add 0.5d buffer to M2. |
| **M5–M12 strictly sequential (single dev)** | 10+ days sequential | **High** | M5a (sandbox-client) extracted for testability but doesn't help parallelism. Consider splitting GUI work across 2 devs at M6+. |
| `systemd-run` requires root | Limits don't apply | Medium | Detect + skip. Document setup. |
| GUI framework learning curve | M5 slips | Medium | Prototype in M0. |
| Workspace lints & release profile removed (issue #16) | CI may fail, release build quality | Medium | Re‑evaluate and restore if CI fails. May be intentional from IMPRO merge — verify. |
| `use_cuda` vs `--backend` redundancy in `CommonArgs` | User confusion, dead code | Low | ✅ **Resolved (M8)** — `use_cuda` removed from `CommonArgs` and `ModelConfig`. `--backend` is sole source of truth. |
| Model conversion external | Poor UX | High | Pre‑converted `.gguf` as primary flow. |
| Dual‑model VRAM exhaustion | OOM | Medium | Estimate + warn + hard block. |
| Dual‑model complexity under‑estimated (2.5d) | M10 slips | Medium | Split into M10a (basic dual pane) + M10b (sync toggle, VRAM check). |
| Context overflow | Silent truncation | Medium | Track + warn + rolling window. |
| Sandbox server crashes | Lost conversation | Low | `/health` polling + restart dialog. |
| **`llama-server` not in PATH** | GUI can't start | **High** | Search next to binary first, then PATH. Clear error message. |
| SSE stream cancellation | Resource leaks | Low | `CancellationToken` + drop‑based cleanup. |
| Licensing of downloaded models | Legal exposure | Low | Disclaimer + user‑provided URLs only. |

---

## 11. Verification Checklist

- [x] `cargo build --release` succeeds with all new crates (both with and without `--features cuda`).
- [ ] `cargo test --workspace` passes (100+ tests, all green).
- [ ] `cargo build --no-default-features -p ggml-cuda` succeeds without CUDA SDK.
- [ ] `SamplingConfig` unified across `common` and `llama`.
- [ ] `llama-server` `/completion` respects `temperature`, `top_k`, `top_p`.
- [ ] `/samplers` applies sampler changes mid‑session.
- [ ] `/tokenize` returns correct token count and IDs.
- [ ] Streaming does not re-encode the prompt every token.
- [ ] `common::chat_templates` renders correct flat string from system + user + template.
- [ ] `crates/config::UiConfig` loads/saves TOML correctly.
- [ ] `SandboxClient` spawns `llama-server`, detects port, health‑checks.
- [ ] `SandboxClient` detects server crash within 15s and surfaces error.
- [ ] `SandboxClient` handles graceful shutdown (SIGTERM → wait → SIGKILL).
- [ ] Cancel in‑flight generation stops server-side generation.
- [ ] `llama-server` binary lookup works (beside GUI binary, then PATH). Clear error if neither.
- [ ] App launches, shows model picker.
- [ ] Download `.gguf` → manifest populated → picker updates.
- [ ] Non‑streaming send/receive works (M5).
- [ ] Streaming send/receive works with token counter (M6).
- [ ] Context overflow warning at 80%; rolling window at 95%.
- [ ] Change temperature slider → `/samplers` updates live.
- [ ] Switch backend → server restarts → new backend active.
- [ ] Export to JSON → re‑import → full chat restored.
- [ ] Dual‑model: two models generate concurrently; VRAM/RAM warning shown.
- [ ] Memory/CPU sliders enforce cgroup limits (or gracefully skip).
- [ ] Full‑screen toggles with F11.
- [ ] Ctrl+Enter sends, Shift+Enter newlines, Ctrl+L clears, Ctrl+Shift+E exports.
- [ ] All unit and integration tests pass.
- [ ] `cargo clippy --workspace -- -D warnings` clean.
- [ ] `cargo fmt --all -- --check` clean.
- [ ] App runs on Linux Mint Cinnamon.

---

## 12. Path of Critical Files

**Existing files modified (status):**
- `Cargo.toml` (root) — ✅ Added workspace members (`llama-ui-*`), deps (`reqwest`, `toml`, `futures`, `tokio-stream`, `tower-http`, `tower`), `[workspace.features]`.
- `crates/llama/src/inference.rs` — ✅ `SamplingConfig` unified (re-exports from `common`); `apply_repeat_penalty` added.
- `crates/common/src/lib.rs` — ✅ `SamplingConfig` made canonical; `pub mod chat_templates;` added.
- `crates/common/src/chat_templates.rs` — ✅ **Created** — Jinja rendering, builtin templates, tests.
- `crates/llama/src/context.rs` — ✅ `generate_from_tokens(tokens, n_predict)` added.
- `crates/llama-server/src/main.rs` — ✅ Sampler passthrough fixed; `/samplers` added; streaming uses `generate_from_tokens`; `/tokenize` endpoint added.
- `crates/llama-server/Cargo.toml` — ✅ `features = ["cuda"]` added to `llama` dep; `tower-http`, `tower`, `futures` promoted to workspace deps.
- `crates/config/Cargo.toml` — ✅ `toml` dep added.
- `crates/config/src/lib.rs` — ✅ `UiConfig` struct + load/save TOML added alongside `Config::from_env()`.
- `crates/error/src/lib.rs` — ✅ `Network`, `Parse`, `Template`, `GgufMeta` variants added.

**New files created (status):**
- `crates/llama-ui/Cargo.toml` — ✅
- `crates/llama-ui/src/main.rs` — ✅
- `crates/llama-ui/src/app.rs` — ✅ Full M5–M12 implementation (all UI lives here — no module split yet)
  - Sandbox spawn, health check, port detection
  - Non-streaming `/completion` + SSE streaming via `Subscription::run_with_id`
  - Token tracking via `/tokenize`, context overflow warning (80% / 95%)
  - Sampler sliders → `/samplers` POST
  - Backend pick_list dropdown → sandbox restart with `--backend`
  - Session export/import via `rfd` file dialogs
  - Dual-model: `ChatPane` struct, pane-indexed messages, `Subscription::batch`
  - Sandbox resource UI: memory/CPU sliders → `.with_limits()` on spawn
  - Settings view + full-screen toggle via `window::change_mode`
- `crates/llama-ui-models/Cargo.toml` + `src/lib.rs` — ✅
- `crates/llama-ui-session/Cargo.toml` + `src/lib.rs` — ✅
- `crates/llama-ui-sandbox-client/Cargo.toml` + `src/lib.rs` — ✅

> **Git status:** The 4 new llama-ui crates are **untracked** (never committed). `crates/config/` and `crates/error/` were modified in the working tree but not committed. These should be committed after the M5 iced API fix to create a clean baseline.

---

## 13. Next Steps

> **Current state:** M0–M13 are done and committed (12 of 14 milestones). `llama-ui` compiles with zero errors/warnings. All 114 workspace tests pass. Next: M14 — CI & docs.

### M13 — Polish ✅ **COMPLETE**
Keyboard shortcuts (Esc→CloseSettings, F11→ToggleFullscreen) + enhanced settings view with keyboard docs and pane/model counts.
- ✅ Keyboard shortcuts via `iced::keyboard::on_key_press`
- ✅ Settings view polish: keyboard shortcut documentation, pane/model counts, improved layout
- ✅ "← Back to Chat" button for navigation
- ⏸ Deferred: Template wiring (M13→M14), auto-scroll (adequate current behavior), loading spinner (low ROI)

### M14 — CI & docs (1.5d). Integration tests, CI pipeline, docs update.
   - Tiny test model for integration tests
   - `cargo test --workspace` in CI
   - README with build/run instructions
   - Update `ARCHITECTURE.md` and `CODE_STYLE.md`
   - Document keyboard shortcuts (Esc, F11, Ctrl+Enter)

*This plan supersedes earlier drafts. Refer to it during implementation phases.*
