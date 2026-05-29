# llama-ui — Desktop GUI Help

`llama-ui` is a native Rust desktop application for interactive LLM inference. Built on **iced 0.13**, it provides a polished dark-themed interface with multi-pane chat, streaming/non-streaming modes, session persistence, and per-pane resource controls.

## Quick Start

```bash
# Build the UI
cargo build -p llama-ui --release

# Run the UI
./target/release/llama-ui

# The UI will:
# 1. Scan for GGUF models in ~/.local/share/llama-ui/models/
# 2. Display the model picker screen
# 3. Let you select a model and start chatting
```

## Building

```bash
# Release build (recommended)
cargo build -p llama-ui --release

# Debug build
cargo build -p llama-ui

# The binary is at: target/release/llama-ui
```

## System Requirements

- **OS:** Linux (X11 or Wayland), macOS, or Windows
- **Display:** Any resolution supported by iced
- **Dependencies:** No external GUI libraries required — iced handles rendering

## Application Screens

### 1. Model Picker (Startup Screen)

The first screen you see when launching `llama-ui`.

**Layout:**
- Title: "llama-rs" with subtitle "LLM Inference Engine — Rust"
- Model list: One button per discovered GGUF model showing name and path
- Selected model is highlighted in green; others are blue

**Controls:**

| Button | Action |
|--------|--------|
| Model card | Select a model (highlights in green) |
| "Browse for More..." | Open native file picker to add a GGUF file |
| "Start Chat" | Launch chat with the selected model |

**Model discovery:**
- Scans `~/.local/share/llama-ui/models/` for `.gguf` files
- Uses a manifest (`models.json`) for metadata caching
- You can also browse for GGUF files anywhere on your system

---

### 2. Chat View

The main interface for interacting with the model. Supports one or two independent chat panes side-by-side.

**Per-Pane Layout (top to bottom):**

| Section | Description |
|---------|-------------|
| Header | "Pane N — {model_name}" with horizontal rule |
| Context warning | Shown when context > 80%; red text when > 95% |
| Context progress bar | Color-coded: blue (< 80%), yellow (80-95%), red (> 95%) |
| Controls row | Backend selector, streaming toggle, system prompt, new chat, settings, fullscreen |
| System prompt editor | Editable text input (toggled via "System" button) |
| Resource sliders | Memory (256-32768 MB) and CPU (10-400%) limits |
| Chat messages | Color-coded bubbles for User (blue), Assistant (green), System (gray) |
| Token counter | "Tokens: used/limit (pct%)" with color coding |
| Generation stats | "Last gen: N tokens in Xms (Y tok/s)" |
| Sampler sliders | Temperature, Top-k, Top-p, Repeat Penalty |
| Export/Import | JSON, MD, TXT export buttons + Import button |
| Text input | "Type your message..." placeholder, submits on Enter |
| Send/Stop button | "Send" or "Send (Stream)" when idle; "Stop" when generating |

---

### 3. Loading Screen

Displayed while the model loads and sandbox server starts.

**Content:**
- Title
- "Loading model..." text
- Status message (e.g., "Starting server...")
- "This may take a moment on first run."

---

### 4. Error Screen

Displayed when an error occurs.

**Content:**
- Title
- "Error" header (red)
- Error message in a styled container
- "Back to Model Picker" button

---

### 5. Settings Screen

Accessible from any chat pane via the "Settings" button.

**Sections:**
- **System Information:** Version, active pane count, available model count
- **Keyboard Shortcuts:** List of all keyboard shortcuts
- **Features:** List of all application features
- **"Back to Chat"** button

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Escape` | Close settings panel / return to chat |
| `F11` | Toggle full-screen mode |
| `Enter` | Send message (in active text input) |
| `Ctrl+Enter` | Send message (same as Enter) |

## Streaming Modes

Each chat pane can operate in one of two modes (toggled via the "Toggle" button):

### Streaming Mode (Default)
- Uses **SSE** (Server-Sent Events) for real-time token delivery
- Tokens appear one-by-one as they are generated
- Visual feedback with "Generating..." indicator
- Can be cancelled mid-generation with the "Stop" button

### Non-Streaming Mode (Block)
- Sends request and waits for the complete response
- Response appears all at once after generation completes
- Faster for short responses; no visual token-by-token feedback

**Toggle between modes:** Click the "Toggle" button in the controls row. The button label changes to show the current mode.

## Sampling Parameters

Each pane has independent sampling controls via sliders:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Temperature (T) | 0.00 – 2.00 | 0.80 | Higher = more creative; 0.0 = greedy |
| Top-k (K) | 0 – 100 | 40 | Number of top tokens to consider; 0 = disabled |
| Top-p (P) | 0.00 – 1.00 | 0.95 | Nucleus sampling threshold; 1.0 = disabled |
| Repeat Penalty (RP) | 1.00 – 2.00 | 1.10 | Penalty for repeated tokens; 1.0 = no penalty |

Changes are applied to the server in real-time via the `/samplers` endpoint.

## Resource Limits

Each pane has independent resource controls:

| Resource | Range | Default | Description |
|----------|-------|---------|-------------|
| Memory | 256 – 32768 MB | 4096 MB | Maximum memory for the sandbox server |
| CPU | 10 – 400% | 100% | CPU quota for the sandbox server |

**Note:** Resource limits are applied when the sandbox server is spawned (on "Start Chat" or backend change). Changing sliders requires restarting the sandbox.

## Session Management

### Export

Each pane's chat history can be exported in three formats:

| Format | Button | Description |
|--------|--------|-------------|
| JSON | "JSON" | Full session with metadata (model, sampler config, timestamps) |
| Markdown | "MD" | Formatted chat with role headers and timestamps |
| Plain Text | "TXT" | Messages only in `"Role: content\n"` format |

Export uses the native file save dialog.

### Import

Click "Import" to load a previously exported JSON session. Uses the native file open dialog.

### Clear Chat

Click "Clear Chat" to reset the conversation history without restarting the server. The context usage counter resets to zero.

### New Chat

Click "New Chat" to reset the session and generation stats without restarting the sandbox server. Useful for starting a fresh conversation with the same model.

## Model Management

### Model Discovery
- `llama-ui` scans `~/.local/share/llama-ui/models/` for `.gguf` files
- A manifest (`models.json`) caches model metadata (architecture, context length, quantization)
- Models are automatically discovered on startup

### Adding Models
1. Place `.gguf` files in `~/.local/share/llama-ui/models/`, or
2. Click "Browse for More..." in the model picker to add files from anywhere

### Model Metadata
The UI extracts metadata from GGUF files:
- Architecture (llama, mistral, phi, gemma, qwen)
- Context length (from GGUF metadata)
- Quantization (inferred from filename)
- File size

## System Prompt

Each pane has an editable system prompt:

1. Click the "System" button to toggle the editor
2. Edit the system prompt text (default: "You are a helpful assistant.")
3. The system prompt is prepended to the conversation for each request

## Dual-Pane Mode

`llama-ui` supports running two independent chat panes side-by-side:

- Each pane has its own model, session, sandbox server, and sampling parameters
- Panes are separated by a vertical rule
- Each pane operates independently — you can compare two models simultaneously
- Resource limits apply per-pane

**Note:** Dual-pane mode is available via the `AddPane` message (code-level API). The UI may not expose a direct button for this in all versions.

## Sandbox Server Management

Each chat pane runs its own `llama-server` instance as a sandboxed subprocess:

- **Auto-discovery:** Finds `llama-server` binary next to `llama-ui` or on `PATH`
- **Port allocation:** Automatically selects a free port
- **Health monitoring:** Polls `/health` every 200ms until ready (30s timeout)
- **Resource limits:** Applied via `systemd-run` on Linux (graceful fallback if unavailable)
- **Crash detection:** Detects server crashes and reports to the UI
- **Graceful shutdown:** Sends SIGTERM, waits 5s, then SIGKILL if needed

## Theme

The UI uses a custom dark theme with:

| Element | Color |
|---------|-------|
| Primary buttons | Blue (#4A90E2) |
| Success buttons | Green (#4AE26A) |
| Danger buttons | Red (#E24A4A) |
| Secondary buttons | Gray (#606060) |
| User messages | Blue tint (15% opacity) |
| Assistant messages | Green tint (15% opacity) |
| System messages | Gray tint (15% opacity) |
| Content background | Very dark (#1A1A1A) |
| Status bar | Dark (#2A2A2A) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Controls log verbosity. Default: `info`. |
| `XDG_DATA_HOME` | Override data directory for model storage (default: `~/.local/share`) |

## Troubleshooting

### No models found
- Place `.gguf` files in `~/.local/share/llama-ui/models/`
- Or click "Browse for More..." to add files from elsewhere

### Server fails to start
- Ensure `llama-server` binary is in the same directory as `llama-ui` or on `PATH`
- Check `RUST_LOG=debug` for detailed error messages
- Verify CUDA toolkit is installed if using GPU backend

### UI is unresponsive during generation
- This is expected during long generations in non-streaming mode
- Switch to streaming mode for better responsiveness
- Click "Stop" to cancel long-running generations

### Context limit warnings
- When context usage exceeds 80%, a yellow warning appears
- When it exceeds 95%, a red alert appears
- Click "Clear Chat" or "New Chat" to reset the context

## Related

- [llama-cli help](CLI_HELP.md) — Command-line text generation
- [llama-server help](SERVER_HELP.md) — HTTP server with REST API
- [Project README](../README.md) — Build instructions and overview
- [Architecture](../ARCHITECTURE.md) — Crate dependency graph and data flow
- [Iced API Reference](ICED_API.md) — Iced 0.13 patterns used by the UI
