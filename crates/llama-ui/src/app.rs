//! Main iced application state machine.
//!
//! Manages sandbox lifecycle (spawn, health, stop), chat sessions,
//! and context tracking. M5+ with M6 non-streaming /completion.

use iced::Length::Fill;
use iced::{Color, Element, Subscription, Task, Theme, window};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use futures::SinkExt;
use iced::keyboard;
use iced::keyboard::key::Named as NamedKey;
use iced::stream;
use iced::widget::{button, column, pick_list, row, scrollable, slider, text_input, vertical_rule};

use llama_ui_models::Manifest;
use llama_ui_sandbox_client::{ResourceLimits, SandboxClient};
use llama_ui_session::{ChatMessage, Role, Session};
use std::path::PathBuf;

/// Application state.
/// Per-pane state for dual-model support (M10).
#[derive(Debug)]
pub struct ChatPane {
    /// Model index into `LlamaApp::models`.
    pub selected_model: usize,
    /// Chat history for this pane.
    pub session: Session,
    /// Message input text.
    pub input_text: String,
    /// Sandbox client for the pane's server.
    pub sandbox: Option<SandboxClient>,
    /// Server base URL.
    pub server_address: String,
    /// Whether an SSE stream is active on this pane.
    pub is_streaming: bool,
    /// Cancellation flag for in-flight generation.
    pub cancelled: Arc<AtomicBool>,
    /// Whether streaming mode is enabled for this pane.
    pub use_streaming: bool,
    /// Running token count.
    pub total_tokens: usize,
    /// Model context limit.
    pub context_limit: usize,
    /// Backend ("auto", "cpu", "cuda").
    pub backend: String,
    /// System prompt for this pane.
    pub system_prompt: String,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k (0 = disabled).
    pub top_k: f32,
    /// Top-p.
    pub top_p: f32,
    /// Repeat penalty.
    pub repeat_penalty: f32,
    /// Resource limits for the sandbox.
    pub resource_limits: ResourceLimits,
    /// Tokens generated in last completion.
    pub last_gen_tokens: usize,
    /// Duration of last generation in milliseconds.
    pub last_gen_ms: u64,
    /// Whether the system prompt editor is visible.
    pub show_system_prompt: bool,
    /// When the last generation started.
    pub send_started_at: Option<Instant>,
}

impl ChatPane {
    fn new(selected_model: usize, model_name: &str) -> Self {
        Self {
            selected_model,
            session: Session::new(model_name),
            input_text: String::new(),
            sandbox: None,
            server_address: String::new(),
            is_streaming: false,
            cancelled: Arc::new(AtomicBool::new(false)),
            use_streaming: true,
            total_tokens: 0,
            context_limit: 4096,
            backend: "auto".to_string(),
            system_prompt: "You are a helpful assistant.".to_string(),
            temperature: 0.8,
            top_k: 40.0,
            top_p: 0.95,
            repeat_penalty: 1.1,
            resource_limits: ResourceLimits {
                memory_mb: 4096,
                cpu_percent: 100,
            },
            last_gen_tokens: 0,
            last_gen_ms: 0,
            show_system_prompt: false,
            send_started_at: None,
        }
    }
}

/// Application state.
#[derive(Debug)]
pub struct LlamaApp {
    /// Current view state.
    state: AppState,
    /// Available models.
    models: Vec<ModelInfo>,
    /// Chat panes (usually 1, up to 2 for dual-model).
    panes: Vec<ChatPane>,
    /// Which pane the user is interacting with.
    active_pane: usize,
    /// Status message.
    status: String,
}

impl Default for LlamaApp {
    fn default() -> Self {
        Self {
            state: AppState::ModelPicker,
            models: Vec::new(),
            panes: Vec::new(),
            active_pane: 0,
            status: String::new(),
        }
    }
}

/// Model information for display.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Human-readable model name.
    pub name: String,
    /// Path to the GGUF file.
    pub path: PathBuf,
}

/// Application state machine states.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AppState {
    /// Model picker view.
    #[default]
    ModelPicker,
    /// Chat view.
    Chat,
    /// Loading model / starting sandbox.
    Loading,
    /// Error state.
    Error(String),
    /// Settings screen.
    Settings,
}

/// Messages that update the application state.
#[derive(Debug, Clone)]
pub enum Message {
    /// Model selected from picker.
    ModelSelected(usize),
    /// Start first chat pane with selected model.
    StartChat,
    /// Send message on a pane.
    Send(usize),
    /// Input text changed on a pane.
    InputChanged(usize, String),
    /// Error occurred.
    Error(String),
    // ─── M6 additions (pane-indexed) ─────────────────────────
    /// Non-streaming completion response received.
    CompletionReceived(usize, String),
    /// Token count from /tokenize.
    TokenCounted(usize, usize),
    /// Sandbox successfully started on the given port.
    SandboxStarted(usize, u16),
    /// Cancel in-flight generation on a pane.
    CancelGeneration(usize),
    /// One chunk from SSE streaming.
    StreamChunk(usize, String),
    /// SSE stream finished cleanly on a pane.
    StreamEnded(usize),
    // ─── M8: Backend changed ────────────────────────────────
    /// Backend selection changed on a pane.
    BackendChanged(usize, String),
    // ─── M9: Session export/import (on active pane) ──────────
    /// Export session as JSON.
    ExportJson,
    /// Export session as Markdown.
    ExportMarkdown,
    /// Export session as plain text.
    ExportPlain,
    /// Import session from file.
    ImportSession,
    // ─── M7: Sampler slider messages (pane-indexed) ──────────
    /// Temperature slider changed on a pane.
    TemperatureChanged(usize, f32),
    /// Top-k slider changed on a pane.
    TopKChanged(usize, f32),
    /// Top-p slider changed on a pane.
    TopPChanged(usize, f32),
    /// Repeat penalty slider changed on a pane.
    RepeatPenaltyChanged(usize, f32),
    // ─── M11: Resource limit sliders ───────────────────────────
    /// Memory limit changed on a pane (MB).
    MemoryChanged(usize, u64),
    /// CPU quota changed on a pane (%).
    CpuChanged(usize, u8),
    // ─── M12: Settings + full-screen ───────────────────────────
    /// Toggle full-screen mode.
    ToggleFullscreen,
    /// Open settings screen.
    OpenSettings,
    /// Close settings screen.
    CloseSettings,
    // ─── M13: UX improvements ────────────────────────────────
    /// Clear chat history on a pane.
    ClearChat(usize),
    /// Toggle streaming mode on a pane.
    ToggleStreaming(usize),
    /// The model selected from the model picker (tracks selection without starting chat).
    ModelPickerSelected(usize),
    /// Browse for a GGUF file to add.
    BrowseModel,
    /// Samplers updated successfully (status display).
    SamplersUpdated,
    /// System prompt changed on a pane.
    SystemPromptChanged(usize, String),
    /// Toggle system prompt editor visibility.
    ToggleSystemPrompt(usize),
    /// Start a new chat (reset session) on a pane without restarting server.
    NewChat(usize),
}

/// Update the application state.
pub fn update(state: &mut LlamaApp, message: Message) -> Task<Message> {
    match message {
        // ─── Model selection ─────────────────────────────────
        Message::ModelSelected(idx) => {
            // Update model in all panes that use this index
            for p in &mut state.panes {
                p.selected_model = idx;
                // Don't restart sandbox — user must start a new chat
            }
            Task::none()
        }

        // ─── Start chat: spawn first pane's sandbox ──────────
        Message::StartChat => {
            if state.models.is_empty() {
                return Task::none();
            }
            state.state = AppState::Loading;
            state.status = "Starting model...".to_string();
            let selected = state.active_pane.min(state.models.len().saturating_sub(1));
            let model = state.models[selected].clone();

            state.panes.push(ChatPane::new(selected, &model.name));
            let pane = state.panes.len() - 1;
            let backend = state.panes[pane].backend.clone();
            let limits = state.panes[pane].resource_limits.clone();

            Task::perform(
                async move {
                    let binary = SandboxClient::resolve_binary().map_err(|e| e.to_string())?;
                    let mut client = SandboxClient::new(binary, model.path, &backend, "llama-ui")
                        .with_limits(limits);
                    client.spawn().map_err(|e| e.to_string())?;
                    client
                        .wait_for_ready(Duration::from_secs(30))
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<u16, String>(client.port)
                },
                move |result| match result {
                    Ok(port) => Message::SandboxStarted(pane, port),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── Sandbox started for a pane ──────────────────────
        Message::SandboxStarted(pane, port) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            p.server_address = format!("http://127.0.0.1:{}", port);
            state.state = AppState::Chat;
            state.active_pane = pane;
            state.status = format!("Pane {} ready (port {})", pane, port);

            let model = state.models[p.selected_model].clone();
            match SandboxClient::resolve_binary() {
                Ok(binary) => {
                    let mut client = SandboxClient::new(binary, model.path, &p.backend, "llama-ui");
                    client.port = port;
                    p.sandbox = Some(client);
                }
                Err(e) => {
                    state.status = format!("Warning: {}", e);
                }
            }
            Task::none()
        }

        // ─── Send message on a pane ──────────────────────────
        Message::Send(pane) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            if p.input_text.is_empty() || p.server_address.is_empty() {
                return Task::none();
            }

            // Add user message to session
            p.session.add_message(ChatMessage {
                role: Role::User,
                content: p.input_text.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                token_count: None,
            });
            p.input_text.clear();
            p.send_started_at = Some(Instant::now());
            state.status = format!("Pane {} generating...", pane);

            if p.use_streaming {
                // Streaming mode: add placeholder message and trigger subscription
                p.session.add_message(ChatMessage {
                    role: Role::Assistant,
                    content: String::new(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    token_count: None,
                });
                p.is_streaming = true;
                let addr = p.server_address.clone();
                let cancelled = p.cancelled.clone();
                let temperature = p.temperature;
                let top_k = p.top_k;
                let top_p = p.top_p;
                let repeat_penalty = p.repeat_penalty;

                // Build prompt from conversation history
                let prompt = p
                    .session
                    .messages
                    .iter()
                    .filter(|m| !m.content.is_empty())
                    .map(|m| {
                        let role = match m.role {
                            Role::User => "User",
                            Role::Assistant => "Assistant",
                            Role::System => "System",
                        };
                        format!("{}: {}", role, m.content)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                Task::perform(
                    async move {
                        let client = reqwest::Client::new();
                        let resp = client
                            .post(format!("{}/completion", addr))
                            .json(&serde_json::json!({
                                "prompt": prompt,
                                "max_tokens": 512,
                                "stream": true,
                                "temperature": temperature,
                                "top_k": top_k,
                                "top_p": top_p,
                                "repeat_penalty": repeat_penalty,
                            }))
                            .send()
                            .await
                            .map_err(|e| e.to_string())?;
                        let mut byte_stream = resp.bytes_stream();
                        let mut full_content = String::new();
                        use futures::StreamExt;
                        while let Some(chunk) = byte_stream.next().await {
                            if cancelled.load(Ordering::Relaxed) {
                                break;
                            }
                            match chunk {
                                Ok(bytes) => {
                                    let text = String::from_utf8_lossy(&bytes);
                                    for line in text.lines() {
                                        if let Some(data) = line.strip_prefix("data: ") {
                                            if let Ok(val) =
                                                serde_json::from_str::<serde_json::Value>(data)
                                            {
                                                if val["stop"].as_bool().unwrap_or(false) {
                                                    return Ok::<String, String>(full_content);
                                                }
                                                if let Some(content) = val["content"].as_str() {
                                                    full_content.push_str(content);
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                        Ok::<String, String>(full_content)
                    },
                    move |result| match result {
                        Ok(content) => Message::CompletionReceived(pane, content),
                        Err(e) => Message::Error(e),
                    },
                )
            } else {
                // Non-streaming mode
                let prompt = p
                    .session
                    .messages
                    .iter()
                    .map(|m| {
                        let role = match m.role {
                            Role::User => "User",
                            Role::Assistant => "Assistant",
                            Role::System => "System",
                        };
                        format!("{}: {}", role, m.content)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let addr = p.server_address.clone();
                let temperature = p.temperature;
                let top_k = p.top_k;
                let top_p = p.top_p;
                let repeat_penalty = p.repeat_penalty;

                Task::perform(
                    async move {
                        let client = reqwest::Client::new();
                        let resp = client
                            .post(format!("{}/completion", addr))
                            .json(&serde_json::json!({
                                "prompt": prompt,
                                "max_tokens": 512,
                                "stream": false,
                                "temperature": temperature,
                                "top_k": top_k,
                                "top_p": top_p,
                                "repeat_penalty": repeat_penalty,
                            }))
                            .send()
                            .await
                            .map_err(|e| e.to_string())?;
                        let body: serde_json::Value =
                            resp.json().await.map_err(|e| e.to_string())?;
                        let content = body["content"].as_str().unwrap_or("").to_string();
                        Ok::<String, String>(content)
                    },
                    move |result| match result {
                        Ok(content) => Message::CompletionReceived(pane, content),
                        Err(e) => Message::Error(e),
                    },
                )
            }
        }

        // ─── Completion received (non-streaming) on a pane ─────
        Message::CompletionReceived(pane, content) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            p.is_streaming = false;

            // Compute generation timing stats
            if let Some(start) = p.send_started_at.take() {
                p.last_gen_ms = start.elapsed().as_millis() as u64;
                // Rough token estimate: ~4 chars per token for English
                p.last_gen_tokens = content.len().div_ceil(4);
            }

            // If the last message is an empty assistant message (from streaming placeholder),
            // update it in-place; otherwise append a new one.
            if let Some(last) = p.session.messages.last_mut() {
                if matches!(last.role, Role::Assistant) && last.content.is_empty() {
                    last.content = content.clone();
                } else {
                    p.session.add_message(ChatMessage {
                        role: Role::Assistant,
                        content: content.clone(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        token_count: None,
                    });
                }
            } else {
                p.session.add_message(ChatMessage {
                    role: Role::Assistant,
                    content: content.clone(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    token_count: None,
                });
            }
            state.status = String::new();

            let addr = p.server_address.clone();
            let full_text = p
                .session
                .messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            Task::perform(
                async move {
                    let client = reqwest::Client::new();
                    let resp = client
                        .post(format!("{}/tokenize", addr))
                        .json(&serde_json::json!({ "text": full_text }))
                        .send()
                        .await
                        .map_err(|e| e.to_string())?;
                    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
                    let count = body["count"].as_u64().unwrap_or(0) as usize;
                    Ok::<usize, String>(count)
                },
                move |result| match result {
                    Ok(count) => Message::TokenCounted(pane, count),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── Token count updated for a pane ──────────────────
        Message::TokenCounted(pane, count) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            p.total_tokens = count;
            let limit = p.context_limit;
            let pct = if limit > 0 {
                (count as f64 / limit as f64) * 100.0
            } else {
                0.0
            };
            if pct > 95.0 {
                state.status = format!(
                    "⚠️ Pane {} context at {}/{} tokens ({:.0}%) — consider clearing history",
                    pane, count, limit, pct
                );
            } else if pct > 80.0 {
                state.status = format!(
                    "⚠️ Pane {} context at {}/{} tokens ({:.0}%) — approaching limit",
                    pane, count, limit, pct
                );
            } else if count > 0 {
                state.status = format!("Pane {} tokens: {}/{} ({:.0}%)", pane, count, limit, pct);
            }
            Task::none()
        }

        // ─── Cancel generation on a pane ─────────────────────
        Message::CancelGeneration(pane) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            p.cancelled.store(true, Ordering::Relaxed);
            p.is_streaming = false;
            state.status = format!("Pane {} cancelled.", pane);
            Task::none()
        }

        // ─── SSE streaming chunk for a pane ──────────────────
        Message::StreamChunk(pane, text) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            if let Some(last) = p.session.messages.last_mut() {
                if matches!(last.role, Role::Assistant) {
                    last.content.push_str(&text);
                }
            }
            Task::none()
        }

        // ─── SSE stream ended for a pane ─────────────────────
        Message::StreamEnded(pane) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            p.is_streaming = false;
            state.status = String::new();

            let addr = p.server_address.clone();
            let full_text = p
                .session
                .messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            Task::perform(
                async move {
                    let client = reqwest::Client::new();
                    let resp = client
                        .post(format!("{}/tokenize", addr))
                        .json(&serde_json::json!({ "text": full_text }))
                        .send()
                        .await
                        .map_err(|e| e.to_string())?;
                    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
                    let count = body["count"].as_u64().unwrap_or(0) as usize;
                    Ok::<usize, String>(count)
                },
                move |result| match result {
                    Ok(count) => Message::TokenCounted(pane, count),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── Input changed on a pane ─────────────────────────
        Message::InputChanged(pane, text) => {
            if pane < state.panes.len() {
                state.panes[pane].input_text = text;
            }
            Task::none()
        }

        // ─── Error ───────────────────────────────────────────
        Message::Error(err) => {
            state.state = AppState::Error(err);
            for p in &mut state.panes {
                p.is_streaming = false;
            }
            Task::none()
        }

        // ─── M8: Backend changed on a pane — restart sandbox ─
        Message::BackendChanged(pane, backend) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            {
                let p = &mut state.panes[pane];
                if backend == p.backend {
                    return Task::none();
                }
                if let Some(ref mut client) = p.sandbox {
                    client.stop();
                }
                p.sandbox = None;
                p.backend = backend;
                p.server_address.clear();
            }
            state.state = AppState::Loading;
            state.status = format!("Restarting pane {}...", pane);

            let model = state.models[state.panes[pane].selected_model].clone();
            let backend = state.panes[pane].backend.clone();
            let limits = state.panes[pane].resource_limits.clone();

            Task::perform(
                async move {
                    let binary = SandboxClient::resolve_binary().map_err(|e| e.to_string())?;
                    let mut client = SandboxClient::new(binary, model.path, &backend, "llama-ui")
                        .with_limits(limits);
                    client.spawn().map_err(|e| e.to_string())?;
                    client
                        .wait_for_ready(Duration::from_secs(30))
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<u16, String>(client.port)
                },
                move |result| match result {
                    Ok(port) => Message::SandboxStarted(pane, port),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── M9: Session export/import (on active pane) ──────
        Message::ExportJson => {
            if state.active_pane < state.panes.len() {
                session_export(state, "session.json", "JSON", &["json"], |s, p| {
                    s.export_json(p)
                })
            } else {
                Task::none()
            }
        }
        Message::ExportMarkdown => {
            if state.active_pane < state.panes.len() {
                session_export(state, "session.md", "Markdown", &["md"], |s, p| {
                    s.export_markdown(p)
                })
            } else {
                Task::none()
            }
        }
        Message::ExportPlain => {
            if state.active_pane < state.panes.len() {
                session_export(state, "session.txt", "Text", &["txt"], |s, p| {
                    s.export_plain(p)
                })
            } else {
                Task::none()
            }
        }
        Message::ImportSession => {
            if state.active_pane >= state.panes.len() {
                return Task::none();
            }
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .pick_file()
            {
                match llama_ui_session::Session::import_json(&path) {
                    Ok(imported) => {
                        state.panes[state.active_pane].session = imported;
                        state.panes[state.active_pane].total_tokens = 0;
                        state.status = format!("Session imported: {}", path.display());
                    }
                    Err(e) => {
                        state.status = format!("Import error: {e}");
                    }
                }
            }
            Task::none()
        }

        // ─── M7: Sampler slider changes on a pane ────────────
        Message::TemperatureChanged(pane, val) => {
            if pane < state.panes.len() {
                state.panes[pane].temperature = val;
                fire_update_sampler(state, pane)
            } else {
                Task::none()
            }
        }
        Message::TopKChanged(pane, val) => {
            if pane < state.panes.len() {
                state.panes[pane].top_k = val;
                fire_update_sampler(state, pane)
            } else {
                Task::none()
            }
        }
        Message::TopPChanged(pane, val) => {
            if pane < state.panes.len() {
                state.panes[pane].top_p = val;
                fire_update_sampler(state, pane)
            } else {
                Task::none()
            }
        }
        Message::RepeatPenaltyChanged(pane, val) => {
            if pane < state.panes.len() {
                state.panes[pane].repeat_penalty = val;
                fire_update_sampler(state, pane)
            } else {
                Task::none()
            }
        }

        // ─── M11: Resource limit changes ──────────────────────
        Message::MemoryChanged(pane, memory_mb) => {
            if pane < state.panes.len() {
                state.panes[pane].resource_limits.memory_mb = memory_mb;
                state.status = format!("Pane {} memory: {} MB", pane, memory_mb);
            }
            Task::none()
        }
        Message::CpuChanged(pane, cpu_percent) => {
            if pane < state.panes.len() {
                state.panes[pane].resource_limits.cpu_percent = cpu_percent;
                state.status = format!("Pane {} CPU: {}%", pane, cpu_percent);
            }
            Task::none()
        }

        // ─── M12: Settings + full-screen ──────────────────────
        Message::ToggleFullscreen => window::get_latest()
            .and_then(move |id| window::get_mode(id).map(move |mode| (id, mode)))
            .then(|(id, current_mode)| match current_mode {
                window::Mode::Fullscreen => {
                    window::change_mode::<Message>(id, window::Mode::Windowed)
                }
                _ => window::change_mode::<Message>(id, window::Mode::Fullscreen),
            }),
        Message::OpenSettings => {
            state.state = AppState::Settings;
            Task::none()
        }
        Message::CloseSettings => {
            state.state = AppState::Chat;
            Task::none()
        }

        // ─── M13: UX improvements ────────────────────────────────
        Message::ClearChat(pane) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            let model_name = state
                .models
                .get(p.selected_model)
                .map(|m| m.name.as_str())
                .unwrap_or("unknown");
            p.session = Session::new(model_name);
            p.total_tokens = 0;
            state.status = format!("Pane {} chat cleared.", pane);
            Task::none()
        }
        Message::ToggleStreaming(pane) => {
            if pane < state.panes.len() {
                state.panes[pane].use_streaming = !state.panes[pane].use_streaming;
                let mode = if state.panes[pane].use_streaming {
                    "streaming"
                } else {
                    "non-streaming"
                };
                state.status = format!("Pane {} set to {} mode.", pane, mode);
            }
            Task::none()
        }
        Message::ModelPickerSelected(idx) => {
            state.active_pane = idx;
            Task::none()
        }
        Message::BrowseModel => {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GGUF Model", &["gguf"])
                .pick_file()
            {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "imported".to_string());
                state.models.push(ModelInfo {
                    name,
                    path: path.clone(),
                });
                state.status = format!("Added model: {}", path.display());
            }
            Task::none()
        }
        Message::SamplersUpdated => Task::none(),
        Message::SystemPromptChanged(pane, text) => {
            if pane < state.panes.len() {
                state.panes[pane].system_prompt = text;
            }
            Task::none()
        }
        Message::ToggleSystemPrompt(pane) => {
            if pane < state.panes.len() {
                state.panes[pane].show_system_prompt = !state.panes[pane].show_system_prompt;
            }
            Task::none()
        }
        Message::NewChat(pane) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            let model_name = state
                .models
                .get(p.selected_model)
                .map(|m| m.name.as_str())
                .unwrap_or("unknown");
            p.session = Session::new(model_name);
            p.total_tokens = 0;
            p.last_gen_tokens = 0;
            p.last_gen_ms = 0;
            state.status = format!("Pane {} — new chat started.", pane);
            Task::none()
        }
    }
}

/// Send the current sampler config to the pane's server.
fn fire_update_sampler(state: &LlamaApp, pane: usize) -> Task<Message> {
    if pane >= state.panes.len() || state.panes[pane].server_address.is_empty() {
        return Task::none();
    }
    let p = &state.panes[pane];
    let addr = p.server_address.clone();
    let temperature = p.temperature;
    let top_k = p.top_k;
    let top_p = p.top_p;
    let repeat_penalty = p.repeat_penalty;

    Task::perform(
        async move {
            let client = reqwest::Client::new();
            let resp = client
                .post(format!("{}/samplers", addr))
                .json(&serde_json::json!({
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                }))
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            Ok::<String, String>(body.to_string())
        },
        |result| match result {
            Ok(_) => Message::SamplersUpdated,
            Err(e) => Message::Error(e),
        },
    )
}

/// Open a save-dialog and call the given export function on the active pane's session.
fn session_export(
    state: &mut LlamaApp,
    filename: &str,
    label: &str,
    exts: &[&str],
    f: fn(&Session, &PathBuf) -> Result<(), llama_ui_session::ExportError>,
) -> Task<Message> {
    if state.active_pane >= state.panes.len() {
        return Task::none();
    }
    let dialog = rfd::FileDialog::new()
        .set_file_name(filename)
        .add_filter(label, exts);
    if let Some(path) = dialog.save_file() {
        match f(&state.panes[state.active_pane].session, &path) {
            Ok(()) => state.status = format!("Session exported: {}", path.display()),
            Err(e) => state.status = format!("Export error: {e}"),
        }
    }
    Task::none()
}

/// View the application UI.
pub fn view(state: &LlamaApp) -> Element<'_, Message> {
    match &state.state {
        AppState::ModelPicker => view_model_picker(state),
        AppState::Chat => view_chat(state),
        AppState::Loading => view_loading(state),
        AppState::Error(err) => view_error(err),
        AppState::Settings => view_settings(state),
    }
}

/// Theme the application.
pub fn theme(_state: &LlamaApp) -> Theme {
    Theme::default()
}

/// Build one SSE stream subscription for a single pane.
fn pane_subscription(pane: usize, p: &ChatPane) -> Subscription<Message> {
    if !p.is_streaming || p.server_address.is_empty() {
        return Subscription::none();
    }

    let addr = p.server_address.clone();
    let cancelled = p.cancelled.clone();
    let temperature = p.temperature;
    let top_k = p.top_k;
    let top_p = p.top_p;
    let repeat_penalty = p.repeat_penalty;

    let prompt = p
        .session
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            format!("{}: {}", role, m.content)
        })
        .chain(std::iter::once(format!("User: {}", p.input_text)))
        .collect::<Vec<_>>()
        .join("\n");

    let id = format!("sse-{}-{}", pane, p.session.messages.len());

    Subscription::run_with_id(
        id,
        stream::channel(32, move |mut output| {
            let addr = addr;
            let cancelled = cancelled;
            let prompt = prompt;

            async move {
                let client = reqwest::Client::new();
                match client
                    .post(format!("{}/completion", addr))
                    .json(&serde_json::json!({
                        "prompt": prompt,
                        "max_tokens": 512,
                        "stream": true,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repeat_penalty": repeat_penalty,
                    }))
                    .send()
                    .await
                {
                    Ok(response) => {
                        let mut byte_stream = response.bytes_stream();
                        use futures::StreamExt;
                        while let Some(chunk) = byte_stream.next().await {
                            if cancelled.load(Ordering::Relaxed) {
                                break;
                            }
                            match chunk {
                                Ok(bytes) => {
                                    let text = String::from_utf8_lossy(&bytes);
                                    for line in text.lines() {
                                        if let Some(data) = line.strip_prefix("data: ") {
                                            if let Ok(val) =
                                                serde_json::from_str::<serde_json::Value>(data)
                                            {
                                                if val["stop"].as_bool().unwrap_or(false) {
                                                    let _ = output
                                                        .send(Message::StreamEnded(pane))
                                                        .await;
                                                    return;
                                                }
                                                if let Some(content) = val["content"].as_str() {
                                                    let _ = output
                                                        .send(Message::StreamChunk(
                                                            pane,
                                                            content.to_string(),
                                                        ))
                                                        .await;
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                        let _ = output.send(Message::StreamEnded(pane)).await;
                    }
                    Err(e) => {
                        let _ = output.send(Message::Error(e.to_string())).await;
                    }
                }
            }
        }),
    )
}

/// Map keyboard shortcuts to messages.
fn handle_key_press(key: keyboard::Key, _: keyboard::Modifiers) -> Option<Message> {
    match key {
        keyboard::Key::Named(NamedKey::Escape) => Some(Message::CloseSettings),
        keyboard::Key::Named(NamedKey::F11) => Some(Message::ToggleFullscreen),
        _ => None,
    }
}

/// Subscription for SSE streaming across all panes + keyboard shortcuts.
pub fn subscription(state: &LlamaApp) -> Subscription<Message> {
    let mut subs: Vec<Subscription<Message>> = Vec::new();

    // Global keyboard shortcuts
    subs.push(iced::keyboard::on_key_press(handle_key_press));

    // Per-pane SSE subscriptions
    for (i, p) in state.panes.iter().enumerate() {
        subs.push(pane_subscription(i, p));
    }
    match subs.len() {
        1 => subs.into_iter().next().unwrap(),
        _ => Subscription::batch(subs),
    }
}

// ---- View helpers ----

/// View for model picker.
fn view_model_picker(state: &LlamaApp) -> Element<'_, Message> {
    let mut children = vec![
        iced::widget::text("llama-rs")
            .size(42)
            .color(Color::from_rgba8(0x4A, 0x90, 0xE2, 1.0))
            .into(),
        iced::widget::text("LLM Inference Engine — Rust")
            .size(16)
            .color(Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0))
            .into(),
        iced::widget::text("").size(12).into(),
    ];

    if !state.status.is_empty() {
        children.push(
            iced::widget::text(&state.status)
                .size(14)
                .color(Color::from_rgba8(0xE2, 0x4A, 0x4A, 1.0))
                .into(),
        );
        children.push(iced::widget::text("").size(6).into());
    }

    if state.models.is_empty() {
        children.push(
            iced::widget::text("No models found. Add a GGUF model:")
                .size(16)
                .color(Color::from_rgba8(0xC0, 0xC0, 0xC0, 1.0))
                .into(),
        );
        children.push(iced::widget::text("").size(8).into());
        children.push(
            iced::widget::text("• Place .gguf files in ~/.local/share/llama-ui/models/")
                .size(13)
                .color(Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0))
                .into(),
        );
        children.push(
            iced::widget::text("• Or click \"Browse\" to select a file")
                .size(13)
                .color(Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0))
                .into(),
        );
        children.push(iced::widget::text("").size(8).into());
        children.push(
            button(iced::widget::text("Browse for GGUF File").size(16))
                .style(llama_ui_core::theme::secondary_button_style)
                .on_press(Message::BrowseModel)
                .padding(iced::Padding::from([10, 24]))
                .width(Fill)
                .into(),
        );
    } else {
        children.push(
            iced::widget::text(format!("Select a model ({} available)", state.models.len()))
                .size(16)
                .color(Color::from_rgba8(0xC0, 0xC0, 0xC0, 1.0))
                .into(),
        );
        children.push(iced::widget::text("").size(8).into());

        for (i, model) in state.models.iter().enumerate() {
            let is_selected = i == state.active_pane;
            let btn = button(
                column(vec![
                    iced::widget::text(&model.name)
                        .size(18)
                        .color(Color::WHITE)
                        .into(),
                    iced::widget::text(model.path.to_string_lossy())
                        .size(11)
                        .color(Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0))
                        .into(),
                ])
                .spacing(4),
            )
            .style(if is_selected {
                llama_ui_core::theme::success_button_style
            } else {
                llama_ui_core::theme::primary_button_style
            })
            .on_press(Message::ModelPickerSelected(i))
            .padding(iced::Padding::from([10, 16]))
            .width(Fill);
            children.push(btn.into());
        }

        children.push(iced::widget::text("").size(6).into());
        children.push(
            button(iced::widget::text("Browse for More...").size(14))
                .style(llama_ui_core::theme::secondary_button_style)
                .on_press(Message::BrowseModel)
                .padding(iced::Padding::from([8, 16]))
                .width(Fill)
                .into(),
        );
    }

    children.push(iced::widget::text("").size(16).into());

    children.push(
        button(iced::widget::text("Start Chat").size(20))
            .style(if state.models.is_empty() {
                llama_ui_core::theme::secondary_button_style
            } else {
                llama_ui_core::theme::success_button_style
            })
            .on_press(Message::StartChat)
            .padding(iced::Padding::from([14, 32]))
            .width(Fill)
            .into(),
    );

    iced::widget::container(
        iced::widget::scrollable(column(children).spacing(4).width(Fill)).height(Fill),
    )
    .style(llama_ui_core::theme::content_area_style)
    .center_x(Fill)
    .center_y(Fill)
    .padding(40)
    .into()
}

/// Render a single chat pane.
fn render_pane(state: &LlamaApp, pane: usize) -> Element<'_, Message> {
    let p = &state.panes[pane];
    let mut children = Vec::new();

    // ── Header: model name + controls ──────────────────────
    let model_name = state
        .models
        .get(p.selected_model)
        .map(|m| m.name.as_str())
        .unwrap_or("unknown");
    children.push(
        row![
            iced::widget::text(format!("Pane {} — {}", pane, model_name))
                .size(16)
                .color(Color::from_rgba8(0xE0, 0xE0, 0xE0, 1.0)),
            iced::widget::container(iced::widget::text("").size(1))
                .width(Fill)
                .height(iced::Length::Fixed(1.0))
                .style(|_theme: &iced::Theme| iced::widget::container::Style {
                    background: Some(Color::from_rgba8(0x60, 0x60, 0x60, 0.5).into()),
                    border: iced::Border::default(),
                    text_color: None,
                    shadow: iced::Shadow::default(),
                }),
        ]
        .spacing(8)
        .into(),
    );

    // ── Context overflow warning ──────────────────────────
    if p.context_limit > 0 {
        let pct = p.total_tokens as f64 / p.context_limit as f64;
        if pct > 0.80 {
            let warning = if pct > 0.95 {
                format!(
                    "Context at {}/{} tokens — consider clearing history",
                    p.total_tokens, p.context_limit
                )
            } else {
                format!(
                    "Context at {}/{} tokens ({:.0}%)",
                    p.total_tokens,
                    p.context_limit,
                    pct * 100.0
                )
            };
            let color = if pct > 0.95 {
                Color::from_rgba8(0xE2, 0x4A, 0x4A, 1.0)
            } else {
                Color::from_rgba8(0xE2, 0xC0, 0x4A, 1.0)
            };
            children.push(iced::widget::text(warning).size(13).color(color).into());
        }
    }

    // ── Context usage progress bar ────────────────────────
    if p.context_limit > 0 {
        let pct = (p.total_tokens as f32 / p.context_limit as f32).min(1.0);
        let bar_color = if pct > 0.95 {
            Color::from_rgba8(0xE2, 0x4A, 0x4A, 0.9)
        } else if pct > 0.80 {
            Color::from_rgba8(0xE2, 0xC0, 0x4A, 0.9)
        } else {
            Color::from_rgba8(0x4A, 0x90, 0xE2, 0.9)
        };
        children.push(
            iced::widget::container(iced::widget::text("").size(1))
                .style(move |_theme: &iced::Theme| iced::widget::container::Style {
                    background: Some(bar_color.into()),
                    border: iced::Border {
                        radius: 2.0.into(),
                        width: 0.0,
                        color: Color::TRANSPARENT,
                    },
                    text_color: None,
                    shadow: iced::Shadow::default(),
                })
                .width(Fill)
                .height(iced::Length::Fixed(4.0))
                .into(),
        );
    }

    // ── Controls row: backend, streaming, settings, fullscreen ──
    children.push(
        row![
            iced::widget::text("Backend:").size(13),
            pick_list(
                vec!["auto", "cpu", "cuda"],
                Some(p.backend.as_str()),
                move |s: &str| Message::BackendChanged(pane, s.to_string()),
            )
            .width(100),
            iced::widget::text("│")
                .size(13)
                .color(Color::from_rgba8(0x60, 0x60, 0x60, 1.0)),
            iced::widget::text(if p.use_streaming {
                "⚡ Stream"
            } else {
                "📄 Block"
            })
            .size(13),
            button(iced::widget::text("Toggle").size(11))
                .on_press(Message::ToggleStreaming(pane))
                .padding(iced::Padding::from([2, 8])),
            iced::widget::text("│")
                .size(13)
                .color(Color::from_rgba8(0x60, 0x60, 0x60, 1.0)),
            button(iced::widget::text("💬 System").size(11))
                .on_press(Message::ToggleSystemPrompt(pane))
                .padding(iced::Padding::from([2, 8])),
            button(iced::widget::text("🔄 New Chat").size(11))
                .on_press(Message::NewChat(pane))
                .padding(iced::Padding::from([2, 8])),
            iced::widget::text("│")
                .size(13)
                .color(Color::from_rgba8(0x60, 0x60, 0x60, 1.0)),
            button(iced::widget::text("⚙ Settings").size(11))
                .on_press(Message::OpenSettings)
                .padding(iced::Padding::from([2, 8])),
            button(iced::widget::text("⛶ Fullscreen").size(11))
                .on_press(Message::ToggleFullscreen)
                .padding(iced::Padding::from([2, 8])),
        ]
        .spacing(6)
        .into(),
    );

    // ── System prompt editor (toggled) ──────────────────────
    if p.show_system_prompt {
        children.push(
            iced::widget::container(
                column(vec![
                    iced::widget::text("System Prompt:")
                        .size(12)
                        .color(Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0))
                        .into(),
                    text_input("Enter system prompt...", &p.system_prompt)
                        .on_input(move |s| Message::SystemPromptChanged(pane, s))
                        .padding(8)
                        .size(13)
                        .into(),
                ])
                .spacing(4),
            )
            .style(llama_ui_core::theme::user_message_style)
            .padding(iced::Padding::from([6, 8]))
            .width(Fill)
            .into(),
        );
    }

    // ── Resource limit sliders ────────────────────────────
    children.push(
        row![
            iced::widget::text(format!("Mem: {} MB", p.resource_limits.memory_mb)).size(11),
            slider(
                256.0..=32768.0,
                p.resource_limits.memory_mb as f32,
                move |v| Message::MemoryChanged(pane, v as u64),
            )
            .step(256.0)
            .width(Fill),
        ]
        .spacing(4)
        .into(),
    );
    children.push(
        row![
            iced::widget::text(format!("CPU: {}%", p.resource_limits.cpu_percent)).size(11),
            slider(
                10.0..=400.0,
                f32::from(p.resource_limits.cpu_percent),
                move |v| Message::CpuChanged(pane, v as u8),
            )
            .step(10.0)
            .width(Fill),
        ]
        .spacing(4)
        .into(),
    );

    // ── Messages ──────────────────────────────────────────
    for msg in &p.session.messages {
        let (role_label, role_color) = match msg.role {
            Role::User => ("You", Color::from_rgba8(0x4A, 0x90, 0xE2, 1.0)),
            Role::Assistant => ("AI", Color::from_rgba8(0x4A, 0xE2, 0x6A, 1.0)),
            Role::System => ("System", Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0)),
        };
        let display_content = if msg.content.is_empty() && matches!(msg.role, Role::Assistant) {
            "Generating...".to_string()
        } else {
            msg.content.clone()
        };
        let msg_container = iced::widget::container(
            column(vec![
                iced::widget::text(role_label)
                    .size(12)
                    .color(role_color)
                    .into(),
                iced::widget::text(display_content).size(14).into(),
            ])
            .spacing(4),
        )
        .padding(iced::Padding::from([8, 12]))
        .width(Fill);

        let styled_msg = match msg.role {
            Role::User => msg_container.style(llama_ui_core::theme::user_message_style),
            Role::Assistant => msg_container.style(llama_ui_core::theme::assistant_message_style),
            Role::System => msg_container.style(llama_ui_core::theme::system_message_style),
        };
        children.push(styled_msg.into());
    }

    // ── Token counter ─────────────────────────────────────
    if p.total_tokens > 0 {
        let pct = if p.context_limit > 0 {
            (p.total_tokens as f64 / p.context_limit as f64) * 100.0
        } else {
            0.0
        };
        let counter_color = if pct > 95.0 {
            Color::from_rgba8(0xE2, 0x4A, 0x4A, 1.0)
        } else if pct > 80.0 {
            Color::from_rgba8(0xE2, 0xC0, 0x4A, 1.0)
        } else {
            Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0)
        };
        children.push(
            iced::widget::text(format!(
                "Tokens: {}/{} ({:.0}%)",
                p.total_tokens, p.context_limit, pct
            ))
            .size(11)
            .color(counter_color)
            .into(),
        );
    }

    // ── Generation stats (tok/s) ──────────────────────────
    if p.last_gen_tokens > 0 && p.last_gen_ms > 0 {
        let tok_per_sec = (p.last_gen_tokens as f64 / (p.last_gen_ms as f64 / 1000.0)) as u64;
        children.push(
            iced::widget::text(format!(
                "Last gen: {} tokens in {}ms ({} tok/s)",
                p.last_gen_tokens, p.last_gen_ms, tok_per_sec
            ))
            .size(11)
            .color(Color::from_rgba8(0x80, 0x80, 0x80, 1.0))
            .into(),
        );
    }

    // ── Sampler sliders ───────────────────────────────────
    children.push(
        iced::widget::text("Sampling")
            .size(12)
            .color(Color::from_rgba8(0x80, 0x80, 0x80, 1.0))
            .into(),
    );
    children.push(
        row![
            iced::widget::text(format!("T: {:.2}", p.temperature)).size(11),
            slider(0.0..=2.0, p.temperature, move |v| {
                Message::TemperatureChanged(pane, v)
            })
            .step(0.05)
            .width(Fill),
        ]
        .spacing(4)
        .into(),
    );
    children.push(
        row![
            iced::widget::text(format!("K: {}", p.top_k)).size(11),
            slider(0.0..=100.0, p.top_k, move |v| Message::TopKChanged(pane, v))
                .step(1.0)
                .width(Fill),
        ]
        .spacing(4)
        .into(),
    );
    children.push(
        row![
            iced::widget::text(format!("P: {:.2}", p.top_p)).size(11),
            slider(0.00..=1.00, p.top_p, move |v| Message::TopPChanged(pane, v))
                .step(0.01)
                .width(Fill),
        ]
        .spacing(4)
        .into(),
    );
    children.push(
        row![
            iced::widget::text(format!("RP: {:.2}", p.repeat_penalty)).size(11),
            slider(1.00..=2.00, p.repeat_penalty, move |v| {
                Message::RepeatPenaltyChanged(pane, v)
            })
            .step(0.05)
            .width(Fill),
        ]
        .spacing(4)
        .into(),
    );

    // ── Export/Import/Clear buttons ───────────────────────
    children.push(
        row![
            button(iced::widget::text("📋 JSON").size(11))
                .on_press(Message::ExportJson)
                .padding(iced::Padding::from([4, 8])),
            button(iced::widget::text("📝 MD").size(11))
                .on_press(Message::ExportMarkdown)
                .padding(iced::Padding::from([4, 8])),
            button(iced::widget::text("📄 TXT").size(11))
                .on_press(Message::ExportPlain)
                .padding(iced::Padding::from([4, 8])),
            button(iced::widget::text("📂 Import").size(11))
                .on_press(Message::ImportSession)
                .padding(iced::Padding::from([4, 8])),
            iced::widget::container(iced::widget::text("").size(1))
                .width(Fill)
                .height(iced::Length::Fixed(1.0))
                .style(|_theme: &iced::Theme| iced::widget::container::Style {
                    background: Some(Color::from_rgba8(0x60, 0x60, 0x60, 0.5).into()),
                    border: iced::Border::default(),
                    text_color: None,
                    shadow: iced::Shadow::default(),
                }),
            button(iced::widget::text("🗑 Clear Chat").size(11))
                .style(llama_ui_core::theme::danger_button_style)
                .on_press(Message::ClearChat(pane))
                .padding(iced::Padding::from([4, 8])),
        ]
        .spacing(4)
        .into(),
    );

    // ── Input area ────────────────────────────────────────
    children.push(
        text_input("Type your message...", &p.input_text)
            .on_input(move |s| Message::InputChanged(pane, s))
            .on_submit(Message::Send(pane))
            .padding(10)
            .size(16)
            .into(),
    );

    // ── Send / Cancel buttons ─────────────────────────────
    if p.is_streaming {
        children.push(
            button(iced::widget::text("■ Stop").size(16))
                .style(llama_ui_core::theme::danger_button_style)
                .on_press(Message::CancelGeneration(pane))
                .padding(iced::Padding::from([8, 16]))
                .width(Fill)
                .into(),
        );
    } else {
        let send_label = if p.use_streaming {
            "▶ Send (Stream)"
        } else {
            "▶ Send"
        };
        children.push(
            button(iced::widget::text(send_label).size(16))
                .style(llama_ui_core::theme::primary_button_style)
                .on_press(Message::Send(pane))
                .padding(iced::Padding::from([8, 16]))
                .width(Fill)
                .into(),
        );
    }

    iced::widget::container(scrollable(column(children)).width(Fill).height(Fill))
        .width(Fill)
        .height(Fill)
        .padding(20)
        .into()
}

/// Chat view — renders a row of panes with a separator between them.
fn view_chat(state: &LlamaApp) -> Element<'_, Message> {
    if state.panes.is_empty() {
        return iced::widget::container(iced::widget::text("No panes").size(24))
            .center_x(Fill)
            .center_y(Fill)
            .into();
    }

    let mut pane_views: Vec<Element<'_, Message>> = Vec::new();
    for i in 0..state.panes.len() {
        if !pane_views.is_empty() {
            pane_views.push(vertical_rule(2).into());
        }
        pane_views.push(render_pane(state, i));
    }

    row(pane_views).into()
}

/// Loading view.
fn view_loading(state: &LlamaApp) -> Element<'_, Message> {
    iced::widget::container(
        column(vec![
            iced::widget::text("llama-rs")
                .size(32)
                .color(Color::from_rgba8(0x4A, 0x90, 0xE2, 1.0))
                .into(),
            iced::widget::text("").size(12).into(),
            iced::widget::text("Loading model...")
                .size(20)
                .color(Color::from_rgba8(0xC0, 0xC0, 0xC0, 1.0))
                .into(),
            iced::widget::text("").size(8).into(),
            iced::widget::text(&state.status)
                .size(14)
                .color(Color::from_rgba8(0xA0, 0xA0, 0xA0, 1.0))
                .into(),
            iced::widget::text("").size(8).into(),
            iced::widget::text("This may take a moment on first run.")
                .size(12)
                .color(Color::from_rgba8(0x80, 0x80, 0x80, 1.0))
                .into(),
        ])
        .spacing(4),
    )
    .style(llama_ui_core::theme::content_area_style)
    .center_x(Fill)
    .center_y(Fill)
    .into()
}

/// Error view.
fn view_error(err: &str) -> Element<'_, Message> {
    iced::widget::container(
        column(vec![
            iced::widget::text("llama-rs")
                .size(32)
                .color(Color::from_rgba8(0x4A, 0x90, 0xE2, 1.0))
                .into(),
            iced::widget::text("").size(8).into(),
            iced::widget::text("Error")
                .size(24)
                .color(Color::from_rgba8(0xE2, 0x4A, 0x4A, 1.0))
                .into(),
            iced::widget::text("").size(8).into(),
            iced::widget::container(
                iced::widget::text(err)
                    .size(14)
                    .color(Color::from_rgba8(0xE2, 0x4A, 0x4A, 1.0)),
            )
            .style(llama_ui_core::theme::system_message_style)
            .padding(iced::Padding::from([8, 12]))
            .width(Fill)
            .into(),
            iced::widget::text("").size(16).into(),
            button(iced::widget::text("Back to Model Picker").size(16))
                .style(llama_ui_core::theme::primary_button_style)
                .on_press(Message::ModelSelected(0))
                .padding(iced::Padding::from([10, 20]))
                .into(),
        ])
        .spacing(8),
    )
    .style(llama_ui_core::theme::content_area_style)
    .center_x(Fill)
    .center_y(Fill)
    .padding(40)
    .into()
}

/// Settings view.
fn view_settings(state: &LlamaApp) -> Element<'_, Message> {
    let mut children: Vec<Element<'_, Message>> = Vec::new();

    children.push(
        iced::widget::text("Settings")
            .size(28)
            .color(Color::from_rgba8(0x4A, 0x90, 0xE2, 1.0))
            .into(),
    );
    children.push(iced::widget::text("").size(8).into());

    // System info section
    children.push(
        iced::widget::text("System Information")
            .size(18)
            .color(Color::from_rgba8(0xC0, 0xC0, 0xC0, 1.0))
            .into(),
    );
    children.push(
        iced::widget::container(
            column(vec![
                iced::widget::text(format!("Version: {}", env!("CARGO_PKG_VERSION")))
                    .size(14)
                    .into(),
                iced::widget::text(format!("Active Panes: {}", state.panes.len()))
                    .size(14)
                    .into(),
                iced::widget::text(format!("Available Models: {}", state.models.len()))
                    .size(14)
                    .into(),
            ])
            .spacing(4),
        )
        .style(llama_ui_core::theme::user_message_style)
        .padding(iced::Padding::from([8, 12]))
        .width(Fill)
        .into(),
    );
    children.push(iced::widget::text("").size(12).into());

    // Keyboard shortcuts section
    children.push(
        iced::widget::text("Keyboard Shortcuts")
            .size(18)
            .color(Color::from_rgba8(0xC0, 0xC0, 0xC0, 1.0))
            .into(),
    );
    children.push(
        iced::widget::container(
            column(vec![
                iced::widget::text("Esc — Close Settings / Return to Chat")
                    .size(14)
                    .into(),
                iced::widget::text("F11 — Toggle Full-screen")
                    .size(14)
                    .into(),
                iced::widget::text("Enter — Send message (in text input)")
                    .size(14)
                    .into(),
                iced::widget::text("Ctrl+Enter — Also sends message")
                    .size(14)
                    .into(),
            ])
            .spacing(4),
        )
        .style(llama_ui_core::theme::user_message_style)
        .padding(iced::Padding::from([8, 12]))
        .width(Fill)
        .into(),
    );
    children.push(iced::widget::text("").size(12).into());

    // Features section
    children.push(
        iced::widget::text("Features")
            .size(18)
            .color(Color::from_rgba8(0xC0, 0xC0, 0xC0, 1.0))
            .into(),
    );
    children.push(
        iced::widget::container(
            column(vec![
                iced::widget::text("Streaming & non-streaming modes per pane")
                    .size(14)
                    .into(),
                iced::widget::text("Per-pane backend selection (auto/cpu/cuda)")
                    .size(14)
                    .into(),
                iced::widget::text("Per-pane resource limits (memory/CPU)")
                    .size(14)
                    .into(),
                iced::widget::text("Session export (JSON, Markdown, Plain text)")
                    .size(14)
                    .into(),
                iced::widget::text("Dual-pane mode for model comparison")
                    .size(14)
                    .into(),
                iced::widget::text("Real-time context usage tracking")
                    .size(14)
                    .into(),
            ])
            .spacing(4),
        )
        .style(llama_ui_core::theme::user_message_style)
        .padding(iced::Padding::from([8, 12]))
        .width(Fill)
        .into(),
    );
    children.push(iced::widget::text("").size(16).into());

    // Back to chat
    children.push(
        button(iced::widget::text("Back to Chat").size(16))
            .style(llama_ui_core::theme::primary_button_style)
            .on_press(Message::CloseSettings)
            .padding(iced::Padding::from([10, 20]))
            .width(Fill)
            .into(),
    );

    iced::widget::container(
        iced::widget::scrollable(column(children).spacing(4).width(Fill)).height(Fill),
    )
    .style(llama_ui_core::theme::content_area_style)
    .center_x(Fill)
    .center_y(Fill)
    .padding(40)
    .into()
}

// ─── Drop: stop sandbox on exit ─────────────────────────────────────────

impl Drop for LlamaApp {
    fn drop(&mut self) {
        for pane in &mut self.panes {
            if let Some(ref mut client) = pane.sandbox {
                client.stop();
            }
        }
    }
}

/// Initialize and run the application.
impl LlamaApp {
    /// Load models from manifest.
    fn load_models() -> Vec<ModelInfo> {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("llama-ui");
        let models_dir = data_dir.join("models");
        let manifest_path = data_dir.join("models.json");

        if let Ok(manifest) = Manifest::load(&manifest_path) {
            manifest
                .models
                .into_iter()
                .map(|m| ModelInfo {
                    name: m.name,
                    path: m.path,
                })
                .collect()
        } else {
            // Scan directory for .gguf files
            let mut models = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().is_some_and(|ext| ext == "gguf") {
                        if let Some(name) = path.file_stem() {
                            models.push(ModelInfo {
                                name: name.to_string_lossy().to_string(),
                                path,
                            });
                        }
                    }
                }
            }
            models
        }
    }

    /// Run the application.
    pub fn run() -> iced::Result {
        iced::application("llama-ui", update, view)
            .theme(theme)
            .subscription(subscription)
            .run_with(|| {
                let models = LlamaApp::load_models();
                (
                    LlamaApp {
                        state: AppState::default(),
                        models,
                        panes: Vec::new(),
                        active_pane: 0,
                        status: String::new(),
                    },
                    Task::none(),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaApp, ModelInfo, view_model_picker};
    use iced::Element;

    #[test]
    fn model_picker_has_buttons_for_each_model() {
        let mut app = LlamaApp::default();
        app.models = vec![
            ModelInfo {
                name: "Model‑A".into(),
                path: std::path::PathBuf::from("/tmp/a.gguf"),
            },
            ModelInfo {
                name: "Model‑B".into(),
                path: std::path::PathBuf::from("/tmp/b.gguf"),
            },
        ];
        let _: Element<'_, _> = view_model_picker(&app);
    }
}
