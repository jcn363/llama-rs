//! Main iced application state machine.
//!
//! Manages sandbox lifecycle (spawn, health, stop), chat sessions,
//! and context tracking. M5+ with M6 non-streaming /completion.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::sink::SinkExt;
use iced::stream;
use iced::widget::{
    button, column, container, pick_list, row, scrollable, slider, text, text_input,
    vertical_rule,
};
use iced::{Element, Fill, Subscription, Task, Theme};
use llama_ui_models::Manifest;
use llama_ui_sandbox_client::SandboxClient;
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
    /// Running token count.
    pub total_tokens: usize,
    /// Model context limit.
    pub context_limit: usize,
    /// Backend ("auto", "cpu", "cuda").
    pub backend: String,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k (0 = disabled).
    pub top_k: f32,
    /// Top-p.
    pub top_p: f32,
    /// Repeat penalty.
    pub repeat_penalty: f32,
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
            total_tokens: 0,
            context_limit: 4096,
            backend: "auto".to_string(),
            temperature: 0.8,
            top_k: 40.0,
            top_p: 0.95,
            repeat_penalty: 1.1,
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

/// Model information for display.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelInfo {
    /// Human-readable model name.
    pub name: String,
    /// Path to the GGUF file.
    pub path: PathBuf,
}

/// Application state machine states.
#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    /// Model picker view.
    ModelPicker,
    /// Chat view.
    Chat,
    /// Loading model / starting sandbox.
    Loading,
    /// Error state.
    Error(String),
}

impl Default for AppState {
    fn default() -> Self {
        AppState::ModelPicker
    }
}

impl Default for LlamaApp {
    fn default() -> Self {
        LlamaApp {
            state: AppState::default(),
            models: Vec::new(),
            panes: Vec::new(),
            active_pane: 0,
            status: String::new(),
        }
    }
}

/// Messages that update the application state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Message {
    /// Model selected from picker.
    ModelSelected(usize),
    /// Start first chat pane with selected model.
    StartChat,
    /// Add a second chat pane (M10 dual-model).
    AddPane(usize),
    /// Send message on a pane.
    SendMessage(usize),
    /// Input text changed on a pane.
    InputChanged(usize, String),
    /// Sandbox status message.
    SandboxStatus(String),
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
    /// Session import/export error.
    ExportError(String),
    // ─── M7: Sampler slider messages (pane-indexed) ──────────
    /// Temperature slider changed on a pane.
    TemperatureChanged(usize, f32),
    /// Top-k slider changed on a pane.
    TopKChanged(usize, f32),
    /// Top-p slider changed on a pane.
    TopPChanged(usize, f32),
    /// Repeat penalty slider changed on a pane.
    RepeatPenaltyChanged(usize, f32),
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
            let model = state.models[0].clone();

            state.panes.push(ChatPane::new(0, &model.name));
            let pane = state.panes.len() - 1;
            let backend = state.panes[pane].backend.clone();

            Task::perform(
                async move {
                    let binary = SandboxClient::resolve_binary().map_err(|e| e.to_string())?;
                    let mut client =
                        SandboxClient::new(binary, model.path, &backend, "llama-ui");
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

        // ─── Add a second pane (M10 dual-model) ──────────────
        Message::AddPane(model_idx) => {
            if model_idx >= state.models.len() || state.panes.len() >= 2 {
                return Task::none();
            }
            let model = state.models[model_idx].clone();
            state.panes.push(ChatPane::new(model_idx, &model.name));
            let pane = state.panes.len() - 1;
            let backend = state.panes[pane].backend.clone();

            Task::perform(
                async move {
                    let binary = SandboxClient::resolve_binary().map_err(|e| e.to_string())?;
                    let mut client =
                        SandboxClient::new(binary, model.path, &backend, "llama-ui");
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
                    let mut client =
                        SandboxClient::new(binary, model.path, &p.backend, "llama-ui");
                    client.port = port;
                    p.sandbox = Some(client);
                }
                Err(e) => {
                    state.status = format!("Warning: {}", e);
                }
            }
            Task::none()
        }

        // ─── Send message (non-streaming) on a pane ──────────
        Message::SendMessage(pane) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            if p.input_text.is_empty() || p.server_address.is_empty() {
                return Task::none();
            }

            // Build conversation history as prompt
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

            p.session.add_message(ChatMessage {
                role: Role::User,
                content: p.input_text.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                token_count: None,
            });
            p.input_text.clear();
            state.status = format!("Pane {} generating...", pane);

            let addr = p.server_address.clone();
            let max_tokens = 512usize;
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
                            "max_tokens": max_tokens,
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

        // ─── Completion received (non-streaming) on a pane ─────
        Message::CompletionReceived(pane, content) => {
            if pane >= state.panes.len() {
                return Task::none();
            }
            let p = &mut state.panes[pane];
            p.session.add_message(ChatMessage {
                role: Role::Assistant,
                content: content.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                token_count: None,
            });
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
                    let body: serde_json::Value =
                        resp.json().await.map_err(|e| e.to_string())?;
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
                    let body: serde_json::Value =
                        resp.json().await.map_err(|e| e.to_string())?;
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

        // ─── Sandbox status ──────────────────────────────────
        Message::SandboxStatus(status) => {
            state.status = format!("Sandbox: {}", status);
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

            Task::perform(
                async move {
                    let binary =
                        SandboxClient::resolve_binary().map_err(|e| e.to_string())?;
                    let mut client =
                        SandboxClient::new(binary, model.path, &backend, "llama-ui");
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
        Message::ExportError(_) => {
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
            let body: serde_json::Value =
                resp.json().await.map_err(|e| e.to_string())?;
            Ok::<String, String>(body.to_string())
        },
        |result| match result {
            Ok(_) => Message::SandboxStatus("Samplers updated".to_string()),
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
                                                serde_json::from_str::<serde_json::Value>(
                                                    data,
                                                )
                                            {
                                                if val["stop"].as_bool().unwrap_or(false)
                                                {
                                                    let _ = output
                                                        .send(Message::StreamEnded(pane))
                                                        .await;
                                                    return;
                                                }
                                                if let Some(content) =
                                                    val["content"].as_str()
                                                {
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
                        let _ = output
                            .send(Message::Error(e.to_string()))
                            .await;
                    }
                }
            }
        }),
    )
}

/// Subscription for SSE streaming across all panes.
pub fn subscription(state: &LlamaApp) -> Subscription<Message> {
    let mut subs = Vec::new();
    for (i, p) in state.panes.iter().enumerate() {
        subs.push(pane_subscription(i, p));
    }
    match subs.len() {
        0 => Subscription::none(),
        1 => subs.into_iter().next().unwrap(),
        _ => Subscription::batch(subs),
    }
}

// ---- View helpers ----

/// View for model picker.
fn view_model_picker(state: &LlamaApp) -> Element<'_, Message> {
    let mut children = vec![
        text("llama-ui").size(32).into(),
        text(&state.status).size(16).into(),
    ];

    for (i, model) in state.models.iter().enumerate() {
        let btn = button(text(&model.name).size(18))
            .on_press(Message::ModelSelected(i))
            .padding(10);
        children.push(btn.into());
    }

    children.push(
        button(text("Start Chat").size(20))
            .on_press(Message::StartChat)
            .padding(20)
            .into(),
    );

    container(column(children))
        .center_x(Fill)
        .center_y(Fill)
        .padding(20)
        .into()
}

/// View for chat interface.
/// Render a single chat pane.
fn render_pane(state: &LlamaApp, pane: usize) -> Element<'_, Message> {
    let p = &state.panes[pane];
    let mut children = Vec::new();

    // Context overflow warning
    if p.context_limit > 0 {
        let pct = p.total_tokens as f64 / p.context_limit as f64;
        if pct > 0.80 {
            let warning = if pct > 0.95 {
                format!(
                    "⚠️ Context at {}/{} tokens — consider clearing history",
                    p.total_tokens, p.context_limit
                )
            } else {
                format!(
                    "⚠️ Context at {}/{} tokens ({:.0}%) — approaching limit",
                    p.total_tokens, p.context_limit, pct * 100.0
                )
            };
            children.push(text(warning).size(13).into());
        }
    }

    // Backend selector
    children.push(
        row![
            text("Backend:").size(14),
            pick_list(
                vec!["auto", "cpu", "cuda"],
                Some(p.backend.as_str()),
                move |s: &str| Message::BackendChanged(pane, s.to_string()),
            )
            .width(150),
        ]
        .spacing(8)
        .into(),
    );

    // Messages
    for msg in &p.session.messages {
        let role = match msg.role {
            Role::User => "You",
            Role::Assistant => "AI",
            Role::System => "System",
        };
        children.push(text(format!("{}: {}", role, msg.content)).size(14).into());
    }

    // Token counter
    if p.total_tokens > 0 {
        children.push(
            text(format!(
                "Tokens: {}/{} ({:.0}%)",
                p.total_tokens,
                p.context_limit,
                if p.context_limit > 0 {
                    (p.total_tokens as f64 / p.context_limit as f64) * 100.0
                } else {
                    0.0
                }
            ))
            .size(12)
            .into(),
        );
    }

    // ─── M7: Sampler sliders ────────────────────────────────
    children.push(
        row![
            text(format!("T: {:.2}", p.temperature)).size(12),
            slider(0.0..=2.0, p.temperature, move |v| Message::TemperatureChanged(pane, v))
                .step(0.05)
                .width(Fill),
        ]
        .spacing(4)
        .into(),
    );
    children.push(
        row![
            text(format!("K: {}", p.top_k)).size(12),
            slider(0.0..=100.0, p.top_k, move |v| Message::TopKChanged(pane, v))
                .step(1.0)
                .width(Fill),
        ]
        .spacing(4)
        .into(),
    );
    children.push(
        row![
            text(format!("P: {:.2}", p.top_p)).size(12),
            slider(0.00..=1.00, p.top_p, move |v| Message::TopPChanged(pane, v))
                .step(0.01)
                .width(Fill),
        ]
        .spacing(4)
        .into(),
    );
    children.push(
        row![
            text(format!("RP: {:.2}", p.repeat_penalty)).size(12),
            slider(1.00..=2.00, p.repeat_penalty, move |v| Message::RepeatPenaltyChanged(pane, v))
                .step(0.05)
                .width(Fill),
        ]
        .spacing(4)
        .into(),
    );

    // ─── M9: Export/Import buttons ──────────────────────────
    children.push(
        row![
            button(text("Export JSON").size(12))
                .on_press(Message::ExportJson)
                .padding(6),
            button(text("Export MD").size(12))
                .on_press(Message::ExportMarkdown)
                .padding(6),
            button(text("Export TXT").size(12))
                .on_press(Message::ExportPlain)
                .padding(6),
            button(text("Import").size(12))
                .on_press(Message::ImportSession)
                .padding(6),
        ]
        .spacing(6)
        .into(),
    );

    // Input area
    children.push(
        text_input("Type your message...", &p.input_text)
            .on_input(move |s| Message::InputChanged(pane, s))
            .on_submit(Message::SendMessage(pane))
            .padding(10)
            .size(16)
            .into(),
    );

    // Send / Cancel buttons row
    if p.is_streaming {
        children.push(
            button(text("■ Stop").size(16))
                .on_press(Message::CancelGeneration(pane))
                .padding(10)
                .into(),
        );
    } else {
        children.push(
            button(text("Send").size(16))
                .on_press(Message::SendMessage(pane))
                .padding(10)
                .into(),
        );
    }

    // Status
    if !state.status.is_empty() {
        children.push(text(&state.status).size(12).into());
    }

    container(scrollable(column(children)).width(Fill).height(Fill))
        .width(Fill)
        .height(Fill)
        .padding(20)
        .into()
}

/// Chat view — renders a row of panes with a separator between them.
fn view_chat(state: &LlamaApp) -> Element<'_, Message> {
    if state.panes.is_empty() {
        return container(text("No panes").size(24))
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
    container(
        column(vec![
            text("Loading...").size(24).into(),
            text(&state.status).size(16).into(),
        ]),
    )
    .center_x(Fill)
    .center_y(Fill)
    .into()
}

/// Error view.
fn view_error(err: &str) -> Element<'_, Message> {
    container(
        column(vec![
            text("Error").size(24).into(),
            text(err).size(16).into(),
            button(text("Back").size(16))
                .on_press(Message::ModelSelected(0))
                .padding(10)
                .into(),
        ]),
    )
    .center_x(Fill)
    .center_y(Fill)
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
