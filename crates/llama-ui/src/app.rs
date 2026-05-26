//! Main iced application state machine.
//!
//! Manages sandbox lifecycle (spawn, health, stop), chat sessions,
//! and context tracking. M5+ with M6 non-streaming /completion.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::sink::SinkExt;
use iced::stream;
use iced::widget::{button, column, container, scrollable, text, text_input};
use iced::{Element, Fill, Subscription, Task, Theme};
use llama_ui_models::Manifest;
use llama_ui_sandbox_client::SandboxClient;
use llama_ui_session::{ChatMessage, Role, Session};
use std::path::PathBuf;

/// Application state.
#[derive(Debug)]
pub struct LlamaApp {
    /// Current view state.
    state: AppState,
    /// Available models.
    models: Vec<ModelInfo>,
    /// Selected model index.
    selected_model: usize,
    /// Current chat session.
    session: Session,
    /// UI message input.
    input_text: String,
    /// Status message.
    status: String,
    /// Sandbox client for llama-server process.
    sandbox: Option<SandboxClient>,
    /// Server base URL (set after sandbox starts).
    server_address: String,
    /// Whether an SSE stream is active.
    is_streaming: bool,
    /// Cancellation flag for in-flight generation.
    cancelled: Arc<AtomicBool>,
    /// Running total of tokens used in this session.
    total_tokens: usize,
    /// Model's context limit (n_ctx).
    context_limit: usize,
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
            selected_model: 0,
            session: Session::new(""),
            input_text: String::new(),
            status: String::new(),
            sandbox: None,
            server_address: String::new(),
            is_streaming: false,
            cancelled: Arc::new(AtomicBool::new(false)),
            total_tokens: 0,
            context_limit: 4096,
        }
    }
}

/// Messages that update the application state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Message {
    /// Model selected from picker.
    ModelSelected(usize),
    /// Start chat with selected model.
    StartChat,
    /// Send message to model.
    SendMessage,
    /// Input text changed.
    InputChanged(String),
    /// Sandbox status changed.
    SandboxStatus(String),
    /// Error occurred.
    Error(String),
    // ─── M6 additions ───────────────────────────────────────
    /// Non-streaming completion response received.
    CompletionReceived(String),
    /// Token count from /tokenize.
    TokenCounted(usize),
    /// Sandbox successfully started on the given port.
    SandboxStarted(u16),
    /// Cancel in-flight generation.
    CancelGeneration,
    /// One chunk from SSE streaming.
    StreamChunk(String),
    /// SSE stream finished cleanly.
    StreamEnded,
}

/// Update the application state.
pub fn update(state: &mut LlamaApp, message: Message) -> Task<Message> {
    match message {
        // ─── Model selection ─────────────────────────────────
        Message::ModelSelected(idx) => {
            state.selected_model = idx;
            Task::none()
        }

        // ─── Start chat: spawn sandbox ───────────────────────
        Message::StartChat => {
            if state.models.is_empty() {
                return Task::none();
            }
            state.state = AppState::Loading;
            state.status = "Starting model...".to_string();

            let model = state.models[state.selected_model].clone();

            Task::perform(
                async move {
                    let binary = SandboxClient::resolve_binary().map_err(|e| e.to_string())?;
                    let mut client =
                        SandboxClient::new(binary, model.path, "auto", "llama-ui");
                    client.spawn().map_err(|e| e.to_string())?;
                    client
                        .wait_for_ready(Duration::from_secs(30))
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<u16, String>(client.port)
                },
                |result| match result {
                    Ok(port) => Message::SandboxStarted(port),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── Sandbox started ─────────────────────────────────
        Message::SandboxStarted(port) => {
            state.server_address = format!("http://127.0.0.1:{}", port);
            state.state = AppState::Chat;
            state.status = format!("Ready (port {})", port);
            state.session = Session::new(&state.models[state.selected_model].name);
            state.total_tokens = 0;

            // Re-create SandboxClient for management in state
            let model = state.models[state.selected_model].clone();
            match SandboxClient::resolve_binary() {
                Ok(binary) => {
                    let mut client =
                        SandboxClient::new(binary, model.path, "auto", "llama-ui");
                    // We can't re-spawn, but we store it for health/stop
                    client.port = port;
                    state.sandbox = Some(client);
                }
                Err(e) => {
                    state.status = format!("Warning: {}", e);
                }
            }
            Task::none()
        }

        // ─── Send message (non-streaming) ────────────────────
        Message::SendMessage => {
            if state.input_text.is_empty() || state.server_address.is_empty() {
                return Task::none();
            }

            // Build conversation history as prompt
            let prompt = state
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
                .chain(std::iter::once(format!("User: {}", state.input_text)))
                .collect::<Vec<_>>()
                .join("\n");

            // Add user message to session
            state.session.add_message(ChatMessage {
                role: Role::User,
                content: state.input_text.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                token_count: None,
            });
            state.input_text.clear();
            state.status = "Generating...".to_string();

            let addr = state.server_address.clone();
            let max_tokens = 512usize;

            Task::perform(
                async move {
                    let client = reqwest::Client::new();
                    let resp = client
                        .post(format!("{}/completion", addr))
                        .json(&serde_json::json!({
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "stream": false,
                        }))
                        .send()
                        .await
                        .map_err(|e| e.to_string())?;
                    let body: serde_json::Value =
                        resp.json().await.map_err(|e| e.to_string())?;
                    let content = body["content"].as_str().unwrap_or("").to_string();
                    Ok::<String, String>(content)
                },
                |result| match result {
                    Ok(content) => Message::CompletionReceived(content),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── Completion received (non-streaming) ─────────────
        Message::CompletionReceived(content) => {
            // Add assistant message
            state.session.add_message(ChatMessage {
                role: Role::Assistant,
                content: content.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                token_count: None,
            });
            state.status = String::new();

            // Fire /tokenize to update context count
            let addr = state.server_address.clone();
            let full_text = state
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
                |result| match result {
                    Ok(count) => Message::TokenCounted(count),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── Token count updated ─────────────────────────────
        Message::TokenCounted(count) => {
            state.total_tokens = count;
            let limit = state.context_limit;
            let pct = if limit > 0 {
                (count as f64 / limit as f64) * 100.0
            } else {
                0.0
            };
            if pct > 95.0 {
                state.status = format!(
                    "⚠️ Context at {}/{} tokens ({:.0}%) — consider clearing history",
                    count, limit, pct
                );
            } else if pct > 80.0 {
                state.status = format!(
                    "⚠️ Context at {}/{} tokens ({:.0}%) — approaching limit",
                    count, limit, pct
                );
            } else if count > 0 {
                state.status = format!("Tokens: {}/{} ({:.0}%)", count, limit, pct);
            }
            Task::none()
        }

        // ─── Cancel generation ───────────────────────────────
        Message::CancelGeneration => {
            state.cancelled.store(true, Ordering::Relaxed);
            state.is_streaming = false;
            state.status = "Cancelled.".to_string();
            Task::none()
        }

        // ─── SSE streaming chunk ─────────────────────────────
        Message::StreamChunk(text) => {
            // Append to last assistant message
            if let Some(last) = state.session.messages.last_mut() {
                if matches!(last.role, Role::Assistant) {
                    last.content.push_str(&text);
                }
            }
            Task::none()
        }

        // ─── SSE stream ended ────────────────────────────────
        Message::StreamEnded => {
            state.is_streaming = false;
            state.status = String::new();
            // Fire /tokenize to update context count
            let addr = state.server_address.clone();
            let full_text = state
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
                |result| match result {
                    Ok(count) => Message::TokenCounted(count),
                    Err(e) => Message::Error(e),
                },
            )
        }

        // ─── Input changed ───────────────────────────────────
        Message::InputChanged(text) => {
            state.input_text = text;
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
            state.is_streaming = false;
            Task::none()
        }
    }
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

/// Subscription for SSE streaming.
pub fn subscription(state: &LlamaApp) -> Subscription<Message> {
    if !state.is_streaming || state.server_address.is_empty() {
        return Subscription::none();
    }

    let addr = state.server_address.clone();
    let cancelled = state.cancelled.clone();

    // Build prompt from full conversation history
    let prompt = state
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
        .chain(std::iter::once(format!(
            "User: {}",
            state.input_text
        )))
        .collect::<Vec<_>>()
        .join("\n");

    // Use a session-length-based ID so the subscription restarts
    // when a new message is sent.
    let id = format!("sse-{}", state.session.messages.len());

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
                                                        .send(Message::StreamEnded)
                                                        .await;
                                                    return;
                                                }
                                                if let Some(content) =
                                                    val["content"].as_str()
                                                {
                                                    let _ = output
                                                        .send(Message::StreamChunk(
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
                        let _ = output.send(Message::StreamEnded).await;
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
fn view_chat(state: &LlamaApp) -> Element<'_, Message> {
    let mut children = Vec::new();

    // Context overflow warning
    if state.context_limit > 0 {
        let pct = state.total_tokens as f64 / state.context_limit as f64;
        if pct > 0.80 {
            let warning = if pct > 0.95 {
                format!(
                    "⚠️ Context at {}/{} tokens — consider clearing history",
                    state.total_tokens, state.context_limit
                )
            } else {
                format!(
                    "⚠️ Context at {}/{} tokens ({:.0}%) — approaching limit",
                    state.total_tokens,
                    state.context_limit,
                    pct * 100.0
                )
            };
            children.push(text(warning).size(13).into());
        }
    }

    // Messages
    for msg in &state.session.messages {
        let role = match msg.role {
            Role::User => "You",
            Role::Assistant => "AI",
            Role::System => "System",
        };
        children.push(text(format!("{}: {}", role, msg.content)).size(14).into());
    }

    // Token counter
    if state.total_tokens > 0 {
        children.push(
            text(format!(
                "Tokens: {}/{} ({:.0}%)",
                state.total_tokens,
                state.context_limit,
                if state.context_limit > 0 {
                    (state.total_tokens as f64 / state.context_limit as f64) * 100.0
                } else {
                    0.0
                }
            ))
            .size(12)
            .into(),
        );
    }

    // Input area
    children.push(
        text_input("Type your message...", &state.input_text)
            .on_input(Message::InputChanged)
            .on_submit(Message::SendMessage)
            .padding(10)
            .size(16)
            .into(),
    );

    // Send / Cancel buttons row
    if state.is_streaming {
        children.push(
            button(text("■ Stop").size(16))
                .on_press(Message::CancelGeneration)
                .padding(10)
                .into(),
        );
    } else {
        children.push(
            button(text("Send").size(16))
                .on_press(Message::SendMessage)
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
        if let Some(ref mut client) = self.sandbox {
            client.stop();
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
                        models,
                        state: AppState::default(),
                        selected_model: 0,
                        session: Session::new(""),
                        input_text: String::new(),
                        status: String::new(),
                        sandbox: None,
                        server_address: String::new(),
                        is_streaming: false,
                        cancelled: Arc::new(AtomicBool::new(false)),
                        total_tokens: 0,
                        context_limit: 4096,
                    },
                    Task::none(),
                )
            })
    }
}
