//! Main iced application state machine.

use iced::widget::{button, column, container, scrollable, text, text_input};
use iced::{Element, Fill, Subscription, Task, Theme};
use llama_ui_models::Manifest;
use llama_ui_session::{ChatMessage, Role, Session};
use llama_ui_sandbox_client::SandboxStatus;
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
    /// Loading model.
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
    /// Sandbox status updated.
    SandboxStatus(SandboxStatus),
    /// Error occurred.
    Error(String),
}

/// Update the application state.
pub fn update(state: &mut LlamaApp, message: Message) -> Task<Message> {
    match message {
        Message::ModelSelected(idx) => {
            state.selected_model = idx;
        }
        Message::StartChat => {
            if !state.models.is_empty() {
                state.state = AppState::Loading;
                state.status = "Starting model...".to_string();
                // In M5+, we'll spawn sandbox here
                state.state = AppState::Chat;
                state.session = Session::new(&state.models[state.selected_model].name);
            }
        }
        Message::SendMessage => {
            if !state.input_text.is_empty() {
                // Add user message
                state.session.add_message(ChatMessage {
                    role: Role::User,
                    content: state.input_text.clone(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    token_count: None,
                });
                state.input_text.clear();
                state.status = "Generating...".to_string();
                // In M5+, send to sandbox
            }
        }
        Message::InputChanged(text) => {
            state.input_text = text;
        }
        Message::SandboxStatus(status) => {
            state.status = format!("Sandbox: {:?}", status);
        }
        Message::Error(err) => {
            state.state = AppState::Error(err);
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

/// Subscription for events.
pub fn subscription(_state: &LlamaApp) -> Subscription<Message> {
    Subscription::none()
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

    // Messages
    for msg in &state.session.messages {
        let role = match msg.role {
            Role::User => "You",
            Role::Assistant => "AI",
            Role::System => "System",
        };
        children.push(text(format!("{}: {}", role, msg.content)).size(14).into());
    }

    // Input area
    children.push(
        text_input("Type your message...", &state.input_text)
            .on_input(Message::InputChanged)
            .padding(10)
            .size(16)
            .into(),
    );

    children.push(
        button(text("Send").size(16))
            .on_press(Message::SendMessage)
            .padding(10)
            .into(),
    );

    children.push(text(&state.status).size(12).into());

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
        ]),
    )
    .center_x(Fill)
    .center_y(Fill)
    .into()
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
                (
                    LlamaApp {
                        models: LlamaApp::load_models(),
                        ..Default::default()
                    },
                    Task::none(),
                )
            })
    }
}
