# Iced 0.13 API Reference for llama-ui

- **Platform support**: Works on Linux with both X11 and Wayland backends, as well as macOS and Windows (via iced).

This document summarizes the Iced 0.13 API patterns used in the llama-ui crate.
The llama-ui application follows The Elm Architecture pattern with four core
concepts:

- **State**: Application data
- **Messages**: User interactions or events
- **View logic**: How to display state as widgets
- **Update logic**: How to react to messages and update state

## Key Changes from Iced 0.12 to 0.13

Iced 0.13 introduced breaking API changes:

- Removed the `Application` trait → function-based `iced::application()` builder
- `Command<Message>` replaced by `Task<Message>`
- `Appearance` trait removed — themes return `Theme` directly
- `Renderer` generics removed from `Element` in user code

## Core API Components

### Application Creation

```rust
use iced::{Element, Fill, Subscription, Task, Theme};

fn main() -> iced::Result {
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
```

#### `iced::application()` Builder

- Signature: `iced::application(title, update, view)` → builder
- Builder methods: `.theme(fn)`, `.subscription(fn)`, `.run_with(fn)`, `.run()`
- Title is a `&str` shown in the window title bar
- Returns `iced::Result` (Ok on clean exit, Err on fatal error)

#### Boot Function (via `run_with`)

- State initializer: `fn() -> (State, Task<Message>)`
- Called once at application start
- `Task::none()` when no initial async work is needed
- Do NOT use `..Default::default()` if the state type implements `Drop`
  (struct update syntax moves fields, which is illegal for `Drop` types)

#### Update Function

```rust
pub fn update(state: &mut LlamaApp, message: Message) -> Task<Message> {
    match message {
        Message::InputChanged(pane, text) => {
            state.panes[pane].input_text = text;
            Task::none()
        }
        Message::TemperatureChanged(pane, value) => {
            state.panes[pane].temperature = value;
            fire_update_sampler(state, pane)
        }
        // ...
    }
}
```

- Signature: `fn(&mut State, Message) -> Task<Message>`
- Return `Task::perform(...)` for async operations (HTTP requests, spawns)
- Return `Task::none()` for synchronous state changes

#### View Function

```rust
pub fn view(state: &LlamaApp) -> Element<'_, Message> {
    match &state.state {
        AppState::ModelPicker => view_model_picker(state),
        AppState::Chat => view_chat(state),
        AppState::Loading => view_loading(state),
        AppState::Error(err) => view_error(err),
    }
}
```

- Signature: `fn(&State) -> Element<'_, Message>`
- Pure: no side effects, called every frame
- Widgets are created fresh each call (iced diffs internally)

### Theme Function

```rust
pub fn theme(_state: &LlamaApp) -> Theme {
    Theme::default()
}
```

- Signature: `fn(&State) -> Theme`
- Called when theme needs to refresh
- Can return different themes per app state

### Subscription Function

```rust
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
```

- Signature: `fn(&State) -> Subscription<Message>`
- Returns a declaration of external events to listen to
- Combine multiple subscriptions with `Subscription::batch(subs)`

#### SSE Stream Subscription

```rust
fn pane_subscription(pane: usize, p: &ChatPane) -> Subscription<Message> {
    if !p.is_streaming || p.server_address.is_empty() {
        return Subscription::none();
    }
    let id = format!("sse-{}-{}", pane, p.session.messages.len());
    Subscription::run_with_id(
        id,
        stream::channel(32, move |mut output| async move {
            // ... send messages via output.send(Message::StreamChunk(pane, text))
        }),
    )
}
```

- `Subscription::run_with_id(id, stream)` — unique ID controls restart behavior
- `stream::channel(buffer_size, producer_fn)` — creates a `Stream` implementation
- Producer receives `Sink` to push messages into the stream
- A new subscription with the same ID replaces the old one (key for re-sending)

## Messages

Messages carry `usize` pane index when they target a specific chat pane:

```rust
#[derive(Debug, Clone)]
pub enum Message {
    // App-level
    ModelSelected(usize),
    StartChat(usize),
    Error(String),

    // Stream events (per-pane)
    StreamChunk(usize, String),
    StreamEnded(usize),

    // Input (per-pane)
    InputChanged(usize, String),
    SendMessage(usize),
    CancelGeneration(usize),

    // Sampler (per-pane)
    TemperatureChanged(usize, f32),
    TopKChanged(usize, f32),
    TopPChanged(usize, f32),
    RepeatPenaltyChanged(usize, f32),

    // Backend (per-pane)
    BackendChanged(usize, String),

    // Session export/import (active pane)
    ExportJson,
    ExportMarkdown,
    ExportPlain,
    ImportSession,

    // Sandbox lifecycle
    SandboxStatus(SandboxStatus),
}
```

## Widget Usage

| Widget | Import | Usage |
|---|---|---|
| `button` | `iced::widget::button` | `button(text("Click")).on_press(Message::Foo)` |
| `column` | `iced::widget::column` | `column(children)` or `column![w1, w2]` |
| `row` | `iced::widget::row` | `row(children)` or `row![w1, w2]` |
| `container` | `iced::widget::container` | `container(child).padding(20).center_x(Fill)` |
| `scrollable` | `iced::widget::scrollable` | `scrollable(content).width(Fill).height(Fill)` |
| `text` | `iced::widget::text` | `text("Hello").size(16)` |
| `text_input` | `iced::widget::text_input` | `text_input("Placeholder", &value).on_input(\|s\| Msg::Input(s)).on_submit(Msg::Send)` |
| `slider` | `iced::widget::slider` | `slider(0.0..=1.0, value, \|v\| Msg::Changed(v)).step(0.01)` |
| `pick_list` | `iced::widget::pick_list` | `pick_list(options, selected, \|s\| Msg::Selected(s.to_string()))` |
| `vertical_rule` | `iced::widget::vertical_rule` | `vertical_rule(2)` — divider between panes |
| `vertical_space` | `iced::widget::vertical_space` | Spacer filling remaining space |

### Layout Helpers

```rust
use iced::widget::{column, row, container, scrollable, vertical_rule};

// Pane divider (M10 dual-model):
row![
    render_pane(state, 0),
    vertical_rule(2),   // 2px wide divider
    render_pane(state, 1),
]

// Fill remaining horizontal/vertical space:
use iced::Fill;
container(child).width(Fill).height(Fill)
```

## Task System

`Task<Message>` replaces `Command<Message>` from iced ≤0.12.

### No-Op Task

```rust
Task::none()
```

### Async HTTP Request

```rust
Task::perform(
    async move {
        let client = reqwest::Client::new();
        let resp = client.post(url).json(&body).send().await;
        // ...
    },
    |result| Message::SomeVariant(result),
)
```

### Sandbox Spawn (multi-step async)

```rust
Task::perform(
    async move {
        let mut client = SandboxClient::spawn(model_path, backend).await?;
        client.wait_for_health(Duration::from_secs(30)).await?;
        Ok::<_, String>(client)
    },
    |result| match result {
        Ok(client) => Message::SandboxReady(pane, client),
        Err(e) => Message::Error(format!("Sandbox failed: {e}")),
    },
)
```

## Architecture Patterns

### Dual-Model (M10)

State is split into per-pane (`ChatPane`) and shared (`LlamaApp`):

```
LlamaApp
 ├── models: Vec<ModelInfo>          (shared — model catalog)
 ├── panes: Vec<ChatPane>            (per-pane state)
 ├── active_pane: usize              (which pane is focused)
 ├── state: AppState                 (ModelPicker | Chat | Loading | Error)
 └── status: String                  (status bar text)

ChatPane
 ├── session: Session                (chat history)
 ├── input_text: String              (current input)
 ├── sandbox: Option<SandboxClient>  (server process)
 ├── server_address: String          ("http://127.0.0.1:PORT")
 ├── is_streaming: bool
 ├── cancelled: Arc<AtomicBool>
 ├── temperature, top_k, top_p, repeat_penalty
 └── backend: String                 ("auto" | "cpu" | "cuda")
```

- `render_pane(state, pane)` renders one pane's widgets
- `view_chat()` wraps in `row![render_pane(0), vertical_rule, render_pane(1)]`
- Messages carry `usize` pane index to target the right pane
- Subscription creates one SSE stream per pane, batched via `Subscription::batch`

### Session Export (M9)

```rust
fn session_export(
    state: &mut LlamaApp,
    filename: &str,         // default filename
    label: &str,            // filter label
    exts: &[&str],          // file extensions
    f: fn(&Session, &PathBuf) -> Result<(), ExportError>,
) -> Task<Message>;
```

Uses `rfd::FileDialog` for native save dialogs. Operates on `active_pane`'s
session. Synchronous (modal) on Linux — `rfd` blocks briefly.

### Sampler Sync (M7)

```rust
fn fire_update_sampler(state: &LlamaApp, pane: usize) -> Task<Message> {
    // POST pane's sampler config to pane's server
    Task::perform(async move { /* HTTP POST */ }, |_| Message::SamplerUpdated)
}
```

Each slider change fires immediately to the server. Sampler params duplicated
in `/completion` body (belt-and-suspenders).

### Context Overflow Warning

- **80%**: yellow warning — "approaching limit"
- **95%**: red warning — "consider clearing history"

Checked in `render_pane()` from `p.total_tokens / p.context_limit`.

## References

- Official Iced Documentation: <https://docs.iced.rs/>
- Iced 0.13 Release Notes: <https://github.com/iced-rs/iced/releases/tag/0.13.0>
- Iced Book: <https://book.iced.rs/>
