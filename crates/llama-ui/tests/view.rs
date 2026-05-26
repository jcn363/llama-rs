use llama_ui::app::{view_model_picker, LlamaApp, ModelInfo};
use iced::{Element, Command};

#[test]
fn model_picker_has_buttons_for_each_model() {
    // Build a minimal app with two dummy models.
    let mut app = LlamaApp::default();
    app.models = vec![
        ModelInfo { name: "Model‑A".into(), path: std::path::PathBuf::from("/tmp/a.gguf") },
        ModelInfo { name: "Model‑B".into(), path: std::path::PathBuf::from("/tmp/b.gguf") },
    ];
    // The view should be constructable without panicking.
    let _: Element<'_, _> = view_model_picker(&app);
}
