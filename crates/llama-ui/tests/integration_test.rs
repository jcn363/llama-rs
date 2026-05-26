//! Integration tests for llama-ui
//!
//! These tests verify that the UI components can:
//! 1. Create and manage chat sessions
//! 2. Create model entries
//! 3. Export sessions to various formats

use std::path::PathBuf;

#[test]
fn test_model_entry_creation() {
    use llama_ui_models::ModelEntry;

    let entry = ModelEntry {
        name: "Tiny LLM".to_string(),
        filename: "tiny-llm-Q4_K_M.gguf".to_string(),
        path: PathBuf::from("test-models/tiny-llm-Q4_K_M.gguf"),
        quantization: "Q4_K_M".to_string(),
        source_url: "https://example.com/tiny-llm.gguf".to_string(),
        file_size_bytes: 12_000_000,
        architecture: "llama".to_string(),
        context_length: 2048,
        downloaded_at: Some("2024-05-26T00:00:00Z".to_string()),
    };

    assert_eq!(entry.name, "Tiny LLM");
    assert_eq!(entry.quantization, "Q4_K_M");
    assert_eq!(entry.context_length, 2048);
}

#[test]
fn test_session_creation() {
    use llama_ui_session::{ChatMessage, Role, Session};

    let mut session = Session::new("test-model");
    session.add_message(ChatMessage {
        role: Role::User,
        content: "Hello, world!".to_string(),
        timestamp: "2024-05-26T00:00:00Z".to_string(),
        token_count: None,
    });
    session.add_message(ChatMessage {
        role: Role::Assistant,
        content: "Hi there!".to_string(),
        timestamp: "2024-05-26T00:00:01Z".to_string(),
        token_count: None,
    });

    assert_eq!(session.messages.len(), 2);
    assert_eq!(session.messages[0].role, Role::User);
    assert_eq!(session.messages[1].role, Role::Assistant);
}

#[test]
fn test_session_export_json() {
    use llama_ui_session::{ChatMessage, Role, Session};
    use std::fs;

    let mut session = Session::new("test-model");
    session.add_message(ChatMessage {
        role: Role::User,
        content: "Test message".to_string(),
        timestamp: "2024-05-26T00:00:00Z".to_string(),
        token_count: None,
    });

    let temp_path = "/tmp/test_session.json";
    let path = PathBuf::from(temp_path);

    session.export_json(&path).expect("Failed to export JSON");

    let content = fs::read_to_string(&path).expect("Failed to read exported JSON");
    assert!(content.contains("Test message"));
    assert!(content.contains("User"));

    let _ = fs::remove_file(&path);
}

#[test]
fn test_session_export_markdown() {
    use llama_ui_session::{ChatMessage, Role, Session};
    use std::fs;

    let mut session = Session::new("test-model");
    session.add_message(ChatMessage {
        role: Role::User,
        content: "What is Rust?".to_string(),
        timestamp: "2024-05-26T00:00:00Z".to_string(),
        token_count: None,
    });
    session.add_message(ChatMessage {
        role: Role::Assistant,
        content: "Rust is a systems programming language.".to_string(),
        timestamp: "2024-05-26T00:00:01Z".to_string(),
        token_count: None,
    });

    let temp_path = "/tmp/test_session.md";
    let path = PathBuf::from(temp_path);

    session
        .export_markdown(&path)
        .expect("Failed to export Markdown");

    let content = fs::read_to_string(&path).expect("Failed to read exported Markdown");
    assert!(content.contains("What is Rust?"));
    assert!(content.contains("Rust is a systems programming language."));

    let _ = fs::remove_file(&path);
}
