//! Chat session persistence for llama-ui.
//!
//! Provides `ChatMessage`, `Session`, and export/import in JSON, Markdown, and plain text.

#![deny(missing_docs)]

use std::path::PathBuf;

use common::sampling::SamplingConfig;
use serde::{Deserialize, Serialize};

/// Role of a chat message participant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    /// Human user message.
    User,
    /// Model assistant response.
    Assistant,
    /// System instruction message.
    System,
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message sender role.
    pub role: Role,
    /// Message text content.
    pub content: String,
    /// ISO-8601 or epoch timestamp.
    pub timestamp: String,
    /// Optional token count for this message.
    pub token_count: Option<usize>,
}

/// Full chat session with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Ordered list of chat messages.
    pub messages: Vec<ChatMessage>,
    /// Model identifier used for this session.
    pub model_id: String,
    /// Snapshot of sampling parameters.
    pub sampler_config: SamplingConfig,
    /// Optional chat template name.
    pub template_name: Option<String>,
    /// ISO-8601 or epoch creation timestamp.
    pub created_at: String,
    /// ISO-8601 or epoch last-updated timestamp.
    pub updated_at: String,
}

impl Session {
    /// Create a new empty session.
    pub fn new(model_id: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string();
        Self {
            messages: Vec::new(),
            model_id: model_id.into(),
            sampler_config: SamplingConfig::default(),
            template_name: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Add a message to the session.
    pub fn add_message(&mut self, msg: ChatMessage) {
        use std::time::{SystemTime, UNIX_EPOCH};
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string();
        self.messages.push(msg);
    }

    /// Export session to JSON file.
    pub fn export_json(&self, path: &PathBuf) -> Result<(), ExportError> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Import session from JSON file.
    pub fn import_json(path: &PathBuf) -> Result<Self, ExportError> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Export session as human-readable Markdown.
    pub fn export_markdown(&self, path: &PathBuf) -> Result<(), ExportError> {
        let mut md = String::new();
        md.push_str(&format!(
            "# Chat Session\n\n**Model:** {}\n\n",
            self.model_id
        ));
        for msg in &self.messages {
            let role = match msg.role {
                Role::User => "## User",
                Role::Assistant => "## Assistant",
                Role::System => "## System",
            };
            md.push_str(&format!(
                "\n{} ({})\n\n{}\n",
                role, msg.timestamp, msg.content
            ));
        }
        std::fs::write(path, md)?;
        Ok(())
    }

    /// Export session as plain text transcript.
    pub fn export_plain(&self, path: &PathBuf) -> Result<(), ExportError> {
        let mut text = String::new();
        for msg in &self.messages {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            text.push_str(&format!("{}: {}\n\n", role, msg.content));
        }
        std::fs::write(path, text)?;
        Ok(())
    }
}

/// Errors that can occur during session export/import.
pub type ExportError = llama_ui_core::error::Error;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new_and_add() {
        let mut sess = Session::new("test-model");
        assert!(sess.messages.is_empty());
        sess.add_message(ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            timestamp: "now".into(),
            token_count: None,
        });
        assert_eq!(sess.messages.len(), 1);
    }

    #[test]
    fn test_session_json_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        let mut sess = Session::new("test-model");
        sess.add_message(ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            timestamp: "t1".into(),
            token_count: None,
        });
        sess.export_json(&path.to_path_buf()).unwrap();
        let loaded = Session::import_json(&path.to_path_buf()).unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].content, "Hello");
        assert_eq!(loaded.model_id, "test-model");
    }

    #[test]
    fn test_export_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chat.md");
        let mut sess = Session::new("test");
        sess.add_message(ChatMessage {
            role: Role::User,
            content: "Hi".into(),
            timestamp: "t1".into(),
            token_count: None,
        });
        sess.export_markdown(&path.to_path_buf()).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("## User"));
        assert!(content.contains("Hi"));
    }

    #[test]
    fn test_export_plain_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chat.txt");
        let mut sess = Session::new("test-model");
        sess.add_message(ChatMessage {
            role: Role::Assistant,
            content: "Reply".into(),
            timestamp: "t2".into(),
            token_count: Some(5),
        });
        sess.export_plain(&path.to_path_buf()).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Assistant: Reply"));
    }
}
