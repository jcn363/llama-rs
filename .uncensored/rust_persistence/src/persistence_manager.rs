use crate::{
    agent_state::{Context, Files, Learning, Memory, SessionState},
    errors::PersistenceError,
};
use serde_json;
use std::collections::HashMap;
use std::fs;
// Unused imports suppressed for now.
use std::path::PathBuf;
use uuid::Uuid;

/// Manages persistence operations for uncensored agent sessions
pub struct PersistenceManager {
    /// Base directory for storing session data
    base_path: PathBuf,
    /// Cache of loaded session states
    cache: HashMap<String, SessionState>,
}

impl PersistenceManager {
    /// Creates a new persistence manager with the specified base directory
    pub fn new(base_path: PathBuf) -> Result<Self, PersistenceError> {
        let manager = PersistenceManager {
            base_path,
            cache: HashMap::new(),
        };
        // Ensure the base directory exists
        if !manager.base_path.exists() {
            fs::create_dir_all(&manager.base_path).map_err(PersistenceError::Io)?;
        }
        Ok(manager)
    }

    /// Saves a session with placeholder data (real agent will provide actual state)
    pub fn save(&mut self, name: &str) -> Result<(), PersistenceError> {
        if !validate_session_name(name) {
            return Err(PersistenceError::InvalidSessionName(name.to_string()));
        }
        let session_dir = self.base_path.join(name);
        fs::create_dir_all(&session_dir).map_err(PersistenceError::Io)?;

        // Placeholder state – replace with real data when integrating
        let state = SessionState {
            session_id: Uuid::new_v4().to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            context: Context {
                goal: "Placeholder goal".to_string(),
                completed: vec![],
                pending: vec![],
                failed: vec![],
                notes: "Placeholder notes".to_string(),
            },
            memory: Memory {
                key_facts: vec![],
                decisions: vec![],
                learnings: vec![],
            },
            files: Files {
                created: vec![],
                modified: vec![],
            },
            failed_tasks: vec![],
            decisions: vec![],
            learning: Learning::default(),
        };
        let json = serde_json::to_string_pretty(&state).map_err(PersistenceError::Json)?;
        let file_path = session_dir.join("state.json");
        fs::write(&file_path, json).map_err(PersistenceError::Io)?;
        self.cache.insert(name.to_string(), state);
        Ok(())
    }

    /// Loads a session state by name
    pub fn load(&mut self, name: &str) -> Result<SessionState, PersistenceError> {
        if !validate_session_name(name) {
            return Err(PersistenceError::InvalidSessionName(name.to_string()));
        }
        if let Some(cached) = self.cache.get(name) {
            return Ok(cached.clone());
        }
        let file_path = self.base_path.join(name).join("state.json");
        if !file_path.exists() {
            return Err(PersistenceError::SessionNotFound(name.to_string()));
        }
        let json = fs::read_to_string(&file_path).map_err(PersistenceError::Io)?;
        let state: SessionState = serde_json::from_str(&json).map_err(PersistenceError::Json)?;
        self.cache.insert(name.to_string(), state.clone());
        Ok(state)
    }

    /// Lists all available session names
    pub fn list(&self) -> Result<Vec<String>, PersistenceError> {
        let mut sessions = Vec::new();
        if !self.base_path.exists() {
            return Ok(sessions);
        }
        for entry in fs::read_dir(&self.base_path).map_err(PersistenceError::Io)? {
            let entry = entry.map_err(PersistenceError::Io)?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    sessions.push(name.to_string());
                }
            }
        }
        Ok(sessions)
    }

    /// Validates a session name for safety
    fn validate_session_name(name: &str) -> bool {
        !name.is_empty()
            && !name.starts_with('.')
            && !name.ends_with('.')
            && name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
    }
}

// Helper function for external callers
fn validate_session_name(name: &str) -> bool {
    PersistenceManager::validate_session_name(name)
}
