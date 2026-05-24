// src/lib.rs
//! Type-safe persistence system for uncensored agents
//!
//! This library provides a robust, type-safe persistence layer for the uncensored agent
//! with proper error handling, validation, and type safety.

pub mod agent_state;
pub mod errors;
pub mod persistence_manager;

pub use agent_state::{Context, Decision, FailedTask, Files, Learning, Memory, SessionState};
pub use errors::PersistenceError;
pub use persistence_manager::PersistenceManager;

#[cfg(test)]
mod tests;

/// Validates a session name for safety
pub fn validate_session_name(name: &str) -> bool {
    // Allow alphanumeric, underscore, hyphen, and dot
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        && !name.starts_with('.')
        && !name.ends_with('.')
}
