use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SessionState {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub context: Context,
    pub memory: Memory,
    pub files: Files,
    pub failed_tasks: Vec<FailedTask>,
    pub decisions: Vec<Decision>,
    pub learning: Learning,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Context {
    pub goal: String,
    pub completed: Vec<String>,
    pub pending: Vec<String>,
    pub failed: Vec<FailedTask>,
    pub notes: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct FailedTask {
    pub task: String,
    pub error: String,
    pub attempts: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Files {
    pub created: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Memory {
    pub key_facts: Vec<String>,
    pub decisions: Vec<Decision>,
    pub learnings: Vec<Learning>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Decision {
    pub topic: String,
    pub decision: String,
    pub rationale: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Learning {
    pub what: String,
    pub source: String,
}
