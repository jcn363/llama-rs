//! Shared sampling configuration for text generation.
//!
//! This is the single canonical `SamplingConfig` used across the workspace
//! by `llama-cli`, `llama-server`, and `llama-ui`.

use serde::{Deserialize, Serialize};

/// Configuration for text generation sampling.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Temperature for sampling (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Top-p nucleus sampling (1.0 = disabled).
    pub top_p: f32,
    /// Repeat penalty (>1.0 penalizes repeated tokens).
    pub repeat_penalty: f32,
    /// Optional random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: None,
        }
    }
}
