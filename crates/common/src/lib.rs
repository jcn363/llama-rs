//! Common utilities for llama inference.
//!
//! This crate provides shared functionality: argument parsing, chat templates,
//! sampling strategies, unicode handling, and Jinja template rendering.

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

/// Command-line argument parsing.
pub mod args {
    use clap::Args;

    /// Common arguments for llama inference tools.
    #[derive(Args, Debug, Clone)]
    pub struct CommonArgs {
        /// Path to the model file (GGUF format).
        #[arg(short, long)]
        pub model: String,

        /// Number of threads to use for computation.
        #[arg(short = 't', long, default_value_t = 0)]
        pub threads: usize,

        /// Context size (number of tokens).
        #[arg(short = 'c', long, default_value_t = 512)]
        pub ctx_size: usize,

        /// Batch size for prompt processing.
        #[arg(long, default_value_t = 512)]
        pub batch_size: usize,

        /// Minimum matrix rows for parallel matmul (default: 128).
        #[arg(long, default_value_t = 128)]
        pub parallel_min_rows: usize,

        /// KV cache strategy: "full" or "prefix".
        #[arg(long, default_value_t = String::from("full"))]
        pub cache_strategy: String,
    }
}

/// Sampling strategies for text generation.
pub mod sampling {
    /// Configuration for token sampling.
    ///
    /// This is the **canonical** `SamplingConfig` for the entire workspace.
    /// The `llama` crate re-exports this type — do NOT define a second copy.
    #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
    pub struct SamplingConfig {
        /// Temperature for sampling (0.0 = greedy argmax).
        pub temperature: f32,
        /// Top-k sampling (0 = disabled, full vocab).
        pub top_k: usize,
        /// Top-p nucleus sampling (0.0 = disabled, 1.0 = disabled).
        pub top_p: f32,
        /// Repeat penalty (1.0 = no penalty).
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
}

/// Chat template rendering using Jinja (minijinja).
pub mod chat_templates;

#[cfg(test)]
mod tests {
    use super::sampling::SamplingConfig;

    #[test]
    fn sampling_config_should_default_reasonable_values() {
        let config = SamplingConfig::default();
        assert!(config.temperature > 0.0);
        assert!(config.top_k > 0);
        assert!(config.top_p > 0.0 && config.top_p <= 1.0);
        assert!(config.repeat_penalty >= 1.0);
        assert!(config.seed.is_none());
    }
}
