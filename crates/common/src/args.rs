//! Shared CLI argument parsing using clap.
//!
//! Provides `CommonArgs` for model path, thread count, context size,
//! sampling parameters, etc. Used by `llama-cli` and `llama-server`
//! via `#[clap(flatten)]`.

use clap::Args;

/// Common CLI arguments shared by `llama-cli` and `llama-server`.
#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Path to the GGUF model file.
    #[arg(short, long)]
    pub model: String,

    /// Number of threads to use for CPU operations (0 = auto-detect).
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// KV cache strategy: full, prefix, prefix_only.
    #[arg(long, default_value_t = String::from("full"))]
    pub cache_strategy: String,

    /// Context size (number of tokens).
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: usize,

    /// Batch size for prompt processing.
    #[arg(long, default_value_t = 512)]
    pub batch_size: usize,

    // ── Sampling parameters (defaults match common::sampling::SamplingConfig) ──

    /// Sampling temperature (0.0 = greedy, default: 0.8).
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,

    /// Top-k sampling (0 = disabled, default: 40).
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Top-p nucleus sampling (1.0 = disabled, default: 0.95).
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Repeat penalty (>1.0 penalises repeated tokens, default: 1.1).
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// Random seed for reproducibility (default: random).
    #[arg(long)]
    pub seed: Option<u64>,

    // ── Backend & model-loading parameters ──

    /// Backend to use: auto, cpu, cuda (default: auto).
    #[arg(long, default_value_t = String::from("auto"))]
    pub backend: String,

    /// Offload FFN weights to RAM (load on demand) to save VRAM.
    #[arg(long, default_value_t = false)]
    pub offload_ffn: bool,

    /// Size of thread-local memory pool for small temporary allocations (in bytes, 0 = disabled).
    #[arg(long, default_value_t = 0)]
    pub memory_pool_size: usize,
}
