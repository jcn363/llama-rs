#![deny(missing_docs)]

//! LLaMA inference engine.
//!
//! This crate provides model loading, tokenization, and inference
//! for LLaMA-family large language models.
//!
//! # Quick Start
//!
//! ```no_run
//! use std::sync::Arc;
//! use llama::{Model, ModelConfig, InferenceContext};
//!
//! let model = Arc::new(Model::load_from_gguf("model.gguf", false).unwrap());
//! let mut ctx = InferenceContext::new(model, ModelConfig::default());
//! let tokens = ctx.generate("Hello, world!", 128).unwrap();
//! println!("{}", ctx.decode(&tokens));
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ─── Module declarations ─────────────────────────────────────────────────────

mod attention;
pub mod backend;
mod context;
mod inference;
mod kv_cache;
mod model;
mod profile;
pub mod tokenizer;

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use attention::{apply_rope_with_config, multi_head_attention_with_cache};
pub use backend::{BackendType, create_backend};
pub use context::{InferenceContext, ModelConfig};
pub use ggml::backend::{Backend, BackendInfo, QuantType};
pub use gguf::GgmlType;
pub use inference::{SamplingConfig, dot_product};
pub use kv_cache::{CacheStrategy, KvCache, KvCacheManager};
pub use profile::ProfileResult;
pub use tokenizer::SimpleTokenizer;

// ─── Architecture Support Types ───────────────────────────────────────────────

/// Normalization type used by different model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// RMSNorm (default for Llama, Mistral, Gemma, Qwen2, etc.)
    RmsNorm,
    /// LayerNorm (used by Phi-2)
    LayerNorm,
}

/// RoPE scaling strategy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScaleType {
    /// No scaling (vanilla RoPE).
    None,
    /// Linear scaling: divide position by `scale_factor`.
    Linear,
    /// NTK-aware scaling: adjust base frequency.
    NtkAware,
    /// Dynamic NTK scaling: adjusts per-step.
    DynamicNtk,
}

/// Configuration for RoPE with optional scaling.
#[derive(Debug, Clone, Copy)]
pub struct RoPEConfig {
    /// RoPE base frequency (theta).
    pub theta: f32,
    /// Scaling type (None = vanilla RoPE).
    pub scale_type: RopeScaleType,
    /// Scale factor for linear/NTK scaling.
    pub scale_factor: f32,
    /// Original max sequence length used during training.
    pub original_max_seq_len: usize,
    /// Optional partial rotation dimension (Phi-3: partial for some layers).
    pub partial_dim: Option<usize>,
}

impl RoPEConfig {
    /// Create a default RoPE config (vanilla, no scaling).
    pub fn new(theta: f32) -> Self {
        Self {
            theta,
            scale_type: RopeScaleType::None,
            scale_factor: 1.0,
            original_max_seq_len: 4096,
            partial_dim: None,
        }
    }
}

// ─── TensorData ──────────────────────────────────────────────────────────────

/// Lazy-loaded tensor data backed by memory-mapped file access.
///
// Raw (quantized) data stays on disk until [`get`](TensorData::get) is called,
// at which point it is dequantized and optionally cached.
#[derive(Debug)]
pub struct TensorData {
    /// Memory-mapped reference to the tensor's raw (quantized) data.
    pub mmap_tensor: gguf::MmapTensor,
    /// Tensor metadata needed for de‑quantization.
    pub info: gguf::TensorInfo,
    /// De‑quantized float values – filled on first access.
    pub data: RwLock<Option<Arc<[f32]>>>,
    /// Shape of the tensor (rows, cols) for 2‑D tensors; empty for scalars.
    pub shape: Vec<usize>,
    /// Whether to cache the dequantized data after first use.
    pub cache: bool,
}

impl TensorData {
    /// Return the de‑quantized data, performing lazy de‑quantization if needed.
    pub fn get(&self) -> Result<Arc<[f32]>, gguf::GgufError> {
        if let Some(ref d) = *self.data.read().expect("lock poisoned") {
            return Ok(d.clone());
        }
        let deq = self.mmap_tensor.dequantize(&self.info)?;
        let arc: Arc<[f32]> = Arc::from(deq.into_boxed_slice());
        if self.cache {
            let mut write = self.data.write().expect("lock poisoned");
            *write = Some(arc.clone());
        }
        Ok(arc)
    }

    /// Return the raw quantized tensor bytes and its GGML type.
    ///
    /// This avoids dequantization entirely — the caller can pass the bytes
    /// directly to [`Backend::mat_vec_quant`] for faster inference.
    pub fn get_quantized_raw(&self) -> Result<(&[u8], gguf::GgmlType), gguf::GgufError> {
        let raw = self.mmap_tensor.as_slice()?;
        Ok((raw, self.info.dtype))
    }
}

// ─── InternedStrings ─────────────────────────────────────────────────────────

/// Simple interner for strings used throughout the model (e.g., tensor names).
#[derive(Debug, Default)]
pub struct InternedStrings {
    /// Vector of unique strings; index is the interned ID.
    pub strings: Vec<String>,
    /// Reverse map for fast lookup.
    pub map: HashMap<String, usize>,
}

impl InternedStrings {
    /// Intern a string, returning its unique ID.
    pub fn intern(&mut self, s: &str) -> usize {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.strings.len();
        self.strings.push(s.to_owned());
        self.map.insert(s.to_owned(), id);
        id
    }

    /// Retrieve a string by its ID.
    pub fn get(&self, id: usize) -> Option<&str> {
        self.strings.get(id).map(|s| s.as_str())
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// The core model struct holding all hyper‑parameters, tensors, and cache.
#[derive(Debug)]
pub struct Model {
    /// Mapping from interned tensor ID to its data.
    pub tensors: HashMap<usize, TensorData>,
    /// Interner for tensor names and other strings.
    pub interned: InternedStrings,
    /// Model architecture detected from GGUF metadata.
    pub architecture: String,
    /// Model hyper‑parameters extracted from GGUF metadata.
    pub n_embd: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Number of key/value heads (GQA).
    pub n_head_kv: usize,
    /// Dimension per attention head.
    pub d_head: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Feed-forward hidden dimension.
    pub n_ff: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RoPE configuration with optional scaling.
    pub rope_config: RoPEConfig,
    /// Whether this model has QK-norm (Gemma2).
    pub has_qk_norm: bool,
    /// RMSNorm epsilon for QK-norm (Gemma2 uses 1e-6).
    pub qk_norm_eps: f32,
    /// RMSNorm epsilon (architecture-dependent).
    pub norm_eps: f32,
    /// Tokenizer vocabulary loaded from GGUF metadata.
    pub vocab_tokens: Vec<String>,
    /// Tokenizer scores (for BPE ranking).
    pub vocab_scores: Vec<f32>,
    /// Tokenizer token types.
    pub vocab_types: Vec<tokenizer::TokenType>,
    /// BOS token ID.
    pub bos_token_id: usize,
    /// EOS token ID.
    pub eos_token_id: usize,
    /// Unknown token ID.
    pub unk_token_id: usize,
    /// Whether to add BOS token automatically.
    pub add_bos_token: bool,
    /// KV cache used during inference (one per layer).
    pub kv_cache: RwLock<kv_cache::KvCacheManager>,
    /// Sliding window size for Mistral/Mixtral (None = full attention).
    pub sliding_window: Option<usize>,
    /// Normalization type used by this architecture.
    pub norm_type: NormType,
}
