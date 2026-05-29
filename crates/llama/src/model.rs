//! Model loading and tensor access.
//!
//! Contains the `impl Model` blocks for loading a model from a GGUF file,
//! querying tensors, and retrieving model metadata.  The `Model` struct
//! definition itself lives in `lib.rs`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use gguf::{GgufError, GgufReader, GgufValue};
use rayon::prelude::*;

use crate::kv_cache::KvCacheManager;
use crate::tokenizer;
use crate::{
    InferenceContext, InternedStrings, Model, ModelConfig, NormType, RoPEConfig, RopeScaleType,
    TensorData,
};

impl Model {
    /// Return a concise summary of the model configuration.
    pub fn summary(&self) -> String {
        format!(
            "Model: arch={}, embd={}, heads={}, kv_heads={}, d_head={}, layers={}, seq_len={}, rope_theta={}, rope_scale={:?}, norm_eps={}",
            self.architecture,
            self.n_embd,
            self.n_head,
            self.n_head_kv,
            self.d_head,
            self.n_layers,
            self.max_seq_len,
            self.rope_theta,
            self.rope_config.scale_type,
            self.norm_eps
        )
    }

    /// Retrieve a tensor by name, returning de-quantized data.
    pub fn get_tensor(&self, name: &str) -> Result<Arc<[f32]>, GgufError> {
        let id = self
            .interned
            .strings
            .iter()
            .position(|s| s == name)
            .ok_or_else(|| GgufError::DecodeError(format!("Tensor not found: {}", name)))?;
        self.tensors
            .get(&id)
            .ok_or_else(|| GgufError::DecodeError(format!("Tensor not found: {}", name)))?
            .get()
    }

    /// Retrieve a tensor by name, returning its shape.
    pub fn get_tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        let id = self.interned.strings.iter().position(|s| s == name)?;
        self.tensors.get(&id).map(|t| t.shape.clone())
    }

    /// Return the number of transformer blocks in the model.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Load a model from a GGUF file, reading all tensors in parallel and
    /// de‑quantizing them eagerly. This is the primary entry point used by the
    /// CLI and server binaries.
    ///
    /// If `offload_ffn` is true, FFN weights (gate, up, down) will not be cached
    /// after dequantization, allowing them to be reloaded from memory-mapped file
    /// on each use to save VRAM.
    pub fn load_from_gguf<P: AsRef<Path>>(path: P, offload_ffn: bool) -> Result<Self, GgufError> {
        let reader = GgufReader::from_file(&path)?;

        let architecture = match reader.get_kv("general.architecture") {
            Some(GgufValue::Str(s)) => s.clone(),
            _ => "llama".to_string(),
        };

        let arch_prefix = format!("{architecture}.");

        let embd_keys = [
            format!("{arch_prefix}embedding_length"),
            "general.embedding_length".to_string(),
            "llama.embedding_length".to_string(),
        ];
        let n_embd =
            reader.get_usize_any(&embd_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        let head_keys = [
            format!("{arch_prefix}attention.head_count"),
            "general.attention.head_count".to_string(),
            "llama.attention.head_count".to_string(),
            "general.attention_head_count".to_string(),
        ];
        let n_head =
            reader.get_usize_any(&head_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        let head_kv_keys = [
            format!("{arch_prefix}attention.head_count_kv"),
            "general.attention.head_count_kv".to_string(),
            "llama.attention.head_count_kv".to_string(),
            "general.attention_head_count_kv".to_string(),
        ];
        let n_head_kv =
            reader.get_usize_any(&head_kv_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        let d_head_keys = [
            format!("{arch_prefix}rope.dimension_count"),
            "general.attention.head_dim".to_string(),
            "llama.rope.dimension_count".to_string(),
            "general.attention_head_dim".to_string(),
        ];
        let d_head =
            reader.get_usize_any(&d_head_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        let ctx_keys = [
            format!("{arch_prefix}context_length"),
            "general.context_length".to_string(),
            "llama.context_length".to_string(),
        ];
        let max_seq_len =
            reader.get_usize_any(&ctx_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        let vocab_keys = [
            format!("{arch_prefix}vocab_size"),
            "general.vocab_size".to_string(),
            "llama.vocab_size".to_string(),
        ];
        let vocab_size =
            reader.get_usize_any(&vocab_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        let n_ff_keys = [
            format!("{arch_prefix}feed_forward_length"),
            format!("{arch_prefix}intermediate_size"),
            "llama.feed_forward_length".to_string(),
            "general.feed_forward_length".to_string(),
            "llama.intermediate_size".to_string(),
        ];
        let n_ff = reader
            .get_usize_any(&n_ff_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .unwrap_or(n_embd * 4);

        let layer_keys = [
            format!("{arch_prefix}block_count"),
            "llama.block_count".to_string(),
            "general.block_count".to_string(),
        ];
        let n_layers =
            reader.get_usize_any(&layer_keys.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        let rope_theta_keys = [
            format!("{arch_prefix}rope.freq_base"),
            "llama.rope.freq_base".to_string(),
        ];
        let rope_theta = match reader.get_kv_any(
            &rope_theta_keys
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        ) {
            Some(GgufValue::F32(v)) => *v,
            Some(GgufValue::F64(v)) => *v as f32,
            _ => match architecture.as_str() {
                "gemma" | "gemma2" => 10000.0,
                "phi2" | "phi3" => 10000.0,
                "qwen2" => 1000000.0,
                _ => 10000.0,
            },
        };

        // ─── RoPE scaling metadata ─────────────────────────────────────
        let rope_scale_type = match reader.get_kv(&format!("{arch_prefix}rope.scaling.type")) {
            Some(GgufValue::Str(s)) => match s.as_str() {
                "linear" => RopeScaleType::Linear,
                "ntk" | "ntk-aware" | "yarn" => RopeScaleType::NtkAware,
                "dynamic" | "dynamic-ntk" => RopeScaleType::DynamicNtk,
                _ => RopeScaleType::None,
            },
            _ => RopeScaleType::None,
        };
        let rope_scale_factor = match reader.get_kv(&format!("{arch_prefix}rope.scaling.factor")) {
            Some(GgufValue::F32(v)) => *v,
            Some(GgufValue::F64(v)) => *v as f32,
            _ => 1.0,
        };
        let rope_original_max_seq = reader
            .get_usize_any(&[&format!(
                "{arch_prefix}rope.scaling.original_max_position_embeddings"
            )])
            .unwrap_or(max_seq_len);
        let rope_dim_count = reader
            .get_usize_any(&[&format!("{arch_prefix}rope.dimension_count")])
            .ok();

        let rope_config = RoPEConfig {
            theta: rope_theta,
            scale_type: rope_scale_type,
            scale_factor: rope_scale_factor,
            original_max_seq_len: rope_original_max_seq,
            partial_dim: if rope_dim_count.is_some() && rope_dim_count != Some(d_head) {
                rope_dim_count
            } else {
                None
            },
        };

        let norm_eps = match architecture.as_str() {
            "gemma" | "gemma2" => 1e-6,
            "phi2" | "phi3" => 1e-5,
            "qwen2" => 1e-6,
            "stablelm" => 1e-5,
            _ => 1e-5,
        };

        // Sliding window attention (Mistral/Mixtral)
        let sliding_window_keys = [
            format!("{arch_prefix}attention.sliding_window"),
            "mistral.attention.sliding_window".to_string(),
            "mixtral.attention.sliding_window".to_string(),
        ];
        let sliding_window = reader
            .get_usize_any(
                &sliding_window_keys
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .ok();

        // QK-norm (Gemma2): detect by checking if first layer QK-norm tensor exists
        let has_qk_norm = reader
            .tensors()
            .iter()
            .any(|t| t.name == "blk.0.attn_q_norm.weight");
        let qk_norm_eps = match architecture.as_str() {
            "gemma2" => 1e-6,
            _ => 1e-5,
        };

        let norm_type = match architecture.as_str() {
            "phi2" => NormType::LayerNorm,
            _ => NormType::RmsNorm,
        };

        let interned = Arc::new(std::sync::Mutex::new(InternedStrings::default()));
        let shared_mmap = reader.mmap_arc().clone();
        let tensors: HashMap<usize, TensorData> = reader
            .tensors()
            .par_iter()
            .map(|info| {
                let mmap_tensor = reader.mmap_tensor(info, shared_mmap.clone())?;
                let shape = info.shape.iter().map(|&d| d as usize).collect();
                 let mut guard = interned.lock().expect("RwLock poisoned - this indicates a bug in the code");
                let id = guard.intern(&info.name);
                drop(guard);
                // Determine if we should cache this tensor.
                // If offload_ffn is true, do not cache FFN weights (gate, up, down).
                let is_ffn_weight = info.name.ends_with(".ffn_gate.weight")
                    || info.name.ends_with(".ffn_up.weight")
                    || info.name.ends_with(".ffn_down.weight");
                let cache = if offload_ffn { !is_ffn_weight } else { true };
                Ok((
                    id,
                    TensorData {
                        mmap_tensor,
                        info: info.clone(),
                        data: RwLock::new(None),
                        shape,
                        cache,
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>, GgufError>>()?;
        let interned = Arc::try_unwrap(interned)
            .expect("no other references to interner")
            .into_inner()
            .expect("mutex not poisoned after try_unwrap");

        let kv_cache = RwLock::new(KvCacheManager::new(
            n_layers,
            max_seq_len,
            n_head_kv,
            d_head,
        ));

        let (vocab_tokens, vocab_types_raw) = (
            reader
                .get_string_array("tokenizer.ggml.tokens")
                .unwrap_or_else(|_| (0..vocab_size).map(|i| format!("<token{}>", i)).collect()),
            reader
                .get_i32_array("tokenizer.ggml.token_type")
                .unwrap_or_else(|_| vec![1; vocab_size]),
        );
        let vocab_types: Vec<tokenizer::TokenType> = vocab_types_raw
            .iter()
            .map(|&v| tokenizer::TokenType::from_i32(v))
            .collect();
        let bos_token_id = reader
            .get_usize_any(&["tokenizer.ggml.bos_token_id"])
            .unwrap_or(1);
        let eos_token_id = reader
            .get_usize_any(&["tokenizer.ggml.eos_token_id"])
            .unwrap_or(2);
        let unk_token_id = reader
            .get_usize_any(&["tokenizer.ggml.unknown_token_id"])
            .unwrap_or(0);
        let add_bos_token = match reader.get_kv("tokenizer.ggml.add_bos_token") {
            Some(GgufValue::Bool(b)) => *b,
            _ => true,
        };

        Ok(Self {
            tensors,
            interned,
            architecture,
            n_embd,
            n_head,
            n_head_kv,
            d_head,
            max_seq_len,
            vocab_size,
            n_ff,
            n_layers,
            rope_theta,
            rope_config,
            has_qk_norm,
            qk_norm_eps,
            norm_eps,
            vocab_tokens,
            vocab_types,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            add_bos_token,
            kv_cache,
            sliding_window,
            norm_type,
        })
    }

    /// Run a batch of prompts, returning a vector of generated token ID sequences.
    /// This is a simple placeholder implementation that creates an inference context for each prompt.
    /// Future versions may implement true parallel batch inference.
    pub fn run_batch(self, prompts: &[&str]) -> Vec<Vec<usize>> {
        let model_arc = Arc::new(self);
        let mut result = Vec::new();
        for prompt in prompts {
            let mut context = InferenceContext::new(model_arc.clone(), ModelConfig::default());
            match context.generate(prompt, 0) {
                Ok(tokens) => result.push(tokens),
                Err(_) => result.push(vec![]),
            }
        }
        result
    }
}
