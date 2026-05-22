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
use crate::{InternedStrings, Model, TensorData};

impl Model {
    /// Return a short summary string for debugging.
    pub fn summary(&self) -> String {
        format!(
            "Model: arch={}, embd={}, heads={}, kv_heads={}, d_head={}, layers={}, seq_len={}, rope_theta={}, norm_eps={}",
            self.architecture,
            self.n_embd,
            self.n_head,
            self.n_head_kv,
            self.d_head,
            self.n_layers,
            self.max_seq_len,
            self.rope_theta,
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
    pub fn load_from_gguf<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
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

        let norm_eps = match architecture.as_str() {
            "gemma" | "gemma2" => 1e-6,
            "phi2" | "phi3" => 1e-5,
            "qwen2" => 1e-6,
            "stablelm" => 1e-5,
            _ => 1e-5,
        };

        let interned = Arc::new(std::sync::Mutex::new(InternedStrings::default()));
        let shared_mmap = reader.mmap_arc().clone();
        let tensors: HashMap<usize, TensorData> = reader
            .tensors()
            .par_iter()
            .map(|info| {
                let mmap_tensor = reader.mmap_tensor(info, shared_mmap.clone())?;
                let shape = info.shape.iter().map(|&d| d as usize).collect();
                let mut guard = interned.lock().unwrap();
                let id = guard.intern(&info.name);
                drop(guard);
                Ok((
                    id,
                    TensorData {
                        mmap_tensor,
                        info: info.clone(),
                        data: RwLock::new(None),
                        shape,
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>, GgufError>>()?;
        let interned = Arc::try_unwrap(interned)
            .expect("no other references to interner")
            .into_inner()
            .unwrap();

        let kv_cache = RwLock::new(KvCacheManager::new(
            n_layers,
            max_seq_len,
            n_head_kv,
            d_head,
        ));

        let (vocab_tokens, vocab_scores, vocab_types_raw) = (
            reader
                .get_string_array("tokenizer.ggml.tokens")
                .unwrap_or_else(|_| (0..vocab_size).map(|i| format!("<token{}>", i)).collect()),
            reader
                .get_f32_array("tokenizer.ggml.scores")
                .unwrap_or_else(|_| vec![0.0; vocab_size]),
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
            norm_eps,
            vocab_tokens,
            vocab_scores,
            vocab_types,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            add_bos_token,
            kv_cache,
        })
    }

    /// Backwards‑compatible wrapper used by existing code.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        Self::load_from_gguf(path)
    }
}
