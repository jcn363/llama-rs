//! Inference context and configuration.
//!
//! Contains [`ModelConfig`] for inference parameters and [`InferenceContext`]
//! which ties together a model, tokenizer, and sampling configuration for
//! text generation.

use bumpalo::Bump;
use std::sync::Arc;
use std::time::Instant;

use crate::attention::multi_head_attention_with_cache;
use crate::backend::BackendType;
use crate::inference::{SamplingConfig, embed_token, layer_norm, relu_squared, sample_logits};
use crate::kv_cache::CacheStrategy;
use crate::profile::ProfileResult;
use crate::{Model, NormType};
use ggml::backend::{Backend, QuantType};
use gguf::GgmlType;

/// Map a [`gguf::GgmlType`] to [`ggml::backend::QuantType`] if we have a kernel for it.
fn quant_type_from_ggml(t: GgmlType) -> Option<QuantType> {
    match t {
        GgmlType::Q4_0 => Some(QuantType::Q4_0),
        GgmlType::Q4_1 => Some(QuantType::Q4_1),
        GgmlType::Q8_0 => Some(QuantType::Q8_0),
        _ => None,
    }
}

/// Configuration for inference.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of inference threads.
    pub n_threads: usize,
    /// Backend type selection (Auto, Cpu, Cuda).
    pub backend_type: BackendType,
    /// Context window size.
    pub n_ctx: usize,
    /// Batch size for prompt processing (independently configurable from n_ctx).
    pub n_batch: usize,
    /// Minimum matrix rows for parallel dispatch (0 = auto).
    pub parallel_min_rows: usize,
    /// KV cache strategy.
    pub cache_strategy: CacheStrategy,
    /// Whether to offload FFN weights to RAM (load on demand) to save VRAM.
    pub offload_ffn: bool,
    /// Size of thread-local memory pool for small temporary allocations (in bytes, 0 = disabled).
    pub memory_pool_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_threads: 4,
            backend_type: BackendType::Auto,
            n_ctx: 2048,
            n_batch: 512,
            parallel_min_rows: 128,
            cache_strategy: CacheStrategy::Full,
            offload_ffn: false,
            memory_pool_size: 0,
        }
    }
}

/// Inference context holding state for a model.
pub struct InferenceContext {
    /// Shared reference to the loaded model.
    pub model: Arc<Model>,
    /// Inference configuration.
    pub config: ModelConfig,
    /// Tokenizer instance.
    pub tokenizer: crate::SimpleTokenizer,
    /// Sampling configuration.
    pub sampling: SamplingConfig,
    /// Thread‑local bump allocator for temporary buffers (reused across forward passes).
    pub bump: bumpalo::Bump,
    /// Hardware backend for tensor operations (CPU, CUDA, etc.).
    pub backend: Arc<dyn Backend>,
    /// Tokens currently in the KV cache (for prefix caching).
    pub cached_tokens: Vec<usize>,
}

impl InferenceContext {
    /// Create a new inference context.
    pub fn new(model: Arc<Model>, config: ModelConfig) -> Self {
        let tokenizer = crate::SimpleTokenizer::from_gguf_vocab(
            model.vocab_tokens.clone(),
            model.vocab_scores.clone(),
            model.vocab_types.clone(),
            model.bos_token_id,
            model.eos_token_id,
            model.unk_token_id,
            model.add_bos_token,
        );
        let backend = crate::backend::create_backend(&config);

        Self {
            model,
            config,
            tokenizer,
            sampling: SamplingConfig::default(),
            backend,
            bump: Bump::new(),
            cached_tokens: Vec::new(),
        }
    }

    /// Set the sampling configuration.
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    /// Encode input text to token ids using the tokenizer.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenizer.encode(text)
    }

    /// Generate token IDs starting from pre-encoded tokens.
    ///
    /// Prefills the KV cache for the given tokens, then generates `n_predict`
    /// new tokens one at a time. This is the O(n²)-free streaming variant —
    /// encode the prompt once, call this to generate.
    pub fn generate_from_tokens(
        &mut self,
        tokens: &[usize],
        n_predict: usize,
    ) -> anyhow::Result<Vec<usize>> {
        let mut toks = tokens.to_vec();

        // Phase 1: Prepare KV cache
        {
            let mut kv_cache = self.model.kv_cache.write().expect("lock poisoned");
            match self.config.cache_strategy {
                CacheStrategy::Prefix => {
                    let common_prefix_len = toks
                        .iter()
                        .zip(self.cached_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    kv_cache.truncate_all(common_prefix_len);
                }
                _ => {
                    kv_cache.reset();
                }
            }
        }

        // Phase 2: Prefill
        match self.config.cache_strategy {
            CacheStrategy::Prefix => {
                let common_prefix_len = toks
                    .iter()
                    .zip(self.cached_tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                if toks.len() > common_prefix_len {
                    let remaining = &toks[common_prefix_len..];
                    for chunk in remaining.chunks(self.config.n_batch) {
                        self.prefill(chunk)?;
                    }
                }
                self.cached_tokens = toks.clone();
            }
            _ => {
                self.cached_tokens.clear();
                if !toks.is_empty() {
                    for chunk in toks.chunks(self.config.n_batch) {
                        self.prefill(chunk)?;
                    }
                }
            }
        }

        // Phase 3: Generate new tokens
        let input_len = toks.len();
        for _ in 0..n_predict {
            let last_token = *toks.last().unwrap_or(&0);

            match self.forward_pass(last_token) {
                Ok(logits) => {
                    let next_token = sample_logits(&logits, &self.sampling, &toks);
                    toks.push(next_token);

                    if next_token == self.model.eos_token_id {
                        break;
                    }
                }
                Err(_) => {
                    toks.push(0);
                }
            }
        }

        // Return only newly generated tokens
        Ok(toks[input_len..].to_vec())
    }

    /// Generate token IDs for a prompt using actual inference.
    pub fn generate(&mut self, prompt: &str, n_predict: usize) -> anyhow::Result<Vec<usize>> {
        let mut toks = self.encode(prompt);

        if toks.len() > self.config.n_ctx {
            toks.truncate(self.config.n_ctx);
        }

        // Phase 1: Prepare KV cache according to strategy.
        // IMPORTANT: drop the lock before Phase 2 (prefill), because
        // forward_pass() also acquires the KV cache write lock.
        {
            let mut kv_cache = self.model.kv_cache.write().expect("lock poisoned");
            match self.config.cache_strategy {
                CacheStrategy::Prefix => {
                    // Find longest common prefix between new prompt and cached tokens
                    let common_prefix_len = toks
                        .iter()
                        .zip(self.cached_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    kv_cache.truncate_all(common_prefix_len);
                }
                _ => {
                    kv_cache.reset();
                }
            }
        }

        // Phase 2: Run prefill for the required tokens.
        // The KV cache lock has been released — forward_pass can acquire it.
        match self.config.cache_strategy {
            CacheStrategy::Prefix => {
                let common_prefix_len = toks
                    .iter()
                    .zip(self.cached_tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                if toks.len() > common_prefix_len {
                    let remaining_tokens = &toks[common_prefix_len..];
                    for chunk in remaining_tokens.chunks(self.config.n_batch) {
                        self.prefill(chunk)?;
                    }
                }
                self.cached_tokens = toks.clone();
            }
            _ => {
                self.cached_tokens.clear();
                if !toks.is_empty() {
                    for chunk in toks.chunks(self.config.n_batch) {
                        self.prefill(chunk)?;
                    }
                }
            }
        }

        // DECODE PHASE: Generate one token at a time
        for _i in 0..n_predict {
            let last_token = *toks.last().unwrap_or(&0);

            match self.forward_pass(last_token) {
                Ok(logits) => {
                    let next_token = sample_logits(&logits, &self.sampling, &toks);
                    toks.push(next_token);

                    if next_token == self.model.eos_token_id {
                        break;
                    }
                }
                Err(_) => {
                    toks.push(0);
                }
            }
        }

        Ok(toks)
    }

    // (the generate function continues below)

    /// Load a weight tensor and compute `weight @ input` in the most efficient
    /// way supported by the backend.
    ///
    /// If the tensor is in a quantized format for which a direct dot-product
    /// kernel exists (Q4_0, Q8_0, Q4_1), the raw quantized bytes are passed
    /// to [`Backend::mat_vec_quant`], avoiding dequantization entirely.
    /// Otherwise the tensor is dequantized to f32 and [`Backend::mat_vec`] is used.
    fn mat_vec_weight(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, gguf::GgufError> {
        let id = self
            .model
            .interned
            .strings
            .iter()
            .position(|s| s == name)
            .ok_or_else(|| gguf::GgufError::DecodeError(format!("Tensor not found: {name}")))?;
        let td = self
            .model
            .tensors
            .get(&id)
            .ok_or_else(|| gguf::GgufError::DecodeError(format!("Tensor not found: {name}")))?;
        Ok(match quant_type_from_ggml(td.info.dtype) {
            Some(qt) => {
                let (raw, _ty) = td.get_quantized_raw()?;
                self.backend.mat_vec_quant(raw, qt, rows, cols, input)
            }
            None => {
                let f32_data = td.get()?;
                self.backend.mat_vec(&f32_data, rows, cols, input)
            }
        })
    }

    /// Run a single forward pass through the model for a given token.
    /// Returns logits of shape (vocab_size,).
    fn forward_pass(&self, token_id: usize) -> anyhow::Result<Vec<f32>> {
        self.forward_pass_with_profile(token_id)
            .map(|(logits, _profile)| logits)
    }

    /// Run forward pass for a batch of tokens (prefill phase).
    /// Processes tokens in batches of size n_batch for better memory locality,
    /// stores KV cache entries, and returns final logits.
    pub fn prefill(&self, tokens: &[usize]) -> anyhow::Result<Vec<f32>> {
        // Process each token in sequence, storing KV cache entries
        // Only return the logits of the final token
        let mut final_logits = None;

        // Process in chunks of n_batch for better memory locality
        for chunk in tokens.chunks(self.config.n_batch) {
            for &token in chunk {
                final_logits = Some(self.forward_pass(token)?);
            }
        }

        final_logits.ok_or_else(|| anyhow::anyhow!("empty batch"))
    }

    /// Run a single forward pass with per-layer profiling.
    /// Returns both logits and timing information.
    fn forward_pass_with_profile(
        &self,
        token_id: usize,
    ) -> anyhow::Result<(Vec<f32>, ProfileResult)> {
        // Reset the thread-local bump allocator at the start of each forward pass
        // to prevent memory accumulation across invocations.
        ggml_cpu::reset_bump_allocator();
        let total_start = Instant::now();
        let mut profile = ProfileResult::default();

        let embed_start = Instant::now();
        let token_embd = self.model.get_tensor("token_embd.weight")?;
        let mut x = embed_token(token_id, &token_embd, self.model.n_embd)?;
        profile.embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

        let n_layers = self.model.n_layers();
        if n_layers == 0 {
            profile.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            return Ok((vec![0.0; self.model.vocab_size], profile));
        }

        let n_head = self.model.n_head;
        let n_head_kv = self.model.n_head_kv;
        let head_dim = self.model.d_head;
        let n_embd = self.model.n_embd;
        let rope_config = &self.model.rope_config;
        let arch = self.model.architecture.as_str();
        let norm_type = self.model.norm_type;

        // Helper: apply the correct norm for the architecture.
        let apply_norm = |x: &[f32], weight: &[f32], eps: f32| -> Vec<f32> {
            match norm_type {
                NormType::RmsNorm => self.backend.rms_norm(x, weight, eps),
                NormType::LayerNorm => {
                    // LayerNorm is not yet implemented in Backend trait, fallback to standalone
                    layer_norm(x, weight, None, eps)
                }
            }
        };

        for layer_idx in 0..n_layers {
            let residual = x.clone();

            // ─── Pre-attention norm ───
            // ─── Pre-attention norm ───
            let attn_norm_name = format!("blk.{}.attn_norm.weight", layer_idx);
            if let Ok(attn_norm_weight) = self.model.get_tensor(&attn_norm_name) {
                x = apply_norm(&x, &attn_norm_weight, self.model.norm_eps);
            }

            // ─── QKV projections ───
            let q_proj_name = format!("blk.{}.attn_q.weight", layer_idx);
            let k_proj_name = format!("blk.{}.attn_k.weight", layer_idx);
            let v_proj_name = format!("blk.{}.attn_v.weight", layer_idx);

            let mut attn_ms = 0.0;
            let mut attn_output_vec = None;
            let mut ffn_output_vec = None;

            let (attn_input, ffn_input) = (x.clone(), x.clone());

            // ─── Attention ───
            if let (Ok(q), Ok(k), Ok(v)) = (
                self.mat_vec_weight(&q_proj_name, n_head * head_dim, n_embd, &attn_input),
                self.mat_vec_weight(&k_proj_name, n_head_kv * head_dim, n_embd, &attn_input),
                self.mat_vec_weight(&v_proj_name, n_head_kv * head_dim, n_embd, &attn_input),
            ) {
                let attn_start = Instant::now();
                let mut q = q;
                let mut k = k;

                // ─── QK-norm (Gemma2): per-head RMSNorm after projection, before RoPE ───
                if self.model.has_qk_norm {
                    let qk_norm_eps = self.model.qk_norm_eps;
                    if let Ok(q_norm_weight) = self
                        .model
                        .get_tensor(&format!("blk.{}.attn_q_norm.weight", layer_idx))
                    {
                        for h in 0..n_head {
                            let start = h * head_dim;
                            let slice = &mut q[start..start + head_dim];
                            let normed = self.backend.rms_norm(slice, &q_norm_weight, qk_norm_eps);
                            slice.copy_from_slice(&normed);
                        }
                    }
                    if let Ok(k_norm_weight) = self
                        .model
                        .get_tensor(&format!("blk.{}.attn_k_norm.weight", layer_idx))
                    {
                        for h in 0..n_head_kv {
                            let start = h * head_dim;
                            let slice = &mut k[start..start + head_dim];
                            let normed = self.backend.rms_norm(slice, &k_norm_weight, qk_norm_eps);
                            slice.copy_from_slice(&normed);
                        }
                    }
                }

                let mut kv_cache = self.model.kv_cache.write().expect("lock poisoned");
                let position_offset = kv_cache.get_layer_ref(layer_idx).cur_len;
                let attn_output = multi_head_attention_with_cache(
                    n_head,
                    n_head_kv,
                    head_dim,
                    1,
                    position_offset,
                    &mut q,
                    &mut k,
                    &v,
                    kv_cache.get_layer(layer_idx),
                    rope_config,
                    self.model.sliding_window,
                );

                let attn_out_name = format!("blk.{}.attn_output.weight", layer_idx);
                if let Ok(attn_proj) =
                    self.mat_vec_weight(&attn_out_name, n_embd, n_head * head_dim, &attn_output)
                {
                    attn_output_vec = Some(attn_proj);
                } else {
                    attn_output_vec = Some(attn_output);
                }
                attn_ms = attn_start.elapsed().as_secs_f64() * 1000.0;
            }

            // ─── FFN ───
            let ffn_start = Instant::now();
            let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);

            // For StableLM, FFN uses the same pre-normed input as attention.
            // For standard architectures, FFN has its own norm.
            let ffn_input = if arch == "stablelm" {
                ffn_input // already normed alongside attention
            } else if let Ok(ffn_norm_weight) = self.model.get_tensor(&ffn_norm_name) {
                apply_norm(&x, &ffn_norm_weight, self.model.norm_eps)
            } else {
                x.clone()
            };

            let gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
            let up_name = format!("blk.{}.ffn_up.weight", layer_idx);
            let down_name = format!("blk.{}.ffn_down.weight", layer_idx);

            if self.model.get_tensor(&gate_name).is_ok()
                && self.model.get_tensor(&up_name).is_ok()
                && self.model.get_tensor(&down_name).is_ok()
            {
                let gate_proj =
                    self.mat_vec_weight(&gate_name, self.model.n_ff, n_embd, &ffn_input)?;
                let up_proj = self.mat_vec_weight(&up_name, self.model.n_ff, n_embd, &ffn_input)?;
                let activated_gate = match arch {
                    "gemma" | "gemma2" => self.backend.gelu(&gate_proj),
                    "phi3" | "phi3small" | "phi3.5" => relu_squared(&gate_proj),
                    _ => self.backend.silu(&gate_proj),
                };
                let ffn_hidden = self.backend.mul(&activated_gate, &up_proj);
                let ffn_output =
                    self.mat_vec_weight(&down_name, n_embd, self.model.n_ff, &ffn_hidden)?;
                ffn_output_vec = Some(ffn_output);
            }
            let ffn_ms = ffn_start.elapsed().as_secs_f64() * 1000.0;

            // ─── Residual connection ───
            match arch {
                // StableLM: parallel residual -- x = x + attn_out + ffn_out
                "stablelm" => {
                    if let Some(attn_out) = &attn_output_vec {
                        x = self.backend.add(&residual, attn_out);
                    }
                    if let Some(ffn_out) = &ffn_output_vec {
                        x = self.backend.add(&x, ffn_out);
                    }
                }
                // Standard sequential residual (Llama, Mistral, Qwen2, etc.)
                _ => {
                    // Attention residual
                    if let Some(attn_out) = attn_output_vec {
                        x = self.backend.add(&residual, &attn_out);
                    } else {
                        x = residual.clone();
                    }

                    // Post-attention norm (Gemma/Gemma2)
                    if matches!(arch, "gemma" | "gemma2") {
                        let post_attn_norm_name =
                            format!("blk.{}.post_attention_norm.weight", layer_idx);
                        if let Ok(post_attn_norm_weight) =
                            self.model.get_tensor(&post_attn_norm_name)
                        {
                            x = self.backend.rms_norm(
                                &x,
                                &post_attn_norm_weight,
                                self.model.norm_eps,
                            );
                        }
                    }

                    // FFN residual
                    let ffn_residual = x.clone();
                    if let Some(ffn_out) = ffn_output_vec {
                        x = self.backend.add(&ffn_residual, &ffn_out);
                    } else {
                        x = ffn_residual;
                    }
                }
            }

            profile.layer_times.push((layer_idx, attn_ms, ffn_ms));
        }

        let output_start = Instant::now();
        if let Ok(final_norm) = self.model.get_tensor("output_norm.weight") {
            x = apply_norm(&x, &final_norm, self.model.norm_eps);
        }

        let logits = if let Ok(logits) =
            self.mat_vec_weight("output.weight", self.model.vocab_size, n_embd, &x)
        {
            logits
        } else {
            self.backend
                .mat_vec(&token_embd, self.model.vocab_size, n_embd, &x)
        };
        profile.output_ms = output_start.elapsed().as_secs_f64() * 1000.0;
        profile.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        Ok((logits, profile))
    }

    /// Generate token IDs with profiling information.
    /// Returns (token_ids, average_profile_result).
    pub fn generate_with_profile(
        &self,
        prompt: &str,
        n_predict: usize,
    ) -> anyhow::Result<(Vec<usize>, ProfileResult)> {
        let mut toks = self.encode(prompt);

        if toks.len() > self.config.n_ctx {
            toks.truncate(self.config.n_ctx);
        }

        let mut profiles = Vec::new();

        for _i in 0..n_predict {
            let last_token = *toks.last().unwrap_or(&0);

            match self.forward_pass_with_profile(last_token) {
                Ok((logits, profile)) => {
                    let next_token = sample_logits(&logits, &self.sampling, &toks);
                    toks.push(next_token);
                    profiles.push(profile);

                    if next_token == self.model.eos_token_id {
                        break;
                    }
                }
                Err(_) => {
                    toks.push(0);
                }
            }
        }

        let avg_profile = if profiles.is_empty() {
            ProfileResult::default()
        } else {
            let n = profiles.len() as f64;
            let mut avg = ProfileResult {
                embed_ms: profiles.iter().map(|p| p.embed_ms).sum::<f64>() / n,
                output_ms: profiles.iter().map(|p| p.output_ms).sum::<f64>() / n,
                total_ms: profiles.iter().map(|p| p.total_ms).sum::<f64>() / n,
                ..Default::default()
            };

            let max_layers = profiles
                .iter()
                .map(|p| p.layer_times.len())
                .max()
                .unwrap_or(0);
            for layer_idx in 0..max_layers {
                let mut sum_attn = 0.0;
                let mut sum_ffn = 0.0;
                let mut count = 0.0;
                for p in &profiles {
                    if layer_idx < p.layer_times.len() {
                        sum_attn += p.layer_times[layer_idx].1;
                        sum_ffn += p.layer_times[layer_idx].2;
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    avg.layer_times
                        .push((layer_idx, sum_attn / count, sum_ffn / count));
                }
            }
            avg
        };

        Ok((toks, avg_profile))
    }

    /// Decode a single token id to string.
    pub fn decode_from_id(&self, id: usize) -> String {
        self.tokenizer.decode(&[id])
    }

    /// Decode a slice of token ids to a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        self.tokenizer.decode(ids)
    }
}
