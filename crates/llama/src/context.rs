//! Inference context and configuration.
//!
//! Contains [`ModelConfig`] for inference parameters and [`InferenceContext`]
//! which ties together a model, tokenizer, and sampling configuration for
//! text generation.

use std::sync::Arc;
use std::time::Instant;

use crate::Model;
use crate::attention::multi_head_attention_with_cache;
use crate::inference::{
    SamplingConfig, add_vec, embed_token, gelu, mat_vec, mul_vec, rms_norm, sample_logits, silu,
};
use crate::profile::ProfileResult;

/// Configuration for inference.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_threads: usize,
    pub use_cuda: bool,
    pub n_ctx: usize,
    pub n_batch: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_threads: 4,
            use_cuda: false,
            n_ctx: 2048,
            n_batch: 512,
        }
    }
}

/// Inference context holding state for a model.
#[derive(Debug)]
pub struct InferenceContext {
    pub model: Arc<Model>,
    pub config: ModelConfig,
    pub tokenizer: crate::SimpleTokenizer,
    pub sampling: SamplingConfig,
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
        Self {
            model,
            config,
            tokenizer,
            sampling: SamplingConfig::default(),
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

    /// Generate token IDs for a prompt using actual inference.
    pub fn generate(&self, prompt: &str, n_predict: usize) -> anyhow::Result<Vec<usize>> {
        let mut toks = self.encode(prompt);

        if toks.len() > self.config.n_ctx {
            toks.truncate(self.config.n_ctx);
        }

        for _i in 0..n_predict {
            let last_token = *toks.last().unwrap_or(&0);

            match self.forward_pass(last_token) {
                Ok(logits) => {
                    let next_token = sample_logits(&logits, &self.sampling);
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

    /// Run a single forward pass through the model for a given token.
    /// Returns logits of shape (vocab_size,).
    fn forward_pass(&self, token_id: usize) -> anyhow::Result<Vec<f32>> {
        self.forward_pass_with_profile(token_id)
            .map(|(logits, _profile)| logits)
    }

    /// Run a single forward pass with per-layer profiling.
    /// Returns both logits and timing information.
    fn forward_pass_with_profile(
        &self,
        token_id: usize,
    ) -> anyhow::Result<(Vec<f32>, ProfileResult)> {
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
        let rope_theta = self.model.rope_theta;

        for layer_idx in 0..n_layers {
            let layer_start = Instant::now();
            let residual = x.clone();

            let attn_norm_name = format!("blk.{}.attn_norm.weight", layer_idx);
            if let Ok(attn_norm_weight) = self.model.get_tensor(&attn_norm_name) {
                x = rms_norm(&x, &attn_norm_weight, self.model.norm_eps);
            }

            let q_proj_name = format!("blk.{}.attn_q.weight", layer_idx);
            let k_proj_name = format!("blk.{}.attn_k.weight", layer_idx);
            let v_proj_name = format!("blk.{}.attn_v.weight", layer_idx);

            let mut attn_ms = 0.0;
            if let (Ok(q_weight), Ok(k_weight), Ok(v_weight)) = (
                self.model.get_tensor(&q_proj_name),
                self.model.get_tensor(&k_proj_name),
                self.model.get_tensor(&v_proj_name),
            ) {
                let attn_start = Instant::now();
                let mut q = mat_vec(&q_weight, n_head * head_dim, n_embd, &x);
                let mut k = mat_vec(&k_weight, n_head_kv * head_dim, n_embd, &x);
                let v = mat_vec(&v_weight, n_head_kv * head_dim, n_embd, &x);

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
                    rope_theta,
                );

                let attn_out_name = format!("blk.{}.attn_output.weight", layer_idx);
                if let Ok(attn_out_weight) = self.model.get_tensor(&attn_out_name) {
                    let attn_proj =
                        mat_vec(&attn_out_weight, n_embd, n_head * head_dim, &attn_output);
                    x = add_vec(&residual, &attn_proj);
                } else {
                    x = add_vec(&residual, &attn_output);
                }
                attn_ms = attn_start.elapsed().as_secs_f64() * 1000.0;
            } else {
                x = residual;
            }

            let ffn_start = Instant::now();
            let ffn_residual = x.clone();

            let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
            if let Ok(ffn_norm_weight) = self.model.get_tensor(&ffn_norm_name) {
                x = rms_norm(&x, &ffn_norm_weight, self.model.norm_eps);
            }

            let gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
            let up_name = format!("blk.{}.ffn_up.weight", layer_idx);
            let down_name = format!("blk.{}.ffn_down.weight", layer_idx);

            if let (Ok(gate), Ok(up), Ok(down)) = (
                self.model.get_tensor(&gate_name),
                self.model.get_tensor(&up_name),
                self.model.get_tensor(&down_name),
            ) {
                let gate_proj = mat_vec(&gate, self.model.n_ff, n_embd, &x);
                let up_proj = mat_vec(&up, self.model.n_ff, n_embd, &x);
                let activated_gate = match self.model.architecture.as_str() {
                    "gemma" | "gemma2" => gelu(&gate_proj),
                    _ => silu(&gate_proj),
                };
                let ffn_hidden = mul_vec(&activated_gate, &up_proj);
                let ffn_output = mat_vec(&down, n_embd, self.model.n_ff, &ffn_hidden);
                x = add_vec(&ffn_residual, &ffn_output);
            } else {
                x = ffn_residual;
            }
            let ffn_ms = ffn_start.elapsed().as_secs_f64() * 1000.0;

            profile.layer_times.push((layer_idx, attn_ms, ffn_ms));
            let _layer_total = layer_start.elapsed();
        }

        let output_start = Instant::now();
        if let Ok(final_norm) = self.model.get_tensor("output_norm.weight") {
            x = rms_norm(&x, &final_norm, self.model.norm_eps);
        }

        let logits = if let Ok(output_weight) = self.model.get_tensor("output.weight") {
            mat_vec(&output_weight, self.model.vocab_size, n_embd, &x)
        } else {
            mat_vec(&token_embd, self.model.vocab_size, n_embd, &x)
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
                    let next_token = sample_logits(&logits, &self.sampling);
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
