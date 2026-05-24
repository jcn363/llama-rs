/// Profiling structures and utilities for the Llama crate.
use serde::{Deserialize, Serialize};

/// Profiling results for a single forward pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileResult {
    /// Time to load embedding (ms).
    pub embed_ms: f64,
    /// Per-layer timing: (layer_idx, attention_ms, ffn_ms).
    pub layer_times: Vec<(usize, f64, f64)>,
    /// Time for final norm + output projection (ms).
    pub output_ms: f64,
    /// Total forward pass time (ms).
    pub total_ms: f64,
}

impl ProfileResult {
    /// Get the total time spent in attention across all layers.
    pub fn total_attention_ms(&self) -> f64 {
        self.layer_times.iter().map(|(_, a, _)| a).sum()
    }

    /// Get the total time spent in FFN across all layers.
    pub fn total_ffn_ms(&self) -> f64 {
        self.layer_times.iter().map(|(_, _, f)| f).sum()
    }

    /// Format as a human‑readable report.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Forward Pass Profile (total: {:.2}ms)\n",
            self.total_ms
        ));
        s.push_str(&format!(
            "  Embedding:     {:.2}ms ({:.1}%)\n",
            self.embed_ms,
            self.embed_ms / self.total_ms * 100.0
        ));
        s.push_str(&format!(
            "  Attention:     {:.2}ms ({:.1}%)\n",
            self.total_attention_ms(),
            self.total_attention_ms() / self.total_ms * 100.0
        ));
        s.push_str(&format!(
            "  FFN:           {:.2}ms ({:.1}%)\n",
            self.total_ffn_ms(),
            self.total_ffn_ms() / self.total_ms * 100.0
        ));
        s.push_str(&format!(
            "  Output:        {:.2}ms ({:.1}%)\n",
            self.output_ms,
            self.output_ms / self.total_ms * 100.0
        ));
        s.push_str("  Per-layer breakdown:\n");
        for (idx, attn, ffn) in &self.layer_times {
            s.push_str(&format!(
                "    Layer {:>2}: attn={:.2}ms, ffn={:.2}ms\n",
                idx, attn, ffn
            ));
        }
        s
    }

    /// Export profiling results as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
