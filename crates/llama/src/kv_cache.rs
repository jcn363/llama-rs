// Per-layer KV cache for multi-head attention.
// Supports MHA, GQA (Grouped Query Attention), and MQA (Multi-Query Attention).

/// KV cache strategy: controls when and how the cache is trimmed.
/// Strategy for managing the KV cache memory.
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Full cache — never trim (default).
    Full,
    /// Prefix caching — trim during generation for long contexts.
    Prefix,
    /// Sliding‑window caching — keep only the most recent `size` tokens.
    /// `size` – maximum number of tokens to retain.
    SlidingWindow { size: usize },
    /// Prefix‑only caching — keep only the initial prefix up to a fixed length.
    PrefixOnly,
}

/// Per-layer KV cache.
///
/// Layout:
/// - `keys`  : [max_seq, n_head_kv, head_dim]
/// - `values`: [max_seq, n_head_kv, head_dim]
///
/// Flattened in row‑major order (seq -> head -> dim).
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Maximum sequence length this cache can hold.
    pub max_seq: usize,
    /// Number of key/value heads (for GQA/MQA).
    pub n_head_kv: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Flattened key cache: [max_seq, n_head_kv, head_dim].
    pub keys: Vec<f32>,
    /// Flattened value cache: [max_seq, n_head_kv, head_dim].
    pub values: Vec<f32>,
    /// Current number of cached positions.
    pub cur_len: usize,
}

impl KvCache {
    /// Create a new KV cache for a single layer.
    pub fn new(max_seq: usize, n_head_kv: usize, head_dim: usize) -> Self {
        let size = max_seq * n_head_kv * head_dim;
        Self {
            max_seq,
            n_head_kv,
            head_dim,
            keys: vec![0.0; size],
            values: vec![0.0; size],
            cur_len: 0,
        }
    }

    /// Reset the cache (clear all entries). O(1) — just sets `cur_len` to 0.
    /// Memory is not zeroed; stale values will be overwritten by new pushes.
    pub fn reset(&mut self) {
        self.cur_len = 0;
    }

    /// Append a new token's key and value vectors.
    /// `k` and `v` must be of length `n_head_kv * head_dim`.
    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.n_head_kv * self.head_dim);
        assert_eq!(v.len(), self.n_head_kv * self.head_dim);
        assert!(self.cur_len < self.max_seq, "KV cache overflow");
        let offset = self.cur_len * self.n_head_kv * self.head_dim;
        self.keys[offset..offset + k.len()].copy_from_slice(k);
        self.values[offset..offset + v.len()].copy_from_slice(v);
        self.cur_len += 1;
    }

    /// Truncate the cache to a new length (for prefix caching).
    /// If `new_len > cur_len`, this is a no-op.
    pub fn truncate(&mut self, new_len: usize) {
        self.cur_len = self.cur_len.min(new_len);
    }

    /// Append multiple tokens' key and value vectors at once.
    /// `k` and `v` must be of length `n_tokens * n_head_kv * head_dim`.
    pub fn push_batch(&mut self, k: &[f32], v: &[f32], n_tokens: usize) {
        let stride = self.n_head_kv * self.head_dim;
        assert_eq!(k.len(), n_tokens * stride);
        assert_eq!(v.len(), n_tokens * stride);
        assert!(self.cur_len + n_tokens <= self.max_seq, "KV cache overflow");
        let offset = self.cur_len * stride;
        self.keys[offset..offset + k.len()].copy_from_slice(k);
        self.values[offset..offset + v.len()].copy_from_slice(v);
        self.cur_len += n_tokens;
    }

    /// Retrieve a slice for a specific head and position.
    /// Returns (key_slice, value_slice) each of length `head_dim`.
    pub fn get(&self, pos: usize, head: usize) -> (&[f32], &[f32]) {
        assert!(pos < self.cur_len);
        assert!(head < self.n_head_kv);
        let base = (pos * self.n_head_kv + head) * self.head_dim;
        (
            &self.keys[base..base + self.head_dim],
            &self.values[base..base + self.head_dim],
        )
    }
}

/// Multi-layer KV cache manager.
///
/// Holds one KvCache per transformer layer.
#[derive(Debug)]
pub struct KvCacheManager {
    /// Per-layer KV caches, indexed by layer index.
    pub caches: Vec<KvCache>,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Cache eviction strategy.
    pub strategy: CacheStrategy,
}

impl KvCacheManager {
    /// Create a new KV cache manager with one cache per layer (default Full strategy).
    pub fn new(n_layers: usize, max_seq: usize, n_head_kv: usize, head_dim: usize) -> Self {
        Self::with_strategy(n_layers, max_seq, n_head_kv, head_dim, CacheStrategy::Full)
    }

    /// Create a KV cache manager with a custom eviction strategy.
    pub fn with_strategy(
        n_layers: usize,
        max_seq: usize,
        n_head_kv: usize,
        head_dim: usize,
        strategy: CacheStrategy,
    ) -> Self {
        Self {
            caches: (0..n_layers)
                .map(|_| KvCache::new(max_seq, n_head_kv, head_dim))
                .collect(),
            n_layers,
            strategy,
        }
    }

    /// Reset all layer caches.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Get mutable reference to a specific layer's cache.
    pub fn get_layer(&mut self, layer: usize) -> &mut KvCache {
        &mut self.caches[layer]
    }

    /// Get immutable reference to a specific layer's cache.
    pub fn get_layer_ref(&self, layer: usize) -> &KvCache {
        &self.caches[layer]
    }

    /// Truncate all layer caches to new_len.
    pub fn truncate_all(&mut self, new_len: usize) {
        for cache in &mut self.caches {
            cache.truncate(new_len);
        }
    }

    /// Get current sequence length (min across layers).
    pub fn cur_len(&self) -> usize {
        self.caches.first().map_or(0, |c| c.cur_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_push_and_get() {
        let mut cache = KvCache::new(4, 2, 3);
        // Push one token: 2 heads × 3 dims
        cache.push(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        );
        assert_eq!(cache.cur_len, 1);
        let (k, v) = cache.get(0, 0);
        assert_eq!(k, &[1.0, 2.0, 3.0]);
        assert_eq!(v, &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_kv_cache_push_batch() {
        let mut cache = KvCache::new(4, 2, 3);
        // Push 2 tokens at once
        cache.push_batch(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[0.0; 12],
            2,
        );
        assert_eq!(cache.cur_len, 2);
        let (k, _) = cache.get(1, 0);
        assert_eq!(k, &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_kv_cache_o1_reset() {
        let mut cache = KvCache::new(4, 1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.cur_len, 1);
        cache.reset();
        assert_eq!(cache.cur_len, 0);
        // O(1) reset does not zero memory — data remains but is inaccessible
        assert_eq!(cache.keys[0], 1.0);
    }

    #[test]
    fn test_kv_cache_truncate() {
        let mut cache = KvCache::new(4, 1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]);
        cache.push(&[5.0, 6.0], &[7.0, 8.0]);
        assert_eq!(cache.cur_len, 2);
        cache.truncate(1);
        assert_eq!(cache.cur_len, 1);
        let (k, v) = cache.get(0, 0);
        assert_eq!(k, &[1.0, 2.0]);
        assert_eq!(v, &[3.0, 4.0]);
    }
}
