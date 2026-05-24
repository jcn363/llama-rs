use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llama::KvCache;

fn kv_cache_push_benchmark(c: &mut Criterion) {
    let head_dim = 128;
    let n_head_kv = 8;
    let max_seq = 4096;
    let mut cache = KvCache::new(max_seq, n_head_kv, head_dim);
    let token_len = n_head_kv * head_dim;
    let k = vec![0.1; token_len];
    let v = vec![0.2; token_len];

    c.bench_function("kv_cache_push_single", |b| {
        b.iter(|| {
            cache.reset();
            for _ in 0..1024 {
                cache.push(&k, &v);
            }
            black_box(&cache);
        })
    });

    c.bench_function("kv_cache_push_batch", |b| {
        b.iter(|| {
            cache.reset();
            // Push 1024 tokens in batches of 32
            for chunk in (0..1024).step_by(32) {
                let n_tokens = 32.min(1024 - chunk);
                let k_batch = vec![0.1; n_tokens * token_len];
                let v_batch = vec![0.2; n_tokens * token_len];
                cache.push_batch(&k_batch, &v_batch, n_tokens);
            }
            black_box(&cache);
        })
    });
}

fn kv_cache_reset_benchmark(c: &mut Criterion) {
    let head_dim = 128;
    let n_head_kv = 8;
    let max_seq = 4096;
    let mut cache = KvCache::new(max_seq, n_head_kv, head_dim);
    let token_len = n_head_kv * head_dim;
    let k = vec![0.1; token_len];
    let v = vec![0.2; token_len];

    // Fill the cache
    cache.push_batch(&k, &v, 1024);

    c.bench_function("kv_cache_reset_old_simulated", |b| {
        b.iter(|| {
            // Simulate the old reset by zeroing out the used portion
            let zero_len = cache.cur_len * token_len;
            cache.keys[..zero_len].fill(0.0);
            cache.values[..zero_len].fill(0.0);
            cache.cur_len = 0;
            black_box(&cache);
        })
    });

    c.bench_function("kv_cache_reset_new", |b| {
        b.iter(|| {
            cache.reset(); // O(1) reset
            black_box(&cache);
        })
    });
}

fn kv_cache_prefix_find_benchmark(c: &mut Criterion) {
    // Simulate the prefix finding logic
    let mut cached_tokens = Vec::new();
    for i in 0..2048 {
        cached_tokens.push(i % 1000); // Some repeating pattern
    }

    let mut new_tokens = Vec::new();
    for i in 0..2048 {
        new_tokens.push((i + 10) % 1000); // Shifted by 10
    }

    c.bench_function("kv_cache_prefix_find", |b| {
        b.iter(|| {
            let common_prefix_len = new_tokens
                .iter()
                .zip(cached_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            black_box(common_prefix_len);
        })
    });
}

criterion_group!(
    kv_cache,
    kv_cache_push_benchmark,
    kv_cache_reset_benchmark,
    kv_cache_prefix_find_benchmark
);
criterion_main!(kv_cache);
