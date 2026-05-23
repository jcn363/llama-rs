use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llama::{
    KvCache, RoPEConfig, RopeScaleType, apply_rope_with_config, multi_head_attention_with_cache,
};

fn rope_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_apply");

    for seq_len in [64usize, 256, 1024, 4096] {
        let head_dim = 64;
        let size = seq_len * head_dim;
        let mut x: Vec<f32> = (0..size).map(|i| (i % 100) as f32 * 0.01).collect();

        let config = RoPEConfig::new(10000.0);

        group.bench_function(&format!("vanilla_seq_{seq_len}"), |b| {
            b.iter(|| {
                apply_rope_with_config(
                    black_box(&mut x),
                    black_box(seq_len),
                    black_box(head_dim),
                    black_box(0),
                    black_box(&config),
                    black_box(None),
                );
            })
        });
    }

    // NTK scaling variant at 4096
    let head_dim = 64;
    let seq_len = 4096usize;
    let size = seq_len * head_dim;
    let mut x: Vec<f32> = (0..size).map(|i| (i % 100) as f32 * 0.01).collect();

    let ntk_config = RoPEConfig {
        theta: 10000.0,
        scale_type: RopeScaleType::NtkAware,
        scale_factor: 2.0,
        original_max_seq_len: 4096,
        partial_dim: None,
    };

    group.bench_function("ntk_seq_4096", |b| {
        b.iter(|| {
            apply_rope_with_config(
                black_box(&mut x),
                black_box(seq_len),
                black_box(head_dim),
                black_box(0),
                black_box(&ntk_config),
                black_box(None),
            );
        })
    });

    // Partial rotation (Phi-3 style)
    let partial_config = RoPEConfig {
        theta: 10000.0,
        scale_type: RopeScaleType::None,
        scale_factor: 1.0,
        original_max_seq_len: 4096,
        partial_dim: Some(32),
    };
    let mut x_partial: Vec<f32> = (0..size).map(|i| (i % 100) as f32 * 0.01).collect();

    group.bench_function("partial_rot_seq_4096", |b| {
        b.iter(|| {
            apply_rope_with_config(
                black_box(&mut x_partial),
                black_box(seq_len),
                black_box(head_dim),
                black_box(0),
                black_box(&partial_config),
                black_box(None),
            );
        })
    });

    group.finish();
}

fn flash_attention_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");

    let n_head = 8;
    let n_head_kv = 8;
    let head_dim = 64;

    for seq_len in [64usize, 256, 512, 1024] {
        // Pre-fill KV cache with realistic data (+2 for the new token being pushed)
        let mut kv_cache = KvCache::new(seq_len + 2, n_head_kv, head_dim);
        let k: Vec<f32> = (0..n_head_kv * head_dim)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();
        let v: Vec<f32> = (0..n_head_kv * head_dim)
            .map(|i| ((i + 50) % 100) as f32 * 0.01)
            .collect();
        for _ in 0..seq_len {
            kv_cache.push(&k, &v);
        }

        let q: Vec<f32> = (0..n_head * head_dim)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();
        let k_new: Vec<f32> = (0..n_head_kv * head_dim)
            .map(|i| (i % 100) as f32 * 0.02)
            .collect();
        let v_new: Vec<f32> = (0..n_head_kv * head_dim)
            .map(|i| (i % 100) as f32 * 0.03)
            .collect();
        let config = RoPEConfig::new(10000.0);

        group.bench_function(&format!("full_attn_seq_{seq_len}"), |b| {
            b.iter(|| {
                kv_cache.truncate(seq_len);
                let mut q_clone = q.clone();
                let mut k_clone = k_new.clone();
                let _output = multi_head_attention_with_cache(
                    black_box(n_head),
                    black_box(n_head_kv),
                    black_box(head_dim),
                    black_box(1),
                    black_box(seq_len),
                    black_box(&mut q_clone),
                    black_box(&mut k_clone),
                    black_box(&v_new),
                    black_box(&mut kv_cache),
                    black_box(&config),
                    black_box(None),
                );
            })
        });
    }

    // Sliding window variant: window=512, seq_len=2048
    let mut kv_cache = KvCache::new(4097, n_head_kv, head_dim);
    let k: Vec<f32> = (0..n_head_kv * head_dim)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();
    let v: Vec<f32> = (0..n_head_kv * head_dim)
        .map(|i| ((i + 50) % 100) as f32 * 0.01)
        .collect();
    for _ in 0..2048 {
        kv_cache.push(&k, &v);
    }
    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();
    let k_new: Vec<f32> = (0..n_head_kv * head_dim)
        .map(|i| (i % 100) as f32 * 0.02)
        .collect();
    let v_new: Vec<f32> = (0..n_head_kv * head_dim)
        .map(|i| (i % 100) as f32 * 0.03)
        .collect();
    let config = RoPEConfig::new(10000.0);

    group.bench_function("window_512_seq_2048", |b| {
        b.iter(|| {
            kv_cache.truncate(2048);
            let mut q_clone = q.clone();
            let mut k_clone = k_new.clone();
            let _output = multi_head_attention_with_cache(
                black_box(8),
                black_box(8),
                black_box(64),
                black_box(1),
                black_box(2048),
                black_box(&mut q_clone),
                black_box(&mut k_clone),
                black_box(&v_new),
                black_box(&mut kv_cache),
                black_box(&config),
                black_box(Some(512)),
            );
        })
    });

    group.finish();
}

criterion_group!(benches, rope_benchmark, flash_attention_benchmark);
criterion_main!(benches);
