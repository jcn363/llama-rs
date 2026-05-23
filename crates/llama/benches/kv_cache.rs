use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llama::KvCache;

fn kv_cache_push_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_push_single");

    for seq_len in [128usize, 512, 2048] {
        let mut cache = KvCache::new(seq_len, 8, 64);
        let k: Vec<f32> = (0..8 * 64).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..8 * 64).map(|i| (i + 1) as f32).collect();

        group.bench_function(&format!("seq_{seq_len}"), |b| {
            b.iter(|| {
                cache.reset();
                for _ in 0..seq_len {
                    cache.push(black_box(&k), black_box(&v));
                }
            })
        });
    }
    group.finish();
}

fn kv_cache_push_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_push_batch");

    for seq_len in [128usize, 512, 2048] {
        let mut cache = KvCache::new(seq_len, 8, 64);
        let stride = 8 * 64;
        let total = seq_len * stride;
        let k: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..total).map(|i| (i + 1) as f32).collect();

        group.bench_function(&format!("seq_{seq_len}"), |b| {
            b.iter(|| {
                cache.reset();
                cache.push_batch(black_box(&k), black_box(&v), black_box(seq_len));
            })
        });
    }
    group.finish();
}

fn kv_cache_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_reset");

    for seq_len in [128usize, 512, 2048] {
        let mut cache = KvCache::new(seq_len, 8, 64);
        let stride = 8 * 64;
        let k: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..stride).map(|i| (i + 1) as f32).collect();

        // Pre-fill the cache
        for _ in 0..seq_len {
            cache.push(&k, &v);
        }

        group.bench_function(&format!("seq_{seq_len}"), |b| {
            b.iter(|| {
                cache.reset();
                black_box(&cache.cur_len);
            })
        });
    }
    group.finish();
}

fn kv_cache_truncate(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_truncate");

    for seq_len in [128usize, 512, 2048] {
        let mut cache = KvCache::new(seq_len, 8, 64);
        let stride = 8 * 64;
        let k: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..stride).map(|i| (i + 1) as f32).collect();

        for _ in 0..seq_len {
            cache.push(&k, &v);
        }

        let half = seq_len / 2;
        group.bench_function(&format!("seq_{seq_len}"), |b| {
            b.iter(|| {
                cache.truncate(black_box(half));
                black_box(&cache.cur_len);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    kv_cache_push_single,
    kv_cache_push_batch,
    kv_cache_reset,
    kv_cache_truncate
);
criterion_main!(benches);
