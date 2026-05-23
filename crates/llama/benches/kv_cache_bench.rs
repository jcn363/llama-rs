use criterion::{criterion_group, criterion_main, Criterion};
use llama::{KvCacheManager, CacheStrategy};

fn bench_kv_cache_full(c: &mut Criterion) {
    let mut manager = KvCacheManager::new();
    // Fill with 10,000 tokens (key+value each)
    for i in 0..10_000 {
        manager.push(i, i);
    }
    c.bench_function("kv_cache_full", |b| b.iter(|| manager.enforce_strategy()));
}

fn bench_kv_cache_sliding(c: &mut Criterion) {
    let mut manager = KvCacheManager::with_strategy(CacheStrategy::SlidingWindow { size: 1024 });
    for i in 0..10_000 {
        manager.push(i, i);
    }
    c.bench_function("kv_cache_sliding", |b| b.iter(|| manager.enforce_strategy()));
}

criterion_group!(benches, bench_kv_cache_full, bench_kv_cache_sliding);
criterion_main!(benches);
