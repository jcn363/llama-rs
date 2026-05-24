use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::f32;

// Mock flash attention function for benchmarking
// In a real scenario, this would be the actual flash attention implementation.
// For the purpose of this benchmark, we'll simulate the computation with a simple operation.
fn flash_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    n_head_kv: usize,
    _offset: usize,
    _sliding_window: Option<usize>,
) -> Vec<f32> {
    // This is a placeholder for the actual flash attention computation.
    // We'll just do a simple dot product for each query with the keys to simulate the work.
    let mut output = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        let mut sum = 0.0;
        for j in 0..seq_len * n_head_kv {
            // Simplified: just multiply and accumulate
            sum += q[i] * k[j * head_dim + i] * v[j * head_dim + i];
        }
        output[i] = sum;
    }
    output
}

fn flash_attention_benchmark(c: &mut Criterion) {
    let head_dim = 128;
    let n_head_kv = 8;

    for &seq_len in &[64, 256, 1024, 4096] {
        let q = vec![0.1; head_dim];
        let keys = vec![0.1; seq_len * n_head_kv * head_dim];
        let values = vec![0.1; seq_len * n_head_kv * head_dim];

        c.bench_function(&format!("flash_attn_seq_{}", seq_len), |b| {
            b.iter(|| {
                flash_attention_head(&q, &keys, &values, seq_len, head_dim, n_head_kv, 0, None)
            })
        });
    }
}

// Benchmark for attention with sliding window
fn attention_with_sliding_window_benchmark(c: &mut Criterion) {
    let head_dim = 128;
    let n_head_kv = 8;
    let seq_len = 1024;
    let sliding_window = 256;

    let q = vec![0.1; head_dim];
    let keys = vec![0.1; seq_len * n_head_kv * head_dim];
    let values = vec![0.1; seq_len * n_head_kv * head_dim];

    c.bench_function("flash_attn_with_sliding_window", |b| {
        b.iter(|| {
            flash_attention_head(
                &q,
                &keys,
                &values,
                seq_len,
                head_dim,
                n_head_kv,
                0,
                Some(sliding_window),
            )
        })
    });
}

// Benchmark for legacy attention (materialized attention) for comparison
fn legacy_attention_benchmark(c: &mut Criterion) {
    let head_dim = 128;
    let n_head_kv = 8;
    let seq_len = 1024;

    let q = vec![0.1; head_dim];
    let keys = vec![0.1; seq_len * n_head_kv * head_dim];
    let values = vec![0.1; seq_len * n_head_kv * head_dim];

    // Simulate legacy attention: compute QK^T, softmax, then weighted sum of values
    c.bench_function("legacy_attention_seq_1024", |b| {
        b.iter(|| {
            // Compute QK^T
            let mut qk = vec![0.0f32; seq_len * n_head_kv];
            for i in 0..seq_len * n_head_kv {
                let mut sum = 0.0f32;
                for j in 0..head_dim {
                    sum += q[j] * keys[i * head_dim + j];
                }
                qk[i] = sum;
            }
            // Softmax (simplified, not actually normalized for speed)
            let mut qk_exp = vec![0.0f32; seq_len * n_head_kv];
            let mut max_val = qk[0];
            for &val in &qk {
                if val > max_val {
                    max_val = val;
                }
            }
            let mut sum_exp = 0.0f32;
            for i in 0..seq_len * n_head_kv {
                qk_exp[i] = (qk[i] - max_val).exp();
                sum_exp += qk_exp[i];
            }
            // Weighted sum of values
            let mut output = vec![0.0f32; head_dim];
            for i in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..seq_len * n_head_kv {
                    sum += qk_exp[j] * values[j * head_dim + i] / sum_exp;
                }
                output[i] = sum;
            }
            black_box(output);
        })
    });
}

criterion_group!(
    attention,
    flash_attention_benchmark,
    attention_with_sliding_window_benchmark,
    legacy_attention_benchmark
);
criterion_main!(attention);
