use criterion::{Criterion, criterion_group, criterion_main};
use ggml::backend::{Backend, QuantType};
use ggml::{DType, Tensor};
use ggml_cpu::{CpuBackend, cpu_features, dot_f32};

fn fill_tensor(tensor: &mut Tensor, seed: u64) {
    let data = tensor.data();
    let f32_data = unsafe {
        std::slice::from_raw_parts_mut(data.as_ptr().cast::<f32>() as *mut f32, data.len() / 4)
    };
    for (i, v) in f32_data.iter_mut().enumerate() {
        *v = ((i as u64).wrapping_mul(seed).wrapping_add(i as u64) % 1000) as f32 * 0.001;
    }
}

fn matmul_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new(1, 0);

    let sizes = [(64, 64), (128, 128), (256, 256), (512, 512)];
    for &(m, k) in &sizes {
        let n = m;
        let mut group = c.benchmark_group(&format!("matmul_{m}x{k}"));

        let mut a = Tensor::new(DType::F32, &[m, k]);
        let mut b = Tensor::new(DType::F32, &[n, k]);
        fill_tensor(&mut a, 42);
        fill_tensor(&mut b, 137);

        group.bench_function("single_thread", |bencher| {
            bencher.iter(|| backend.matmul(&a, &b))
        });

        let parallel = CpuBackend::new(0, 0);
        group.bench_function("parallel", |bencher| {
            bencher.iter(|| parallel.matmul(&a, &b))
        });

        group.finish();
    }
}

fn parallel_threshold_benchmark(c: &mut Criterion) {
    // Small matrices where thread overhead matters
    let sizes = [(8, 64), (16, 64), (32, 64), (64, 64), (128, 64), (256, 64)];
    for &(m, k) in &sizes {
        let n = m;
        let mut group = c.benchmark_group(&format!("parallel_threshold_{m}x{k}"));
        let mut a = Tensor::new(DType::F32, &[m, k]);
        let mut b_tensor = Tensor::new(DType::F32, &[n, k]);
        fill_tensor(&mut a, 42);
        fill_tensor(&mut b_tensor, 137);

        // Single-thread (threshold always blocks)
        let single = CpuBackend::new_with_min_rows(0, usize::MAX, 0);
        group.bench_function("single", |bencher| {
            bencher.iter(|| single.matmul(&a, &b_tensor))
        });

        // Auto-parallel with default threshold (128)
        let parallel = CpuBackend::new_with_min_rows(0, 128, 0);
        group.bench_function("thresh128", |bencher| {
            bencher.iter(|| parallel.matmul(&a, &b_tensor))
        });

        // Low threshold (16) — parallel on small matrices
        let low_thresh = CpuBackend::new_with_min_rows(0, 16, 0);
        group.bench_function("thresh16", |bencher| {
            bencher.iter(|| low_thresh.matmul(&a, &b_tensor))
        });

        group.finish();
    }
}

fn dot_product_benchmark(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096];
    for &n in &sizes {
        let mut group = c.benchmark_group(&format!("dot_{n}"));

        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) * 0.001).collect();

        group.bench_function("simd", |b| b.iter(|| dot_f32(&x, &y)));

        group.finish();
    }
}

fn cpu_feature_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_features");
    group.bench_function("detect_sse4_2", |b| b.iter(|| cpu_features::has_sse4_2()));
    group.bench_function("detect_avx", |b| b.iter(|| cpu_features::has_avx()));
    group.finish();
}

/// Build a synthetic Q4_0 quantized weight matrix from f32 values.
fn build_q4_0_weights(values: &[f32]) -> Vec<u8> {
    use half::f16;
    let n = values.len();
    // Pad to block boundary
    let n_blocks = n.div_ceil(32);
    let mut out = Vec::with_capacity(n_blocks * 18);
    for b in 0..n_blocks {
        let start = b * 32;
        // Compute scale = max absolute value / 7 (Q4_0 range is -8..7)
        let mut max_abs = 0.0f32;
        for i in 0..32 {
            let idx = start + i;
            let v = if idx < n { values[idx] } else { 0.0 };
            max_abs = max_abs.max(v.abs());
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        let inv_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };
        let scale_bytes = f16::from_f32(scale).to_le_bytes();
        out.extend_from_slice(&scale_bytes);
        for i in 0..16 {
            let idx0 = start + i * 2;
            let idx1 = start + i * 2 + 1;
            let v0 = if idx0 < n { values[idx0] } else { 0.0 };
            let v1 = if idx1 < n { values[idx1] } else { 0.0 };
            let q0 = ((v0 * inv_scale).round() as i8).clamp(-8, 7).wrapping_add(8) as u8 & 0x0F;
            let q1 = ((v1 * inv_scale).round() as i8).clamp(-8, 7).wrapping_add(8) as u8 & 0x0F;
            out.push(q0 | (q1 << 4));
        }
    }
    out
}

fn quantized_matvec_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new(1, 0);
    let parallel_backend = CpuBackend::new(0, 0);

    let sizes = [(256, 256), (512, 512), (1024, 4096)];
    for &(rows, cols) in &sizes {
        let mut group = c.benchmark_group(&format!("quant_matvec_{rows}x{cols}"));

        // Generate f32 weight data
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as u64).wrapping_mul(42).wrapping_add(7) % 1000) as f32 * 0.001)
            .collect();
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.001).collect();
        let quantized = build_q4_0_weights(&weight);

        // F32 baseline
        group.bench_function("f32_baseline", |b| {
            b.iter(|| backend.mat_vec(&weight, rows, cols, &input))
        });

        // Q4_0 quantized (single thread)
        group.bench_function("q4_0_quant", |b| {
            b.iter(|| backend.mat_vec_quant(&quantized, QuantType::Q4_0, rows, cols, &input))
        });

        // Q4_0 quantized (parallel)
        group.bench_function("q4_0_parallel", |b| {
            b.iter(|| parallel_backend.mat_vec_quant(&quantized, QuantType::Q4_0, rows, cols, &input))
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    matmul_benchmark,
    parallel_threshold_benchmark,
    dot_product_benchmark,
    cpu_feature_benchmark,
    quantized_matvec_benchmark
);
criterion_main!(benches);
