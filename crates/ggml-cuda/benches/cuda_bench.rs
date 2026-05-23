//! CUDA backend benchmarks.
//!
//! These benchmarks measure CUDA-specific operations for the GTX 1050 target.
//! They run in CI only when CUDA toolkit is available.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ggml::Tensor;
use ggml_cuda::CudaBackend;

fn cuda_init_benchmark(c: &mut Criterion) {
    c.bench_function("cuda_init", |b| {
        b.iter(|| {
            let backend = CudaBackend::new().unwrap_or_default();
            black_box(backend);
        })
    });
}

fn cuda_copy_benchmark(c: &mut Criterion) {
    // Only run meaningful benchmarks when CUDA is actually available
    if let Ok(backend) = CudaBackend::new() {
        let tensor = Tensor::new(ggml::DType::F32, &[1_000_000]); // ~4MB

        c.bench_function("cuda_copy_to_device_4mb", |b| {
            b.iter(|| {
                let dev = backend.copy_to_device(black_box(&tensor)).unwrap();
                black_box(dev);
            })
        });

        if let Ok(dev_tensor) = backend.copy_to_device(&tensor) {
            c.bench_function("cuda_copy_to_host_4mb", |b| {
                b.iter(|| {
                    let host = dev_tensor.to_host().unwrap();
                    black_box(host);
                })
            });
        }
    } else {
        c.bench_function("cuda_init_stub", |b| {
            b.iter(|| {
                let backend = CudaBackend::new().unwrap_or_default();
                black_box(backend);
            })
        });
    }
}

criterion_group!(benches, cuda_init_benchmark, cuda_copy_benchmark);
criterion_main!(benches);
