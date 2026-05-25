//! Benchmarks for GGUF file parsing performance.
//!
//! Creates a realistic GGUF file on disk and benchmarks parsing the header
//! and metadata via `GgufReader::from_file`.

use criterion::{Criterion, criterion_group, criterion_main};
use std::io::Write;

fn build_gguf_bytes(n_kv: usize, n_tensors: usize) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&gguf::GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&gguf::GGUF_VERSION.to_le_bytes());
    data.extend_from_slice(&(n_tensors as i64).to_le_bytes());
    data.extend_from_slice(&(n_kv as i64).to_le_bytes());

    for i in 0..n_kv {
        let key = format!("metadata.key_{i}");
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&8i32.to_le_bytes()); // GgufType::String
        let val = format!("value_{i}");
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val.as_bytes());
    }

    for i in 0..n_tensors {
        let name = format!("tensor_{i}.weight");
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&256i64.to_le_bytes());
        data.extend_from_slice(&4096i64.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes()); // GgmlType::F32
        data.extend_from_slice(&0u64.to_le_bytes());
    }
    data
}

fn gguf_read_benchmark(c: &mut Criterion) {
    let data = build_gguf_bytes(10, 50);

    // Write a temp GGUF file once (outside the measured loop).
    let tmp_dir = std::env::temp_dir();
    let gguf_path = tmp_dir.join("gguf_bench_test.gguf");
    std::fs::write(&gguf_path, &data).expect("failed to write temp GGUF");
    let file_size = data.len();

    c.bench_function("gguf_from_file", |b| {
        b.iter(|| {
            let reader = gguf::GgufReader::from_file(&gguf_path).unwrap();
            std::hint::black_box(reader);
        });
    });

    // Benchmark metadata iteration
    c.bench_function("gguf_iterate_tensors", |b| {
        b.iter(|| {
            let reader = gguf::GgufReader::from_file(&gguf_path).unwrap();
            let tensors: Vec<_> = reader.tensors().iter().map(|t| t.name.clone()).collect();
            std::hint::black_box(tensors);
        });
    });

    // Report file size for context
    eprintln!("\nBenchmark GGUF file: {file_size} bytes, 10 KV pairs, 50 tensors\n");

    std::fs::remove_file(&gguf_path).ok();
}

criterion_group!(benches, gguf_read_benchmark);
criterion_main!(benches);
