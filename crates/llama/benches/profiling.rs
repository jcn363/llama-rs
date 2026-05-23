use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llama::{InferenceContext, Model, ModelConfig};
use std::path::Path;
use std::sync::Arc;

/// Benchmark forward pass with profiling.
fn bench_forward_pass_profile(c: &mut Criterion) {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test-models/tiny-llm-Q4_K_M.gguf");

    if !model_path.exists() {
        println!("Skipping forward pass benchmark: test model not found");
        return;
    }

    let model = Model::load_from_gguf(&model_path, false).expect("Failed to load model");
    let ctx = InferenceContext::new(Arc::new(model), ModelConfig::default());

    c.bench_function("forward_pass_with_profile", |b| {
        b.iter(|| {
            let (toks, profile) = ctx
                .generate_with_profile("Hello", 1)
                .expect("Failed to generate");
            // Test both human-readable report and JSON export
            let report = profile.report();
            let json = profile.to_json().expect("Failed to export to JSON");
            black_box((toks, report, json))
        })
    });
}

/// Benchmark generating multiple tokens with profiling.
fn bench_generate_profile(c: &mut Criterion) {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test-models/tiny-llm-Q4_K_M.gguf");

    if !model_path.exists() {
        println!("Skipping generate benchmark: test model not found");
        return;
    }

    let model = Model::load_from_gguf(&model_path, false).expect("Failed to load model");
    let ctx = InferenceContext::new(Arc::new(model), ModelConfig::default());

    c.bench_function("generate_5_tokens_profile", |b| {
        b.iter(|| {
            let (toks, profile) = ctx
                .generate_with_profile("Hello", 5)
                .expect("Failed to generate");
            black_box((toks, profile.report()))
        })
    });
}

criterion_group!(benches, bench_forward_pass_profile, bench_generate_profile);
criterion_main!(benches);
