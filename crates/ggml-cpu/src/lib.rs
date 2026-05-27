//! CPU backend for ggml, optimized for AMD Opteron 3280 (bdver1).
//!
//! This crate implements tensor operations for CPU execution, with explicit
//! optimizations for SSE4.2 and AVX instruction sets available on bdver1.
//!
//! # Hardware Target
//!
//! - **CPU:** AMD Opteron 3280 (Bulldozer bdver1)
//! - **Supported:** SSE4.2, AVX, AES, POPCNT
//! - **Not supported:** AVX2, FMA, F16C, BMI2, AVX512

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(
    clippy::many_single_char_names,
    clippy::wildcard_imports,
    clippy::missing_panics_doc,
    clippy::items_after_statements,
    clippy::too_many_arguments,
    clippy::cast_ptr_alignment,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::doc_markdown
)]

pub mod cpu_features;
pub use cpu_features::{has_aes, has_avx, has_popcnt, has_sse4_2};

pub mod reduce;

pub mod activations;

mod simd;
pub use simd::dot_f32;

pub mod elementwise;

mod matmul;
pub use matmul::matmul_f32;

pub mod other;

mod backend;
pub use backend::{CpuBackend, reset_bump_allocator};

pub mod quant_dot;

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ggml::{DType, Tensor};

    #[test]
    fn cpu_backend_should_default_thread_count() {
        let backend = CpuBackend::new(0, 0);
        assert!(backend.n_threads() > 0);
    }

    #[test]
    fn cpu_backend_should_use_explicit_thread_count() {
        let backend = CpuBackend::new(4, 0);
        assert_eq!(backend.n_threads(), 4);
    }

    #[test]
    fn cpu_features_should_detect_sse4_2() {
        assert!(cpu_features::has_sse4_2());
    }

    #[test]
    fn cpu_features_should_detect_avx() {
        assert!(cpu_features::has_avx());
    }

    #[test]
    fn dot_f32_should_compute_correct_result() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = dot_f32(&x, &y);
        assert!((result - 70.0).abs() < 0.001);
    }

    #[test]
    fn dot_f32_should_handle_empty() {
        assert_eq!(dot_f32(&[], &[]), 0.0);
    }

    #[test]
    fn dot_f32_should_handle_large_vectors() {
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        // sum(i * 2i) = 2 * sum(i^2) = 2 * n*(n-1)*(2n-1)/6
        let expected = 2.0 * (n as f64) * ((n - 1) as f64) * ((2 * n - 1) as f64) / 6.0;
        let result = f64::from(dot_f32(&x, &y));
        assert!((result - expected).abs() < expected * 0.001);
    }

    #[test]
    fn matmul_f32_should_compute_correct_result() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A * B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        //             = [[17, 23], [39, 53]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2, 1, 0);

        assert!((c[0] - 17.0).abs() < 0.001, "c[0] = {}", c[0]);
        assert!((c[1] - 23.0).abs() < 0.001, "c[1] = {}", c[1]);
        assert!((c[2] - 39.0).abs() < 0.001, "c[2] = {}", c[2]);
        assert!((c[3] - 53.0).abs() < 0.001, "c[3] = {}", c[3]);
    }

    #[test]
    fn matmul_f32_should_handle_non_square() {
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[7, 8, 9], [10, 11, 12], [13, 14, 15]]  (3x3)
        // C = A * B^T  (2x3)
        let a: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let b: Vec<f32> = (7..=15).map(|x| x as f32).collect();
        let mut c = vec![0.0; 6];
        matmul_f32(&a, &b, &mut c, 2, 3, 3, 1, 0);

        // C[0,0] = 1*7 + 2*8 + 3*9 = 50
        assert!((c[0] - 50.0).abs() < 0.001);
        // C[0,1] = 1*10 + 2*11 + 3*12 = 68
        assert!((c[1] - 68.0).abs() < 0.001);
    }

    #[test]
    fn matmul_should_require_2d_tensors() {
        let backend = CpuBackend::new(1, 0);
        let a = Tensor::new(DType::F32, &[2, 3]);
        let b = Tensor::new(DType::F32, &[4, 3]);
        let _result = backend.matmul(&a, &b);
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match")]
    fn matmul_should_panic_on_incompatible_shapes() {
        let backend = CpuBackend::new(1, 0);
        let a = Tensor::new(DType::F32, &[2, 3]);
        let b = Tensor::new(DType::F32, &[4, 5]);
        let _result = backend.matmul(&a, &b);
    }

    #[test]
    fn matmul_parallel_should_match_single_thread() {
        let n = 64;
        let k = 128;
        let a: Vec<f32> = (0..n * k).map(|i| (i % 100) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i + 37) % 100) as f32 * 0.01).collect();

        let mut c1 = vec![0.0; n * n];
        matmul_f32(&a, &b, &mut c1, n, n, k, 1, 0);

        let mut c2 = vec![0.0; n * n];
        matmul_f32(&a, &b, &mut c2, n, n, k, 4, 0);

        for i in 0..n * n {
            let diff = (c1[i] - c2[i]).abs();
            assert!(
                diff < 0.001,
                "mismatch at index {i}: {} vs {}",
                c1[i],
                c2[i]
            );
        }
    }
}
