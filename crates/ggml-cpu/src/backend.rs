//! Minimal CPU backend stub for ggml-cpu.
//! This provides just enough implementation to satisfy the workspace compilation
//! while the full backend is under development.

use ggml::backend::{Backend, BackendInfo};
use ggml::{DType, Tensor};
use std::cell::RefCell;

thread_local! {
    static BUMP_ALLOCATOR: RefCell<Option<bumpalo::Bump>> = const { RefCell::new(None) };
}

/// Resets the thread‑local bump allocator.
pub fn reset_bump_allocator() {
    BUMP_ALLOCATOR.with(|cell| *cell.borrow_mut() = None);
}

/// Simple CPU backend used by the ggml‑cpu crate.
#[derive(Debug, Clone, Copy)]
pub struct CpuBackend {
    /// Number of threads for parallel computation.
    pub n_threads: usize,
    /// Minimum number of matrix rows to use parallel execution.
    pub parallel_min_rows: usize,
    /// Size of the thread-local memory pool in bytes (0 = disabled).
    pub memory_pool_size: usize,
}

impl CpuBackend {
    /// Returns the number of threads.
    #[must_use]
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Returns the minimum parallel rows threshold.
    #[must_use]
    pub fn parallel_min_rows(&self) -> usize {
        self.parallel_min_rows
    }

    /// Returns the memory pool size in bytes.
    #[must_use]
    pub fn memory_pool_size(&self) -> usize {
        self.memory_pool_size
    }
}

impl Backend for CpuBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "CPU",
            is_available: true,
            total_memory: 0,
            free_memory: 0,
            parallelism: self.n_threads,
        }
    }

    fn mat_vec(&self, weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        ggml::backend::default_mat_vec(weight, rows, cols, input)
    }
}

impl CpuBackend {
    /// Create a new backend.
    #[must_use]
    pub fn new(n_threads: usize, memory_pool_size: usize) -> Self {
        Self::new_with_min_rows(n_threads, 0, memory_pool_size)
    }

    /// Create a new backend with a custom minimum row count.
    #[must_use]
    pub fn new_with_min_rows(
        n_threads: usize,
        parallel_min_rows: usize,
        memory_pool_size: usize,
    ) -> Self {
        Self {
            n_threads: if n_threads == 0 {
                std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
            } else {
                n_threads
            },
            parallel_min_rows: if parallel_min_rows == 0 {
                128
            } else {
                parallel_min_rows
            },
            memory_pool_size,
        }
    }

    /// Matrix multiply: `C = A @ B^T` where A has shape `[m, k]`, B has shape `[n, k]`.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions of A and B do not match, or if the tensors
    /// are not 2-dimensional or not F32.
    #[must_use]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(a_shape.len(), 2, "A must be 2-dimensional");
        assert_eq!(b_shape.len(), 2, "B must be 2-dimensional");
        assert_eq!(a.dtype(), DType::F32, "A must be F32");
        assert_eq!(b.dtype(), DType::F32, "B must be F32");
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[0];
        let k2 = b_shape[1];
        assert_eq!(k, k2, "inner dimensions must match (A: {k}, B: {k2})");
        let a_f32: &[f32] = bytemuck::cast_slice(a.data());
        let b_f32: &[f32] = bytemuck::cast_slice(b.data());
        let mut c = vec![0.0f32; m * n];
        crate::matmul::matmul_f32(
            a_f32,
            b_f32,
            &mut c,
            m,
            n,
            k,
            self.n_threads,
            self.parallel_min_rows,
        );
        Tensor::from_f32(&[m, n], &c)
    }
}
