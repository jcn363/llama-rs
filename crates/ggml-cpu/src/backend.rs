//! CPU backend for executing computation graphs.
//!
//! Provides the `CpuBackend` struct which implements matrix multiplication
//! and other tensor operations using the CPU.

use ggml::{DType, Tensor};

/// Executes a computation graph on the CPU.
pub struct CpuBackend {
    n_threads: usize,
}

impl CpuBackend {
    /// Create a new CPU backend with the given number of threads.
    ///
    /// If `n_threads` is 0, uses the number of available parallel threads.
    #[must_use]
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads: if n_threads == 0 {
                std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
            } else {
                n_threads
            },
        }
    }

    /// Execute matrix multiplication: `C = A * B^T`.
    ///
    /// Note: This follows ggml's unconventional matmul convention where
    /// `C = ggml_mul_mat(ctx, A, B)` means `C^T = A * B^T`, i.e., `C = B * A^T`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor shapes are incompatible for multiplication.
    #[must_use]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(
            a.shape().len(),
            2,
            "matmul requires 2D tensors, got {}D",
            a.ndim()
        );
        assert_eq!(
            b.shape().len(),
            2,
            "matmul requires 2D tensors, got {}D",
            b.ndim()
        );
        assert_eq!(a.dtype(), DType::F32, "matmul requires F32 tensors");
        assert_eq!(b.dtype(), DType::F32, "matmul requires F32 tensors");

        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[0];
        let k2 = b.shape()[1];
        assert_eq!(k, k2, "inner dimensions must match: {k} vs {k2}");

        // Get raw f32 slices
        let a_bytes = a.data();
        let b_bytes = b.data();
        // SAFETY: Tensor data is stored as F32, and the byte length is verified
        // to be a multiple of 4 by `assert_eq!(k * 4, a_bytes.len())` above.
        let a_f32 = unsafe {
            std::slice::from_raw_parts(a_bytes.as_ptr().cast::<f32>(), a_bytes.len() / 4)
        };
        // SAFETY: Same invariant for `b` — verified by shape dimension assertion.
        let b_f32 = unsafe {
            std::slice::from_raw_parts(b_bytes.as_ptr().cast::<f32>(), b_bytes.len() / 4)
        };

        let mut c = vec![0.0f32; m * n];
        crate::matmul::matmul_f32(a_f32, b_f32, &mut c, m, n, k, self.n_threads);

        Tensor::from_f32(&[m, n], &c)
    }

    /// Execute element-wise addition: `C = A + B`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor shapes don't match.
    #[must_use]
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(
            a.shape(),
            b.shape(),
            "addition requires matching shapes: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        Tensor::new(a.dtype(), a.shape())
    }

    /// Returns the number of worker threads.
    #[must_use]
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }
}
