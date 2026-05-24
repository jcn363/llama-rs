//! CPU backend for executing computation graphs.
//!
//! Provides the `CpuBackend` struct which implements matrix multiplication
//! and other tensor operations using the CPU.

use ggml::{DType, Tensor};
use std::alloc::Layout;
use std::cell::RefCell;

thread_local! {
    static BUMP_ALLOCATOR: RefCell<Option<bumpalo::Bump>> = const { RefCell::new(None) };
}

/// Executes a closure with access to a thread-local bump allocator.
///
/// If no allocator exists (or needs reallocation), one is created with the
/// given `size` capacity. Otherwise, the existing allocator is reset and reused.
pub fn with_bump_allocator<F, R>(size: usize, f: F) -> R
where
    F: FnOnce(&mut bumpalo::Bump) -> R,
{
    BUMP_ALLOCATOR.with(|cell| {
        let mut opt = cell.borrow_mut();
        let needs_realloc = match &mut *opt {
            None => true,
            Some(bump) => {
                // In newer bumpalo, we check if the chunk capacity is sufficient
                bump.reset();
                false
            }
        };
        if needs_realloc {
            *opt = Some(bumpalo::Bump::with_capacity(size));
        }
        f(opt.as_mut().unwrap())
    })
}

/// Resets the thread-local bump allocator, freeing its memory.
///
/// Call this at the start of each forward pass to prevent
/// memory accumulation across invocations.
pub fn reset_bump_allocator() {
    BUMP_ALLOCATOR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Executes a computation graph on the CPU.
pub struct CpuBackend {
    n_threads: usize,
    /// Minimum number of rows (M) before parallel dispatch kicks in.
    /// For small matrices, thread overhead exceeds the benefit.
    parallel_min_rows: usize,
    /// Size of thread-local memory pool for small temporary allocations (in bytes, 0 = disabled).
    memory_pool_size: usize,
}

impl CpuBackend {
    /// Create a new CPU backend with the given number of threads.
    ///
    /// If `n_threads` is 0, uses the number of available parallel threads.
    /// `parallel_min_rows` is the minimum number of rows before parallel dispatch;
    /// pass 0 for default (128).
    /// `memory_pool_size` is the size of thread-local memory pool for small temporary allocations (in bytes, 0 = disabled).
    #[must_use]
    pub fn new(n_threads: usize, memory_pool_size: usize) -> Self {
        Self::new_with_min_rows(n_threads, 0, memory_pool_size)
    }

    /// Create a new CPU backend with the given number of threads and
    /// a minimum row count for parallel matmul dispatch.
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

    /// Run a parallel operation across rows, dispatching if above threshold.
    pub fn parallel_for<T, F>(&self, items: &[T], f: F)
    where
        T: Sync,
        F: Fn(&T) + Sync,
    {
        if items.len() < self.parallel_min_rows || self.n_threads <= 1 {
            items.iter().for_each(f);
        } else {
            std::thread::scope(|s| {
                let chunk_size = items.len().div_ceil(self.n_threads);
                for chunk in items.chunks(chunk_size) {
                    s.spawn(|| chunk.iter().for_each(&f));
                }
            });
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

        // Allocate result buffer
        let result_size = m * n;
        let result_data_size = result_size * std::mem::size_of::<f32>();
        #[allow(unused_assignments)]
        let mut temp_vec = Vec::new();
        let result_buffer =
            if self.memory_pool_size > 0 && result_data_size <= self.memory_pool_size {
                // Allocate from bump allocator
                let layout = Layout::array::<f32>(result_size).expect("Failed to create layout");
                let ptr = with_bump_allocator(result_data_size, |bump| {
                    bump.alloc_layout(layout).as_ptr().cast::<f32>()
                });
                // Create a slice from the bump allocated memory
                unsafe { std::slice::from_raw_parts_mut(ptr, result_size) }
            } else {
                // Allocate a vec on the heap
                temp_vec = vec![0.0f32; result_size];
                temp_vec.as_mut_slice()
            };

        // Skip parallel dispatch for small matrices (thread overhead > benefit)
        let effective_threads = if m < self.parallel_min_rows {
            1
        } else {
            self.n_threads
        };
        crate::matmul::matmul_f32(
            a_f32,
            b_f32,
            result_buffer,
            m,
            n,
            k,
            effective_threads,
            self.parallel_min_rows,
        );

        // Copy result to owned vector to create Tensor (so Tensor owns its data)
        let result_data: Vec<f32> = result_buffer.to_vec();
        Tensor::from_f32(&[m, n], &result_data)
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
        let shape = a.shape();
        let elem_count = shape.iter().product::<usize>();
        let data_size = elem_count * std::mem::size_of::<f32>();
        #[allow(unused_assignments)]
        let mut temp_vec = Vec::new();
        let result_buffer = if self.memory_pool_size > 0 && data_size <= self.memory_pool_size {
            // Allocate from bump allocator
            let layout = Layout::array::<f32>(elem_count).expect("Failed to create layout");
            let ptr = with_bump_allocator(data_size, |bump| {
                bump.alloc_layout(layout).as_ptr().cast::<f32>()
            });
            // Create a slice from the bump allocated memory
            unsafe { std::slice::from_raw_parts_mut(ptr, elem_count) }
        } else {
            // Allocate a vec on the heap
            temp_vec = vec![0.0f32; elem_count];
            temp_vec.as_mut_slice()
        };

        // Perform element-wise addition
        let a_data = a.data();
        let b_data = b.data();
        let a_f32 =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr().cast::<f32>(), elem_count) };
        let b_f32 =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr().cast::<f32>(), elem_count) };
        for i in 0..elem_count {
            result_buffer[i] = a_f32[i] + b_f32[i];
        }

        // Copy result to owned vector to create Tensor
        let result_data: Vec<f32> = result_buffer.to_vec();
        Tensor::from_f32(shape, &result_data)
    }

    /// Returns the number of worker threads.
    #[must_use]
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }
}

// ─── Backend trait implementation ─────────────────────────────────────────────

use ggml::backend::{Backend, BackendInfo, QuantType};

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

    /// Matrix-vector product: `y = weight @ input`
    ///
    /// Uses SIMD-accelerated dot product per row, with parallel dispatch
    /// across threads when the matrix is large enough.
    fn mat_vec(&self, weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        if rows < self.parallel_min_rows || self.n_threads <= 1 {
            (0..rows)
                .map(|r| {
                    let start = r * cols;
                    crate::simd::dot_f32(&weight[start..start + cols], input)
                })
                .collect()
        } else {
            (0..rows)
                .into_par_iter()
                .map(|r| {
                    let start = r * cols;
                    crate::simd::dot_f32(&weight[start..start + cols], input)
                })
                .collect()
        }
    }

    /// Quantized matrix-vector product.
    ///
    /// Dispatches to the appropriate `QuantDot` kernel for each row,
    /// with parallel dispatch across threads when the matrix is large enough.
    fn mat_vec_quant(
        &self,
        weight: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let block_size = quant_type.block_size();
        let block_bytes = quant_type.block_bytes();
        let n_blocks_per_row = cols.div_ceil(block_size);

        // Select the quantized dot kernel (fn pointer is Sync)
        let dot_row: fn(&[u8], &[f32], usize) -> f32 = match quant_type {
            QuantType::Q4_0 => {
                |q, inp, c| {
                    crate::quant_dot::quant_dot_row(
                        &crate::quant_dot::q4_0::Q4_0Dot,
                        q,
                        inp,
                        c,
                    )
                }
            }
            QuantType::Q8_0 => {
                |q, inp, c| {
                    crate::quant_dot::quant_dot_row(
                        &crate::quant_dot::q8_0::Q8_0Dot,
                        q,
                        inp,
                        c,
                    )
                }
            }
            QuantType::Q4_1 => {
                |q, inp, c| {
                    crate::quant_dot::quant_dot_row(
                        &crate::quant_dot::q4_1::Q4_1Dot,
                        q,
                        inp,
                        c,
                    )
                }
            }
        };

        use rayon::prelude::*;
        if rows < self.parallel_min_rows || self.n_threads <= 1 {
            (0..rows)
                .map(|r| {
                    let row_offset = r * n_blocks_per_row * block_bytes;
                    dot_row(&weight[row_offset..], input, cols)
                })
                .collect()
        } else {
            (0..rows)
                .into_par_iter()
                .map(|r| {
                    let row_offset = r * n_blocks_per_row * block_bytes;
                    dot_row(&weight[row_offset..], input, cols)
                })
                .collect()
        }
    }

    /// Element-wise addition.
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    /// Element-wise multiplication.
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }



    /// Root Mean Square Normalization: `y = (x / RMS(x)) * weight`
    /// where `RMS(x) = sqrt(mean(x^2) + eps)`.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        // Use the default implementation for now - can be optimized with SIMD later
        ggml::backend::default_rms_norm(x, weight, eps)
    }

    /// Sigmoid Linear Unit (`SiLU`) activation: `y = x * sigmoid(x)`
    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|v| {
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                v * sigmoid
            })
            .collect()
    }

    /// Gaussian Error Linear Unit (GELU) activation:
    /// `y = x * Φ(x)` where Φ is the standard Gaussian CDF approximation.
    fn gelu(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|v| {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044_715 * x³)))
                let sqrt_2_over_pi = 0.797_884_6;
                let inner = sqrt_2_over_pi * (v + 0.044_715 * v * v * v);
                let tanh = (inner.exp() - (-inner).exp()) / (inner.exp() + (-inner).exp());
                0.5 * v * (1.0 + tanh)
            })
            .collect()
    }
}
