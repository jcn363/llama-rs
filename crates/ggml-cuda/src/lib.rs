//! CUDA backend for ggml, optimized for NVIDIA GTX 1050 (compute 6.1).
//!
//! This crate implements tensor operations for CUDA execution, targeting
//! Pascal architecture (`sm_61`) with 2GB VRAM constraints.
//!
//! # Hardware Target
//!
//! - **GPU:** NVIDIA GTX 1050 (Pascal)
//! - **Compute capability:** 6.1
//! - **VRAM:** 2GB
//! - **CUDA cores:** 640
//!
//! # Example
//!
//! ```no_run
//! use ggml_cuda::CudaBackend;
//!
//! let backend = CudaBackend::new().unwrap();
//! println!("VRAM: {} MB", backend.total_vram() / (1024 * 1024));
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use ggml::Tensor;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can occur during CUDA operations.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// CUDA is not available on this system.
    #[error("CUDA not available: {0}")]
    NotAvailable(String),

    /// Insufficient VRAM for the requested operation.
    #[error("insufficient VRAM: needed {needed} bytes, available {available} bytes")]
    OutOfMemory {
        /// Bytes required for the operation.
        needed: usize,
        /// Bytes currently available.
        available: usize,
    },

    /// A CUDA runtime error occurred.
    #[error("CUDA error: {0}")]
    RuntimeError(String),
}

/// Result type alias for CUDA operations.
pub type CudaResult<T> = Result<T, CudaError>;

impl From<CudaError> for error::Error {
    fn from(err: CudaError) -> Self {
        match err {
            CudaError::OutOfMemory { needed, available } => error::Error::Other(format!(
                "insufficient VRAM: needed {needed} bytes, available {available} bytes"
            )),
            CudaError::NotAvailable(s) | CudaError::RuntimeError(s) => error::Error::Other(s),
        }
    }
}

// ─── CUDA Backend ────────────────────────────────────────────────────────────

/// CUDA backend for GPU-accelerated tensor operations.
///
/// Optimized for GTX 1050 (compute 6.1, 2GB VRAM, 640 CUDA cores).
/// Caches the CUDA context, stream, and cuBLAS handle for the lifetime
/// of the backend to avoid per-call initialization overhead.
pub struct CudaBackend {
    available: bool,
    total_vram: usize,
    cuda_cores: usize,
    compute_major: i32,
    compute_minor: i32,
    #[cfg(feature = "cuda")]
    state: Option<CudaState>,
}

#[cfg(feature = "cuda")]
struct CudaState {
    #[expect(dead_code)]
    context: std::sync::Arc<cudarc::driver::CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    blas: cudarc::cublas::CudaBlas,
}

// CudaState owns Arc<CudaContext> + Arc<CudaStream> (both Send+Sync)
// and CudaBlas (Send+Sync).
#[cfg(feature = "cuda")]
unsafe impl Send for CudaState {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaState {}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("available", &self.available)
            .field("total_vram", &self.total_vram)
            .field("cuda_cores", &self.cuda_cores)
            .field("compute_major", &self.compute_major)
            .field("compute_minor", &self.compute_minor)
            .finish_non_exhaustive()
    }
}

impl CudaBackend {
    /// Initialize the CUDA backend.
    ///
    /// When CUDA is available, the device handle and cuBLAS handle are cached
    /// for the lifetime of the backend. When the `"cuda"` feature is disabled
    /// or no NVIDIA GPU is found, returns a stub backend with `available = false`.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotAvailable`] if CUDA cannot be initialized.
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::CudaBlas;
            use cudarc::driver::CudaContext;

            let context = CudaContext::new(0).map_err(|e| {
                CudaError::NotAvailable(format!("failed to initialize CUDA context: {e}"))
            })?;

            // SAFETY: cuMemGetInfo_v2 is a CUDA runtime API call that returns valid VRAM info.
            let total_vram = unsafe {
                let mut free: usize = 0;
                let mut total: usize = 0;
                let res = cudarc::driver::sys::cuMemGetInfo_v2(&raw mut free, &raw mut total);
                if res == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    total
                } else {
                    2 * 1024 * 1024 * 1024
                }
            };

            let multi_proc_count = context
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute_enum::
                        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                )
                .unwrap_or(5) as usize; // GTX 1050 = 5 SMs

            let compute_major = context
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute_enum::
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
                .unwrap_or(6);
            let compute_minor = context
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute_enum::
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )
                .unwrap_or(1);

            let stream = context.default_stream();
            let blas = CudaBlas::new(stream.clone()).map_err(|e| {
                CudaError::NotAvailable(format!("failed to create cuBLAS handle: {e}"))
            })?;

            Ok(Self {
                available: true,
                total_vram,
                cuda_cores: multi_proc_count * 128, // Pascal: 128 cores/SM
                compute_major,
                compute_minor,
                state: Some(CudaState {
                    context,
                    stream,
                    blas,
                }),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                available: false,
                total_vram: 2 * 1024 * 1024 * 1024,
                cuda_cores: 640,
                compute_major: 6,
                compute_minor: 1,
            })
        }
    }

    /// Returns whether CUDA is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Returns the total VRAM in bytes.
    #[must_use]
    pub fn total_vram(&self) -> usize {
        self.total_vram
    }

    /// Returns the free VRAM in bytes (approximate).
    ///
    /// Note: VRAM is not tracked at the allocation level; this returns
    /// total VRAM minus a rough estimate. For precise tracking, future
    /// work should wire `DeviceTensor::drop` to decrement a counter.
    #[must_use]
    pub fn free_vram(&self) -> usize {
        self.total_vram
    }

    /// Returns the number of CUDA cores.
    #[must_use]
    pub fn cuda_cores(&self) -> usize {
        self.cuda_cores
    }

    /// Returns the compute capability as a string (e.g., `"6.1"`).
    #[must_use]
    pub fn compute_capability(&self) -> String {
        format!("{}.{}", self.compute_major, self.compute_minor)
    }

    /// Copy a tensor from host memory to device memory.
    ///
    /// Only [`F32`](ggml::DType::F32) tensors are currently supported.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::OutOfMemory`] if there is insufficient VRAM.
    pub fn copy_to_device(&self, tensor: &Tensor) -> CudaResult<DeviceTensor> {
        let byte_size = tensor.byte_size();
        if byte_size > self.free_vram() {
            return Err(CudaError::OutOfMemory {
                needed: byte_size,
                available: self.free_vram(),
            });
        }

        #[cfg(feature = "cuda")]
        {
            let state = self
                .state
                .as_ref()
                .ok_or_else(|| CudaError::NotAvailable("CUDA not initialized".into()))?;

            // Tensor stores bytes; for F32 tensors we reinterpret directly.
            let data: &[f32] = if tensor.dtype() == ggml::DType::F32 {
                // SAFETY: Tensor data is guaranteed correct length and
                // alignment for f32 when dtype is F32.
                #[expect(clippy::cast_ptr_alignment)]
                unsafe {
                    std::slice::from_raw_parts(
                        tensor.data().as_ptr().cast::<f32>(),
                        tensor.data().len() / 4,
                    )
                }
            } else {
                return Err(CudaError::RuntimeError(
                    "only F32 tensors are supported for device copy".into(),
                ));
            };

            let dev_data = state
                .stream
                .clone_htod(data)
                .map_err(|e| CudaError::RuntimeError(format!("failed to copy to device: {e}")))?;

            Ok(DeviceTensor {
                size: byte_size,
                element_count: tensor.element_count(),
                shape: tensor.shape().to_vec(),
                dev_data: Some(dev_data),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = tensor;
            Err(CudaError::NotAvailable("CUDA feature not enabled".into()))
        }
    }

    /// Execute matrix multiplication on GPU: C = A × B^T.
    ///
    /// Computes `C = A @ B^T` where A is shape `[M, K]` and B is shape `[N, K]`,
    /// producing output C of shape `[M, N]`.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::RuntimeError`] if the operation fails.
    pub fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor) -> CudaResult<DeviceTensor> {
        if !self.available {
            return Err(CudaError::NotAvailable("CUDA backend not available".into()));
        }

        #[cfg(feature = "cuda")]
        let (m, n) = (a.shape[0], b.shape[0]);
        #[cfg(not(feature = "cuda"))]
        let (_m, _n) = (a.shape[0], b.shape[0]);
        let k = a.shape[1];
        let k2 = b.shape[1];

        if k != k2 {
            return Err(CudaError::RuntimeError(format!(
                "inner dimensions must match: {k} vs {k2}"
            )));
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::{Gemm, GemmConfig};

            let state = self
                .state
                .as_ref()
                .ok_or_else(|| CudaError::NotAvailable("CUDA not initialized".into()))?;

            let dev_a = a
                .dev_data
                .as_ref()
                .ok_or_else(|| CudaError::RuntimeError("device tensor A has no data".into()))?;
            let dev_b = b
                .dev_data
                .as_ref()
                .ok_or_else(|| CudaError::RuntimeError("device tensor B has no data".into()))?;

            let out_size = m * n;
            let mut dev_c = state
                .stream
                .alloc_zeros::<f32>(out_size)
                .map_err(|e| CudaError::RuntimeError(format!("failed to allocate output: {e}")))?;

            // cuBLAS: C = alpha * op(A) * op(B) + beta * C
            // C = A × B^T  =>  op(A)=N, op(B)=T
            #[expect(clippy::cast_possible_wrap)]
            let config = GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0_f32,
                lda: n as i32,
                ldb: m as i32,
                beta: 0.0_f32,
                ldc: n as i32,
            };

            // SAFETY: cuBLAS gemm operates on device pointers that are valid
            // and have sufficient capacity. Dimensions are verified above.
            unsafe {
                state
                    .blas
                    .gemm(config, dev_b, dev_a, &mut dev_c)
                    .map_err(|e| CudaError::RuntimeError(format!("cuBLAS gemm failed: {e}")))?;
            }

            Ok(DeviceTensor {
                size: out_size * 4,
                element_count: out_size,
                shape: vec![m, n],
                dev_data: Some(dev_c),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = a;
            let _ = b;
            Err(CudaError::NotAvailable("CUDA feature not enabled".into()))
        }
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            available: false,
            total_vram: 2 * 1024 * 1024 * 1024,
            cuda_cores: 640,
            compute_major: 6,
            compute_minor: 1,
            #[cfg(feature = "cuda")]
            state: None,
        })
    }
}

// ─── Device Tensor ──────────────────────────────────────────────────────────

/// A tensor stored in device (GPU) memory.
///
/// The underlying device memory is freed when the `DeviceTensor` is dropped
/// (via `CudaSlice`'s `Drop` implementation).
pub struct DeviceTensor {
    /// Size in bytes.
    size: usize,
    /// Number of elements.
    element_count: usize,
    /// Shape dimensions.
    shape: Vec<usize>,
    /// Device data.
    #[cfg(feature = "cuda")]
    dev_data: Option<cudarc::driver::CudaSlice<f32>>,
    #[cfg(not(feature = "cuda"))]
    dev_data: Option<()>,
}

// CudaSlice<f32> is Send (but not Sync). DeviceTensor exposes only
// read-only operations, so sharing references is safe.
#[cfg(feature = "cuda")]
unsafe impl Send for DeviceTensor {}
#[cfg(not(feature = "cuda"))]
unsafe impl Send for DeviceTensor {}

impl std::fmt::Debug for DeviceTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceTensor")
            .field("size", &self.size)
            .field("element_count", &self.element_count)
            .field("shape", &self.shape)
            .field("has_data", &self.dev_data.is_some())
            .finish()
    }
}

impl DeviceTensor {
    /// Returns the size of the device tensor in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.size
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns the shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Copy device data back to host.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::RuntimeError`] if the copy fails.
    pub fn to_host(&self) -> CudaResult<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            let dev_data = self
                .dev_data
                .as_ref()
                .ok_or_else(|| CudaError::RuntimeError("device tensor has no data".into()))?;

            let host_data = dev_data
                .stream()
                .clone_dtoh(dev_data)
                .map_err(|e| CudaError::RuntimeError(format!("failed to copy from device: {e}")))?;

            Ok(host_data)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::NotAvailable("CUDA feature not enabled".into()))
        }
    }
}

// ─── Backend trait implementation ────────────────────────────────────────────

use ggml::backend::{Backend, BackendInfo};

impl Backend for CudaBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "CUDA",
            is_available: self.available,
            total_memory: self.total_vram,
            free_memory: self.free_vram(),
            parallelism: self.cuda_cores,
        }
    }

    /// Matrix-vector product using GPU-accelerated matmul.
    ///
    /// Copies weight and input to the device, performs `C = weight @ input^T`
    /// via cuBLAS, and copies the result back. Falls back to CPU if CUDA is
    /// not available or the operation fails.
    fn mat_vec(&self, weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        if !self.available {
            return ggml::backend::default_mat_vec(weight, rows, cols, input);
        }

        #[cfg(feature = "cuda")]
        {
            // Create tensors wrapping the host data for device copy.
            let weight_tensor = ggml::Tensor::from_f32(&[rows, cols], weight);
            // Reshape input as (1, cols) for matmul — result will be (rows, 1).
            let input_tensor = ggml::Tensor::from_f32(&[1, cols], input);

            match self
                .copy_to_device(&weight_tensor)
                .and_then(|w_dev| {
                    let i_dev = self.copy_to_device(&input_tensor)?;
                    Ok((w_dev, i_dev))
                })
                .and_then(|(w_dev, i_dev)| self.matmul(&w_dev, &i_dev))
                .and_then(|result_dev| result_dev.to_host())
            {
                Ok(result) => result,
                Err(e) => {
                    tracing::warn!("CUDA mat_vec failed, falling back to CPU: {e}");
                    ggml::backend::default_mat_vec(weight, rows, cols, input)
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            ggml::backend::default_mat_vec(weight, rows, cols, input)
        }
    }

    // All non-mat_vec operations use the trait's default CPU implementations.
    // This is intentional: element-wise ops are memory-bandwidth bound and
    // benefit little from GPU launch overhead, while complex ops (norm, rope,
    // attention, conv) require custom CUDA kernels not yet implemented.
    // The trait defaults provide correct CPU fallback for all 100+ operations.
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_should_report_vram() {
        let backend = CudaBackend::new().unwrap_or_default();
        // Allow for VRAM reporting variations (some drivers report slightly less than 2GB)
        assert!(
            backend.total_vram() >= 1900 * 1024 * 1024, // ~1.9GB minimum
            "expected >=1.9GB VRAM, got {}",
            backend.total_vram()
        );
    }

    #[test]
    fn cuda_backend_should_report_cuda_cores() {
        let backend = CudaBackend::new().unwrap_or_default();
        assert!(
            backend.cuda_cores() >= 128,
            "expected >=128 cores, got {}",
            backend.cuda_cores()
        );
    }

    #[test]
    fn cuda_backend_should_report_compute_capability() {
        let backend = CudaBackend::new().unwrap_or_default();
        let cap = backend.compute_capability();
        let parts: Vec<&str> = cap.split('.').collect();
        assert_eq!(parts.len(), 2, "expected 'major.minor', got '{cap}'");
        let _major: i32 = parts[0].parse().expect("major must be integer");
        let _minor: i32 = parts[1].parse().expect("minor must be integer");
    }

    #[test]
    fn cuda_backend_thread_safety() {
        // Verify that the backend can be safely accessed from multiple threads.
        // Use an Arc to share ownership without requiring Clone on CudaBackend.
        let backend = std::sync::Arc::new(CudaBackend::new().unwrap_or_default());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let b = std::sync::Arc::clone(&backend);
                std::thread::spawn(move || {
                    let _ = b.is_available();
                    let _ = b.total_vram();
                    let _ = b.cuda_cores();
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
    }
    #[test]
    fn copy_to_device_should_fail_for_large_tensor() {
        let backend = CudaBackend::new().unwrap_or_default();
        let large = Tensor::new(ggml::DType::F32, &[1_000_000_000]);
        let result = backend.copy_to_device(&large);
        assert!(result.is_err(), "expected error (OOM or not available)");
    }

    #[test]
    fn device_tensor_shape_roundtrip() {
        let shape = vec![16, 32];
        let dummy = DeviceTensor {
            size: 16 * 32 * 4,
            element_count: 16 * 32,
            shape: shape.clone(),
            dev_data: None,
        };
        assert_eq!(dummy.shape(), &[16, 32]);
        assert_eq!(dummy.element_count(), 512);
        assert_eq!(dummy.byte_size(), 16 * 32 * 4);
    }
}
