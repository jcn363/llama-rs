//! Hardware backend factory and registry.
//!
//! Provides [`create_backend`] which auto-selects the best available
//! hardware backend based on the configuration and runtime environment.
//!
//! # Extending
//!
//! To add a new hardware backend:
//!
//! 1. Implement [`ggml::backend::Backend`] for your backend struct.
//! 2. Add a variant to the auto-selection logic in [`create_backend`].
//!
//! The registry is intentionally simple — it's a linear-priority fallback
//! chain rather than a dynamic plugin loader, which keeps compile times
//! fast and avoids runtime linking complexity.

use std::sync::Arc;

use ggml::backend::Backend;
use ggml_cpu::CpuBackend;

use crate::ModelConfig;

/// The backend type to use for inference.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Auto-select the best available backend (CUDA > CPU).
    #[default]
    Auto,
    /// Use CPU backend only.
    Cpu,
    /// Use CUDA backend if available.
    Cuda,
}

/// Create a hardware backend based on the configuration.
///
/// Selection priority:
/// 1. If `BackendType::Cuda` or `Auto` with `use_cuda = true`, try CUDA.
/// 2. Fall back to CPU.
///
/// The returned [`Backend`] trait object can be used for all tensor
/// operations in the inference pipeline.
///
/// # Panics
///
/// Panics if CPU backend creation fails (should never happen).
#[must_use]
pub fn create_backend(config: &ModelConfig) -> Arc<dyn Backend> {
    // Determine whether to attempt CUDA.
    let try_cuda = match config.backend_type {
        BackendType::Cuda => true,
        BackendType::Auto | BackendType::Cpu => config.use_cuda,
    };

    // Attempt CUDA backend if requested and feature-enabled.
    #[cfg(feature = "cuda")]
    if try_cuda {
        match ggml_cuda::CudaBackend::new() {
            Ok(cuda) => {
                let info = cuda.info();
                tracing::info!(
                    "Using CUDA backend ({} cores, {} MB VRAM)",
                    info.parallelism,
                    info.total_memory / (1024 * 1024),
                );
                return Arc::new(cuda);
            }
            Err(e) => {
                tracing::warn!("CUDA requested but unavailable: {e}");
            }
        }
    }

    // Suppress unused-variable warning when cuda feature is disabled.
    #[cfg(not(feature = "cuda"))]
    let _ = try_cuda;

    // Fall back to CPU.
    let cpu = CpuBackend::new_with_min_rows(
        config.n_threads,
        config.parallel_min_rows,
        config.memory_pool_size,
    );
    tracing::info!("Using CPU backend ({} threads)", cpu.n_threads());
    Arc::new(cpu)
}
