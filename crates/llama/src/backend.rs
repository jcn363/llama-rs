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
/// 1. `BackendType::Auto` — try CUDA, fall back to CPU.
/// 2. `BackendType::Cuda` — use CUDA, panic if unavailable.
/// 3. `BackendType::Cpu` — use CPU unconditionally.
///
/// The returned [`Backend`] trait object can be used for all tensor
/// operations in the inference pipeline.
///
/// # Panics
///
/// Panics if the requested backend cannot be created.
#[must_use]
pub fn create_backend(config: &ModelConfig) -> Arc<dyn Backend> {
    // Attempt CUDA for Auto or Cuda modes (only when feature is enabled).
    #[cfg(feature = "cuda")]
    if config.backend_type != BackendType::Cpu {
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
                if config.backend_type == BackendType::Cuda {
                    panic!("CUDA backend explicitly requested but unavailable: {e}");
                }
                tracing::warn!("CUDA unavailable, falling back to CPU: {e}");
            }
        }
    }

    // Fall back to CPU.
    let cpu = CpuBackend::new_with_min_rows(
        config.n_threads,
        config.parallel_min_rows,
        config.memory_pool_size,
    );
    tracing::info!("Using CPU backend ({} threads)", cpu.n_threads());
    Arc::new(cpu)
}
