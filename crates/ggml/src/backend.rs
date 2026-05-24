//! Hardware backend trait for tensor operations.
//!
//! Defines the [`Backend`] trait that all hardware backends (CPU, CUDA, etc.)
//! must implement, and the [`BackendInfo`] struct for reporting capabilities.
//!
//! This is the plugin interface: adding a new hardware backend means creating a
//! new crate that implements [`Backend`] and registering it with the registry.

/// Information about a hardware backend's capabilities.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Human-readable name (e.g. `"CPU"`, `"CUDA"`).
    pub name: &'static str,
    /// Whether the backend is available and ready for use.
    pub is_available: bool,
    /// Total device memory in bytes (0 if not applicable, e.g. CPU).
    pub total_memory: usize,
    /// Free device memory in bytes.
    pub free_memory: usize,
    /// Degree of parallelism (threads, SM count, CUDA cores, etc.).
    pub parallelism: usize,
}

/// A hardware backend capable of executing tensor operations.
///
/// This is **the** plugin interface for supporting different hardware.
/// Each backend implements the core math operations needed by the
/// inference engine. The trait is object-safe so backends can be
/// used polymorphically via `Arc<dyn Backend>`.
///
/// # Extending
///
/// To add a new hardware backend:
///
/// 1. Create a struct for your backend (e.g. `VulkanBackend`).
/// 2. Implement `Backend` for it.
/// 3. Register it with [`BackendRegistry`] so it participates in auto-selection.
///
/// # Notes on object-safety
///
/// The trait avoids generic parameters and uses only `&[f32]` / `Vec<f32>`
/// signatures so it remains object-safe. Parallelism is handled internally
/// by each backend.
pub trait Backend: Send + Sync {
    /// Returns information about this backend.
    fn info(&self) -> BackendInfo;

    /// Matrix-vector product: `y = weight @ input`
    ///
    /// `weight` has shape `(rows, cols)` in row-major order.
    /// `input` has length `cols`.
    /// Returns a vector of length `rows`.
    fn mat_vec(&self, weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32>;

    /// Element-wise addition: `c[i] = a[i] + b[i]`
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// Element-wise multiplication: `c[i] = a[i] * b[i]`
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
}

// ─── Default (CPU fallback) implementations ──────────────────────────────────

/// Default matrix-vector product implementation using sequential iteration.
///
/// This is a portable fallback used by backends that don't accelerate
/// matrix-vector multiplication (or for small matrices where dispatch
/// overhead isn't worth it).
pub fn default_mat_vec(weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let start = r * cols;
            let row = &weight[start..start + cols];
            row.iter().zip(input.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Default element-wise addition.
pub fn default_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Default element-wise multiplication.
pub fn default_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}
