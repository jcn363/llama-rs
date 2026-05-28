//! Core types shared across the llama-rs workspace.
//!
//! This crate provides foundational types that are used by multiple
//! crates in the workspace, extracted to avoid duplication and circular
//! dependencies.

use memmap2::Mmap;
use std::sync::Arc;
use thiserror::Error;

/// Error types for core operations.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("tensor data extends beyond mmap (need {need}, have {have})")]
    MmapBoundsExceeded { need: usize, have: usize },
}

// ... rest of file ...

/// Information about a single tensor in a GGUF file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape (dimensions in reverse order from file, matching ggml convention).
    pub shape: Vec<i64>,
    /// Data type.
    pub dtype: GgmlType,
    /// Offset into the tensor data blob.
    pub offset: u64,
}

/// Memory‑mapped tensor reference — holds a shared mmap plus offset/size.
#[derive(Debug, Clone)]
pub struct MmapTensor {
    /// Shared reference to the memory‑mapped file.
    pub mmap: Arc<Mmap>,
    /// Byte offset within the mmap where this tensor's data starts.
    pub offset: usize,
    /// Size of the tensor data in bytes.
    pub size: usize,
}

impl MmapTensor {
    /// Create a new memory‑mapped tensor reference.
    #[must_use]
    pub fn new(mmap: Arc<Mmap>, offset: usize, size: usize) -> Self {
        Self { mmap, offset, size }
    }

    /// Get a slice of the raw tensor data from the mmap.
    pub fn as_slice(&self) -> Result<&[u8], CoreError> {
        let end = self.offset + self.size;
        if end > self.mmap.len() {
            return Err(CoreError::MmapBoundsExceeded {
                need: end,
                have: self.mmap.len(),
            });
        }
        Ok(&self.mmap[self.offset..end])
    }
}

pub mod backend;
pub mod dtype;
pub mod ggml_type;

pub use backend::{BackendInfo, QuantType};
pub use dtype::DType;
pub use ggml_type::GgmlType;
