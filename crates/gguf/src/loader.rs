//! Loader utilities for [`GgufReader`].
//!
//! This module provides the public entry points for opening a GGUF file
//! either from a path (`from_file`) or from an existing memory‑mapped
//! region (`from_mmap`).  The implementation was previously part of the
//! monolithic `lib.rs` file; extracting it improves readability and makes
//! the core reader logic easier to discover.

use std::path::Path;
use std::sync::Arc;

use super::{
    GgufError,
    GgufResult,
    GgufReader,
    GgufType,
    GgufValue,
    GgmlType,
    TensorInfo,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUF_DEFAULT_ALIGNMENT,
    align_up,
};
use crate::cursor::CursorReader;

impl GgufReader {
    /// Open a GGUF file from the given path.
    ///
    /// # Errors
    /// Returns [`GgufError`] if the file cannot be opened or is not a valid
    /// GGUF file.
    pub fn from_file(path: impl AsRef<Path>) -> GgufResult<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_mmap(Arc::new(mmap))
    }

    /// Parse a GGUF file from a memory‑mapped region.
    ///
    /// # Errors
    /// Returns [`GgufError`] if the data is not a valid GGUF file.
    pub fn from_mmap(mmap: Arc<memmap2::Mmap>) -> GgufResult<Self> {
        let mut reader = CursorReader::new(&mmap);

        // 1. Magic
        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }

        // 2. Version
        let version = reader.read_u32()?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // 3. Tensor count
        let tensor_count = reader.read_i64()?;

        // 4. KV pair count
        let metadata_count = reader.read_i64()?;

        // 5. KV pairs
        let mut kv_pairs = Vec::with_capacity(metadata_count as usize);
        for _ in 0..metadata_count {
            let key = reader.read_string()?;
            let type_raw = reader.read_i32()?;
            let gguf_type = GgufType::from_i32(type_raw)?;
            let value = reader.read_value(gguf_type)?;
            kv_pairs.push((key, value));
        }

        // Determine alignment from metadata
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        for (key, value) in &kv_pairs {
            if *key == "general.alignment" {
                if let GgufValue::U32(v) = value {
                    alignment = *v as usize;
                }
                break;
            }
        }

        // 6. Tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(reader.read_i64()?);
            }
            let dtype_raw = reader.read_i32()?;
            let dtype = GgmlType::from_i32(dtype_raw)?;
            let offset = reader.read_u64()?;
            tensors.push(TensorInfo {
                name,
                n_dims,
                shape,
                dtype,
                offset,
            });
        }

        // 7. Data offset (current position, aligned)
        let data_offset = reader.position();
        let aligned_offset = align_up(data_offset, alignment);

        Ok(Self {
            data: mmap,
            version,
            tensor_count,
            metadata_count,
            kv_pairs,
            tensors,
            alignment,
            data_offset: aligned_offset,
        })
    }
}

