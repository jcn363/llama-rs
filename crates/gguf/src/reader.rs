//! GgufReader convenience methods for accessing metadata and tensor data.
//!
//! This module contains the method implementations for [`GgufReader`] that
//! were previously part of the monolithic `lib.rs`.  The struct definition
//! and core file-loading logic live in `lib.rs` and `loader.rs` respectively.

use std::sync::Arc;

use super::{
    GgmlType, GgufError, GgufReader, GgufResult, GgufType, GgufValue, MmapTensor, TensorInfo,
};

impl GgufReader {
    // ─── Metadata convenience getters ─────────────────────────────────────────

    /// Get a usize metadata value by key (expects u32 stored).
    ///
    /// # Errors
    /// Returns a `GgufError::DecodeError` if the key is missing or has an
    /// unexpected type.
    pub fn get_usize(&self, key: &str) -> GgufResult<usize> {
        match self.get_kv(key) {
            Some(GgufValue::U32(v)) => Ok(*v as usize),
            Some(GgufValue::U64(v)) => {
                Ok(usize::try_from(*v).map_err(|e| GgufError::DecodeError(e.to_string()))?)
            }
            Some(other) => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' has unexpected type: {other:?}"
            ))),
            None => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' not found"
            ))),
        }
    }

    /// Get a usize metadata value, trying multiple keys in order.
    /// Returns the first key that exists and has a valid type.
    ///
    /// # Errors
    /// Returns a `GgufError::DecodeError` if none of the keys are found or
    /// all have unexpected types.
    pub fn get_usize_any(&self, keys: &[&str]) -> GgufResult<usize> {
        for &key in keys {
            if let Some(val) = self.get_kv(key) {
                return match val {
                    GgufValue::U32(v) => Ok(*v as usize),
                    GgufValue::U64(v) => {
                        Ok(usize::try_from(*v)
                            .map_err(|e| GgufError::DecodeError(e.to_string()))?)
                    }
                    other => Err(GgufError::DecodeError(format!(
                        "metadata key '{key}' has unexpected type: {other:?}"
                    ))),
                };
            }
        }
        Err(GgufError::DecodeError(format!(
            "none of the metadata keys found: {:?}",
            keys
        )))
    }

    /// Get a string metadata value by key.
    ///
    /// # Errors
    /// Returns a `GgufError::DecodeError` if the key is missing or has an
    /// unexpected type.
    pub fn get_string(&self, key: &str) -> GgufResult<String> {
        match self.get_kv(key) {
            Some(GgufValue::Str(s)) => Ok(s.clone()),
            Some(other) => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' has unexpected type: {other:?}"
            ))),
            None => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' not found"
            ))),
        }
    }

    /// Get an array of strings metadata value by key.
    ///
    /// # Errors
    /// Returns a `GgufError::DecodeError` if the key is missing, has an
    /// unexpected type, or the array contains non-string elements.
    pub fn get_string_array(&self, key: &str) -> GgufResult<Vec<String>> {
        match self.get_kv(key) {
            Some(GgufValue::Array { elem_type, data }) => {
                if *elem_type != GgufType::String {
                    return Err(GgufError::DecodeError(format!(
                        "metadata key '{key}' expected string array, got {elem_type:?}"
                    )));
                }
                let mut result = Vec::with_capacity(data.len());
                for val in data {
                    if let GgufValue::Str(s) = val {
                        result.push(s.clone());
                    } else {
                        return Err(GgufError::DecodeError(format!(
                            "metadata key '{key}' contains non-string element"
                        )));
                    }
                }
                Ok(result)
            }
            Some(other) => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' has unexpected type: {other:?}"
            ))),
            None => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' not found"
            ))),
        }
    }

    /// Get an array of f32 metadata value by key.
    ///
    /// # Errors
    /// Returns a `GgufError::DecodeError` if the key is missing, has an
    /// unexpected type, or the array contains non-f32 elements.
    pub fn get_f32_array(&self, key: &str) -> GgufResult<Vec<f32>> {
        match self.get_kv(key) {
            Some(GgufValue::Array { elem_type, data }) => {
                if *elem_type != GgufType::Float32 {
                    return Err(GgufError::DecodeError(format!(
                        "metadata key '{key}' expected f32 array, got {elem_type:?}"
                    )));
                }
                let mut result = Vec::with_capacity(data.len());
                for val in data {
                    if let GgufValue::F32(v) = val {
                        result.push(*v);
                    } else {
                        return Err(GgufError::DecodeError(format!(
                            "metadata key '{key}' contains non-f32 element"
                        )));
                    }
                }
                Ok(result)
            }
            Some(other) => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' has unexpected type: {other:?}"
            ))),
            None => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' not found"
            ))),
        }
    }

    /// Get an array of i32 metadata value by key.
    ///
    /// # Errors
    /// Returns a `GgufError::DecodeError` if the key is missing, has an
    /// unexpected type, or the array contains non-i32 elements.
    pub fn get_i32_array(&self, key: &str) -> GgufResult<Vec<i32>> {
        match self.get_kv(key) {
            Some(GgufValue::Array { elem_type, data }) => {
                if *elem_type != GgufType::Int32 {
                    return Err(GgufError::DecodeError(format!(
                        "metadata key '{key}' expected i32 array, got {elem_type:?}"
                    )));
                }
                let mut result = Vec::with_capacity(data.len());
                for val in data {
                    if let GgufValue::I32(v) = val {
                        result.push(*v);
                    } else {
                        return Err(GgufError::DecodeError(format!(
                            "metadata key '{key}' contains non-i32 element"
                        )));
                    }
                }
                Ok(result)
            }
            Some(other) => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' has unexpected type: {other:?}"
            ))),
            None => Err(GgufError::DecodeError(format!(
                "metadata key '{key}' not found"
            ))),
        }
    }

    // ─── Tensor data access ───────────────────────────────────────────────────

    /// Load raw tensor bytes for a given `TensorInfo`.
    ///
    /// # Errors
    /// Returns [`GgufError`] if the tensor data cannot be read.
    pub fn load_tensor_raw(&self, info: &TensorInfo) -> GgufResult<&[u8]> {
        self.read_tensor_data(info)
    }

    /// Calculate the byte size of a tensor from its shape and dtype.
    pub fn tensor_byte_size(&self, info: &TensorInfo) -> GgufResult<usize> {
        tensor_byte_size_for_type(info.dtype, &info.shape)
    }

    /// Create a memory-mapped tensor reference for lazy loading.
    /// The tensor data is accessed from the shared mmap on demand.
    ///
    /// # Errors
    /// Returns [`GgufError`] if the tensor size cannot be calculated.
    pub fn mmap_tensor(
        &self,
        info: &TensorInfo,
        mmap: Arc<memmap2::Mmap>,
    ) -> GgufResult<MmapTensor> {
        let byte_size = self.tensor_byte_size(info)?;
        let offset = self.data_offset + info.offset as usize;
        Ok(MmapTensor::new(mmap, offset, byte_size))
    }

    /// Read tensor data bytes from the memory-mapped file.
    ///
    /// The data is located at `data_offset + tensor.offset`.
    ///
    /// # Errors
    /// Returns an error if the offset is out of bounds.
    pub fn read_tensor_data(&self, tensor: &TensorInfo) -> GgufResult<&[u8]> {
        let start = self.data_offset + tensor.offset as usize;
        if start > self.data.len() {
            return Err(GgufError::DecodeError(format!(
                "tensor {} offset {} out of bounds (file size {})",
                tensor.name,
                tensor.offset,
                self.data.len()
            )));
        }

        let byte_size = tensor_byte_size_for_type(tensor.dtype, &tensor.shape)?;

        let end = start + byte_size;
        if end > self.data.len() {
            return Err(GgufError::DecodeError(format!(
                "tensor {} data extends beyond file (need {}, have {})",
                tensor.name,
                end,
                self.data.len()
            )));
        }

        Ok(&self.data[start..end])
    }

    // ─── Accessors ────────────────────────────────────────────────────────────

    /// Get a shared reference to the memory-mapped file.
    #[must_use]
    pub fn mmap(&self) -> &memmap2::Mmap {
        &self.data
    }

    /// Get a cloneable Arc reference to the memory-mapped file for sharing.
    #[must_use]
    pub fn mmap_arc(&self) -> &Arc<memmap2::Mmap> {
        &self.data
    }

    /// Get the byte offset where tensor data begins in the file.
    #[must_use]
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Returns the GGUF version.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> i64 {
        self.tensor_count
    }

    /// Returns the number of metadata KV pairs.
    #[must_use]
    pub fn metadata_count(&self) -> i64 {
        self.metadata_count
    }

    /// Returns the alignment used for tensor data.
    #[must_use]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Retrieve the first metadata value that matches any of the provided keys.
    ///
    /// Returns `Some(&GgufValue)` for the first key found, or `None` if none of the
    /// keys exist in the file.
    ///
    /// # Example
    /// ```ignore
    /// // Suppose the GGUF file may contain either "general.name" or "model.name"
    /// let keys = ["general.name", "model.name"];
    /// // In real code, obtain a `GgufReader` instance and call:
    /// // let value = reader.get_kv_any(&keys);
    /// ```
    #[must_use]
    pub fn get_kv_any<'a>(&'a self, keys: &[&str]) -> Option<&'a GgufValue> {
        for &key in keys {
            if let Some(val) = self.get_kv(key) {
                return Some(val);
            }
        }
        None
    }

    /// Get a metadata value by key.
    ///
    /// Returns `None` if the key is not found.
    #[must_use]
    pub fn get_kv(&self, key: &str) -> Option<&GgufValue> {
        self.kv_pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Get a metadata key-value pair by index.
    ///
    /// # Errors
    /// Returns an error if the index is out of range.
    pub fn kv_pair(&self, index: usize) -> GgufResult<(&str, &GgufValue)> {
        if index >= self.kv_pairs.len() {
            return Err(GgufError::KvIndexOutOfRange(index, self.kv_pairs.len()));
        }
        let (ref k, ref v) = self.kv_pairs[index];
        Ok((k, v))
    }

    /// Get tensor info by index.
    ///
    /// # Errors
    /// Returns an error if the index is out of range.
    pub fn tensor_info(&self, index: usize) -> GgufResult<&TensorInfo> {
        if index >= self.tensors.len() {
            return Err(GgufError::TensorIndexOutOfRange(index, self.tensors.len()));
        }
        Ok(&self.tensors[index])
    }

    /// Find a tensor by name.
    ///
    /// Returns `None` if not found.
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Returns a slice of all tensor info entries.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Returns a slice of all KV pairs.
    #[must_use]
    pub fn kv_pairs(&self) -> &[(String, GgufValue)] {
        &self.kv_pairs
    }

    /// Get a reference to the raw mmap data.
    #[must_use]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Compute the byte size of a tensor from its dtype and shape.
/// Shared by `tensor_byte_size()` and `read_tensor_data()` to avoid DRY.
fn tensor_byte_size_for_type(dtype: GgmlType, shape: &[i64]) -> GgufResult<usize> {
    let element_count: usize = shape.iter().map(|&d| d as usize).product();
    let byte_size = match dtype {
        GgmlType::F32 | GgmlType::I32 => element_count * 4,
        GgmlType::F16 | GgmlType::I16 | GgmlType::Bf16 => element_count * 2,
        GgmlType::F64 | GgmlType::I64 => element_count * 8,
        GgmlType::I8 | GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Q8_K => element_count,
        GgmlType::Q4_0 | GgmlType::Q4_1 => element_count / 2,
        GgmlType::Q5_0 | GgmlType::Q5_1 => (element_count / 2) + (element_count / 32) * 2,
        GgmlType::Q2_K | GgmlType::Q3_K => {
            element_count / 4 + element_count / 64 + element_count / 64
        }
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
            element_count / 2 + element_count / 64 + element_count / 64
        }
        // IQ (importance-matrix) quantization types
        GgmlType::Iq1S => element_count / 256 * 50,
        GgmlType::Iq1M => element_count / 256 * 56,
        GgmlType::Iq2S => element_count / 256 * 82,
        GgmlType::Iq2Xxs => element_count / 256 * 66,
        GgmlType::Iq2Xs => element_count / 256 * 74,
        GgmlType::Iq3S => element_count / 256 * 110,
        GgmlType::Iq3Xxs => element_count / 256 * 98,
        GgmlType::Iq3Xs => element_count / 256 * 98,
        GgmlType::Iq3M => element_count / 256 * 112,
        GgmlType::Iq4Nl => element_count / 32 * 18,
        GgmlType::Iq4Xs => element_count / 256 * 136,
        _ => {
            return Err(GgufError::DecodeError(format!(
                "unsupported dtype for size calculation: {dtype:?}"
            )));
        }
    };
    Ok(byte_size)
}
