#![allow(dead_code, unused_imports, unused_variables)]
use memmap2::Mmap;
use std::sync::Arc;

/// Align `value` up to the next multiple of `align` (must be a power of two).
#[inline]
pub const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

/// Top‑level GGUF reader exposing the parsed file.
#[derive(Debug)]
pub struct GgufReader {
    data: Arc<Mmap>,
    version: u32,
    tensor_count: i64,
    metadata_count: i64,
    kv_pairs: Vec<(String, GgufValue)>,
    tensors: Vec<TensorInfo>,
    alignment: usize,
    data_offset: usize,
}

mod constants;
mod cursor;
mod dequant;
mod errors;
mod imatrix;
mod loader;
mod reader;
mod tensor;
mod types;
mod value;

pub use constants::*;
pub use errors::{GgufError, GgufResult};
pub use tensor::{MmapTensor, TensorInfo, mmap_tensor_dequantize};
pub use types::{GgmlType, GgufType};
pub use value::GgufValue;
