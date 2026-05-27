#![allow(dead_code, unused_imports, unused_variables)]
use std::sync::Arc;
use memmap2::Mmap;

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
mod errors;
mod types;
mod value;
mod tensor;
mod reader;
mod loader;
mod imatrix;
mod dequant;
mod cursor;

pub use constants::*;
pub use errors::{GgufError, GgufResult};
pub use types::{GgufType, GgmlType};
pub use value::GgufValue;
pub use tensor::{TensorInfo, MmapTensor, mmap_tensor_dequantize};
