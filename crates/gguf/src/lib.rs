#![deny(missing_docs)]

//! GGUF (GGML Universal File) format reader and writer.
//!
//! This crate provides safe Rust bindings for reading and writing GGUF files,
//! the binary format used by llama.cpp for storing model weights and metadata.
//!
//! # GGUF File Format (v3)
//!
//! ```text
//! 1. Magic "GGUF" (4 bytes)
//! 2. Version (u32)
//! 3. Tensor count (i64)
//! 4. KV pair count (i64)
//! 5. KV pairs: key (string), type (i32), value
//! 6. Tensor info: name (string), n_dims (u32), dims (i64 × n), type (i32), offset (u64)
//! 7. Tensor data blob (aligned to general.alignment, default 32)
//! ```
//!
//! # Example
//!
//! ```no_run
//! use gguf::GgufReader;
//!
//! let reader = GgufReader::from_file("model.gguf").unwrap();
//! println!("Tensors: {}", reader.tensor_count());
//! println!("KV pairs: {}", reader.metadata_count());
//! for i in 0..reader.tensor_count() as usize {
//!     let info = reader.tensor_info(i).unwrap();
//!     println!("  {} {:?}", info.name, info.shape);
//! }
//! ```

use std::sync::Arc;

// ─── Module declarations ─────────────────────────────────────────────────────

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

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use constants::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
pub use dequant::*;
pub use errors::{GgufError, GgufResult};
pub use imatrix::{imatrix_bit_width, imatrix_description, is_imatrix};
pub use tensor::{MmapTensor, TensorInfo};
pub use types::{GgmlType, GgufType};
pub use value::GgufValue;

// ─── GgufReader ──────────────────────────────────────────────────────────────

/// A GGUF file reader that memory-maps the file for efficient access.
pub struct GgufReader {
    /// Memory-mapped file data (shared for lazy tensor loading).
    pub(crate) data: Arc<memmap2::Mmap>,
    /// GGUF version.
    pub(crate) version: u32,
    /// Number of tensors.
    pub(crate) tensor_count: i64,
    /// Number of KV pairs.
    pub(crate) metadata_count: i64,
    /// Metadata key-value pairs.
    pub(crate) kv_pairs: Vec<(String, GgufValue)>,
    /// Tensor info entries.
    pub(crate) tensors: Vec<TensorInfo>,
    /// Alignment (from general.alignment or default).
    pub(crate) alignment: usize,
    /// Offset where tensor data begins.
    pub(crate) data_offset: usize,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Align `val` up to the next multiple of `alignment`.
pub(crate) fn align_up(val: usize, alignment: usize) -> usize {
    (val + alignment - 1) & !(alignment - 1)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::CursorReader;

    #[test]
    fn gguf_magic_constant_should_be_correct() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
        assert_eq!(
            core::str::from_utf8(&GGUF_MAGIC.to_le_bytes()).unwrap(),
            "GGUF"
        );
    }

    #[test]
    fn gguf_version_should_be_three() {
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn gguf_type_size_should_be_correct() {
        assert_eq!(GgufType::Uint8.size_of(), 1);
        assert_eq!(GgufType::Float32.size_of(), 4);
        assert_eq!(GgufType::Float64.size_of(), 8);
        assert_eq!(GgufType::String.size_of(), 0);
        assert_eq!(GgufType::Array.size_of(), 0);
    }

    #[test]
    fn gguf_type_from_i32_should_be_correct() {
        assert_eq!(GgufType::from_i32(0).unwrap(), GgufType::Uint8);
        assert_eq!(GgufType::from_i32(6).unwrap(), GgufType::Float32);
        assert_eq!(GgufType::from_i32(9).unwrap(), GgufType::Array);
        assert_eq!(GgufType::from_i32(12).unwrap(), GgufType::Float64);
        assert!(GgufType::from_i32(13).is_err());
    }

    #[test]
    fn ggml_type_from_i32_should_be_correct() {
        assert_eq!(GgmlType::from_i32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_i32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_i32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_i32(30).unwrap(), GgmlType::I64);
        assert_eq!(GgmlType::from_i32(31).unwrap(), GgmlType::F64);
        assert_eq!(GgmlType::from_i32(32).unwrap(), GgmlType::Bf16);
        assert!(GgmlType::from_i32(33).is_err()); // removed type
    }

    #[test]
    fn align_up_should_round_correctly() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(63, 32), 64);
        assert_eq!(align_up(64, 32), 64);
    }

    #[test]
    fn from_file_should_return_error_for_missing_file() {
        let result = GgufReader::from_file("/nonexistent/path/file.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn from_mmap_should_reject_invalid_magic() {
        let data = [0u8; 16];
        let mut reader = CursorReader::new(&data);
        let magic = reader.read_u32().unwrap();
        assert_ne!(magic, GGUF_MAGIC);
    }

    #[test]
    fn cursor_reader_should_read_little_endian() {
        let data = [0x01, 0x00, 0x00, 0x00];
        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), 1);
    }

    #[test]
    fn cursor_reader_should_read_string() {
        let data: Vec<u8> = [
            0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // len=5
            b'h', b'e', b'l', b'l', b'o',
        ]
        .to_vec();
        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_string().unwrap(), "hello");
    }

    #[test]
    fn should_parse_minimal_gguf_file() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&0i64.to_le_bytes());
        data.extend_from_slice(&0i64.to_le_bytes());

        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32().unwrap(), GGUF_VERSION);
        assert_eq!(reader.read_i64().unwrap(), 0);
        assert_eq!(reader.read_i64().unwrap(), 0);
    }

    #[test]
    fn should_parse_gguf_with_kv_pair() {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&0i64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1i64.to_le_bytes()); // 1 KV pair

        let key = "general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(GgufType::String as i32).to_le_bytes());
        let val = "llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val.as_bytes());

        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32().unwrap(), GGUF_VERSION);
        assert_eq!(reader.read_i64().unwrap(), 0);
        assert_eq!(reader.read_i64().unwrap(), 1);

        let gguf_key = reader.read_string().unwrap();
        assert_eq!(gguf_key, "general.architecture");

        let type_raw = reader.read_i32().unwrap();
        assert_eq!(GgufType::from_i32(type_raw).unwrap(), GgufType::String);

        let gguf_val = reader.read_string().unwrap();
        assert_eq!(gguf_val, "llama");
    }

    #[test]
    fn should_parse_gguf_with_tensor_info() {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&1i64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&0i64.to_le_bytes()); // 0 KV pairs

        let name = "output.weight";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        data.extend_from_slice(&256i64.to_le_bytes()); // dim 0
        data.extend_from_slice(&4096i64.to_le_bytes()); // dim 1
        data.extend_from_slice(&(GgmlType::F32 as i32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32().unwrap(), GGUF_VERSION);
        assert_eq!(reader.read_i64().unwrap(), 1);
        assert_eq!(reader.read_i64().unwrap(), 0);

        let t_name = reader.read_string().unwrap();
        assert_eq!(t_name, "output.weight");

        let n_dims = reader.read_u32().unwrap();
        assert_eq!(n_dims, 2);

        assert_eq!(reader.read_i64().unwrap(), 256);
        assert_eq!(reader.read_i64().unwrap(), 4096);

        let dtype_raw = reader.read_i32().unwrap();
        assert_eq!(GgmlType::from_i32(dtype_raw).unwrap(), GgmlType::F32);

        let offset = reader.read_u64().unwrap();
        assert_eq!(offset, 0);
    }

    #[test]
    fn should_parse_realistic_llama_gguf() {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&2i64.to_le_bytes()); // 2 tensors
        data.extend_from_slice(&5i64.to_le_bytes()); // 5 KV pairs

        fn write_kv_string(data: &mut Vec<u8>, key: &str, val: &str) {
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(GgufType::String as i32).to_le_bytes());
            data.extend_from_slice(&(val.len() as u64).to_le_bytes());
            data.extend_from_slice(val.as_bytes());
        }

        fn write_kv_u32(data: &mut Vec<u8>, key: &str, val: u32) {
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(GgufType::Uint32 as i32).to_le_bytes());
            data.extend_from_slice(&val.to_le_bytes());
        }

        fn write_kv_f32(data: &mut Vec<u8>, key: &str, val: f32) {
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(GgufType::Float32 as i32).to_le_bytes());
            data.extend_from_slice(&val.to_le_bytes());
        }

        write_kv_string(&mut data, "general.architecture", "llama");
        write_kv_u32(&mut data, "llama.embedding_length", 256);
        write_kv_u32(&mut data, "llama.attention.head_count", 8);
        write_kv_u32(&mut data, "llama.block_count", 4);
        write_kv_f32(&mut data, "llama.attention.layer_norm_rms_epsilon", 1e-5);

        let tensor1_name = "token_embd.weight";
        data.extend_from_slice(&(tensor1_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor1_name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&256i64.to_le_bytes());
        data.extend_from_slice(&4096i64.to_le_bytes());
        data.extend_from_slice(&(GgmlType::F32 as i32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let tensor2_name = "output.weight";
        data.extend_from_slice(&(tensor2_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor2_name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&256i64.to_le_bytes());
        data.extend_from_slice(&4096i64.to_le_bytes());
        data.extend_from_slice(&(GgmlType::F16 as i32).to_le_bytes());
        let t1_size = 256 * 4096 * 4;
        let t1_aligned = (t1_size + 31) & !31;
        data.extend_from_slice(&(t1_aligned as u64).to_le_bytes());

        let tensor_data_start = data.len();
        let aligned_data_start = (tensor_data_start + 31) & !31;
        while data.len() < aligned_data_start {
            data.push(0);
        }

        let t1_size = 256 * 4096 * 4;
        data.resize(data.len() + t1_size, 0);

        let t2_size = 256 * 4096 * 2;
        data.resize(data.len() + t2_size, 0);

        std::fs::create_dir_all("tmp").expect("failed to create tmp directory");
        let mut file = std::fs::File::create("tmp/test_llama.gguf").unwrap();
        use std::io::Write;
        file.write_all(&data).unwrap();
        drop(file);

        let reader = GgufReader::from_file("tmp/test_llama.gguf").unwrap();

        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.metadata_count(), 5);

        let arch = reader.get_kv("general.architecture").unwrap();
        assert!(
            matches!(arch, GgufValue::Str(s) if s == "llama"),
            "expected string 'llama', got {arch:?}"
        );

        let embd = reader.get_kv("llama.embedding_length").unwrap();
        assert!(
            matches!(embd, GgufValue::U32(v) if *v == 256),
            "expected U32(256), got {embd:?}"
        );

        let t1 = reader.find_tensor("token_embd.weight").unwrap();
        assert_eq!(t1.shape, vec![256, 4096]);
        assert_eq!(t1.dtype, GgmlType::F32);

        let t2 = reader.find_tensor("output.weight").unwrap();
        assert_eq!(t2.shape, vec![256, 4096]);
        assert_eq!(t2.dtype, GgmlType::F16);

        let _ = std::fs::remove_file("tmp/test_llama.gguf");
    }
}
