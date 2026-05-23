//! Tensor definition and basic operations for the ggml core.

use crate::dtype::DType;
use std::sync::Arc;

/// A multi‑dimensional array of a homogeneous data type.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Shape of the tensor, e.g. `[2, 3, 4]`.
    shape: Vec<usize>,
    /// Underlying data buffer.
    data: TensorData,
    /// Data type of each element.
    dtype: DType,
}

/// Underlying data buffer for a tensor.
#[derive(Debug)]
pub enum TensorData {
    /// Arc‑shared data (used for persistent tensors like model weights).
    Arc(Arc<[u8]>),
    /// Bump‑allocated data (used for temporary tensors in computation graphs).
    Bump(Box<[u8]>),
}

impl TensorData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            TensorData::Arc(arc) => arc.as_ref(),
            TensorData::Bump(bump) => bump.as_ref(),
        }
    }
}

impl Clone for TensorData {
    fn clone(&self) -> Self {
        match self {
            TensorData::Arc(arc) => TensorData::Arc(arc.clone()),
            TensorData::Bump(bump) => {
                // Copy the data from the bump allocation into an Arc
                let vec = bump.to_vec();
                TensorData::Arc(Arc::from(vec.into_boxed_slice()))
            }
        }
    }
}

impl Tensor {
    /// Create a new tensor filled with zeros.
    pub fn new(dtype: DType, shape: &[usize]) -> Self {
        let elem_count = shape.iter().product::<usize>();
        let size = elem_count * dtype.size_of();
        let vec = vec![0u8; size];
        // Zero‑initialisation is already done by vec![0u8; size]
        Self {
            shape: shape.to_vec(),
            data: TensorData::Arc(Arc::from(vec.into_boxed_slice())),
            dtype,
        }
    }

    /// Create a tensor from pre‑allocated data.
    ///
    /// This is useful for creating tensors from bump‑allocated memory.
    pub fn from_data(data: TensorData, dtype: DType, shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data,
            dtype,
        }
    }

    /// Return the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the data type of the tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return a reference to the raw byte buffer.
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// Return the total number of elements in the tensor.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return the number of dimensions (rank) of the tensor.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return the total byte size of the tensor's data buffer.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.element_count() * self.dtype.size_of()
    }

    /// Direct access to the underlying byte slice (alias for `as_bytes`).
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.as_bytes()
    }

    /// Convenience constructor for a tensor of `f32` values from a slice.
    /// The slice is copied into the internal buffer.
    pub fn from_f32(shape: &[usize], values: &[f32]) -> Self {
        let dtype = DType::F32;
        let elem_count = shape.iter().product::<usize>();
        assert_eq!(
            elem_count,
            values.len(),
            "shape does not match values length"
        );
        // Convert f32 slice to bytes (little‑endian).
        let mut bytes = Vec::with_capacity(elem_count * dtype.size_of());
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            shape: shape.to_vec(),
            data: TensorData::Arc(Arc::from(bytes.into_boxed_slice())),
            dtype,
        }
    }
}
