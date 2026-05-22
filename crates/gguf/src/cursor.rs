//! Little-endian binary reader over a byte slice.
//!
//! Provides the low-level reading primitives used by the GGUF parser.
//! Extracted from the monolithic `lib.rs` to keep modules focused.

use crate::{GgufError, GgufResult, GgufType, GgufValue};

/// Little-endian binary reader over a byte slice.
pub(crate) struct CursorReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CursorReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    fn ensure(&self, n: usize) -> GgufResult<()> {
        if self.pos + n > self.data.len() {
            Err(GgufError::UnexpectedEof)
        } else {
            Ok(())
        }
    }

    pub(crate) fn read_u8(&mut self) -> GgufResult<u8> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub(crate) fn read_i8(&mut self) -> GgufResult<i8> {
        self.ensure(1)?;
        let v = i8::from_ne_bytes([self.data[self.pos]]);
        self.pos += 1;
        Ok(v)
    }

    pub(crate) fn read_u16(&mut self) -> GgufResult<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    pub(crate) fn read_i16(&mut self) -> GgufResult<i16> {
        self.ensure(2)?;
        let v = i16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    pub(crate) fn read_u32(&mut self) -> GgufResult<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    pub(crate) fn read_i32(&mut self) -> GgufResult<i32> {
        self.ensure(4)?;
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    pub(crate) fn read_u64(&mut self) -> GgufResult<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    pub(crate) fn read_i64(&mut self) -> GgufResult<i64> {
        self.ensure(8)?;
        let v = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    pub(crate) fn read_f32(&mut self) -> GgufResult<f32> {
        self.ensure(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    pub(crate) fn read_f64(&mut self) -> GgufResult<f64> {
        self.ensure(8)?;
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    pub(crate) fn read_string(&mut self) -> GgufResult<String> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| GgufError::DecodeError(format!("invalid UTF-8 in string: {e}")))?;
        let s = s.to_string();
        self.pos += len;
        Ok(s)
    }

    pub(crate) fn read_value(&mut self, gguf_type: GgufType) -> GgufResult<GgufValue> {
        match gguf_type {
            GgufType::Uint8 => Ok(GgufValue::U8(self.read_u8()?)),
            GgufType::Int8 => Ok(GgufValue::I8(self.read_i8()?)),
            GgufType::Uint16 => Ok(GgufValue::U16(self.read_u16()?)),
            GgufType::Int16 => Ok(GgufValue::I16(self.read_i16()?)),
            GgufType::Uint32 => Ok(GgufValue::U32(self.read_u32()?)),
            GgufType::Int32 => Ok(GgufValue::I32(self.read_i32()?)),
            GgufType::Float32 => Ok(GgufValue::F32(self.read_f32()?)),
            GgufType::Bool => {
                let v = self.read_i8()?;
                Ok(GgufValue::Bool(v != 0))
            }
            GgufType::String => Ok(GgufValue::Str(self.read_string()?)),
            GgufType::Uint64 => Ok(GgufValue::U64(self.read_u64()?)),
            GgufType::Int64 => Ok(GgufValue::I64(self.read_i64()?)),
            GgufType::Float64 => Ok(GgufValue::F64(self.read_f64()?)),
            GgufType::Array => {
                let elem_type_raw = self.read_i32()?;
                let elem_type = GgufType::from_i32(elem_type_raw)?;
                let n = self.read_u64()? as usize;
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    data.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array { elem_type, data })
            }
        }
    }
}
