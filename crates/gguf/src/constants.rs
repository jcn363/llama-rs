// ─── Constants ───────────────────────────────────────────────────────────────

/// GGUF magic bytes: "GGUF" in little-endian u32.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF version.
pub const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data.
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;
