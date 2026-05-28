// Error handling shared across crates
pub use error::Error;
pub use error::Result;

/// Shared CLI argument parsing.
pub mod args;

/// Shared sampling configuration for text generation.
pub mod sampling;
