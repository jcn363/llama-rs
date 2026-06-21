//! Shared utilities for the llama-rs workspace.
//!
//! This crate provides reusable components used by CLI, server, and GUI
//! binaries: CLI argument parsing, sampling configuration, and chat
//! template rendering.

#![deny(missing_docs)]

/// Re-export of the unified error type.
pub use error::Error;
/// Re-export of the unified result alias.
pub use error::Result;

/// Shared CLI argument parsing.
pub mod args;

/// Shared sampling configuration for text generation.
pub mod sampling;

/// Chat template rendering using minijinja.
pub mod chat_templates;
