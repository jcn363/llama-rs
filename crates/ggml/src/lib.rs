#![deny(missing_docs)]
#![allow(clippy::too_many_arguments)]

//! Core tensor library and computation graph for the `ggml` crate.
//!
//! This module currently provides the public API surface for the ggml core.
//! It re‑exports the most commonly used types from the sub‑modules.
//!
//! The implementation details are split into several sub‑modules to keep the
//! codebase modular and maintainable.  New functionality should be added to a
//! dedicated module rather than expanding this file.
//!
//! # Modules
//!
//! * `tensor` – Tensor definition and basic operations.
//! * `graph` – Simple computation graph utilities.
//! * `dtype` – Data‑type enumeration and conversion helpers.
//!
//! The sub‑modules are declared here and can be expanded as needed.

pub mod backend;
pub mod dtype;
pub mod graph;
pub mod improvements;
pub mod tensor;

// Re‑export the most important items for a convenient top‑level API.
// Backend module removed; QuantType not currently used.
pub use dtype::DType;
pub use graph::Graph;
pub use tensor::Tensor;

// The crate is deliberately minimal at this stage.  Additional functionality
// should be implemented in the appropriate sub‑module and then re‑exported
// above if it forms part of the public API.
