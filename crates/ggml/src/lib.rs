#![deny(missing_docs)]
#![allow(clippy::too_many_arguments)]

/*!
 * Core tensor library and computation graph for the `ggml` crate.
 *
 * This module currently provides the public API surface for the ggml core.
 * It re‑exports the most commonly used types from the sub‑modules.
 *
 * The implementation details are split into several sub‑modules to keep the
 * codebase modular and maintainable.  New functionality should be added to a
 * dedicated module rather than expanding this file.
 *
 * # Modules
 *
 * * `tensor` – Tensor definition and basic operations.
 * * `graph` – Simple computation graph utilities.
 * * `dtype` – Data‑type enumeration and conversion helpers.
 *
 * The sub‑modules are declared here and can be expanded as needed.
 */

/// Hardware backend trait for tensor operations.
///
/// Defines the [`Backend`] trait that all hardware backends (CPU, CUDA, etc.)
/// must implement.
///
/// This is the plugin interface: adding a new hardware backend means creating a
/// new crate that implements [`Backend`] and registering it with the registry.
pub mod backend;

/// Operation types for the ggml computation graph.
///
/// Defines the various operations that can be performed in the computation graph.
/// Each variant corresponds to a specific tensor operation.
pub mod op_type;

/// Data-type enumeration and conversion helpers.
///
/// This module defines the `DType` enum and provides conversion helpers.
pub mod dtype;

/// Simple computation graph utilities.
///
/// This module provides utilities for building and manipulating computation graphs.
pub mod graph;

/// Default CPU implementations of tensor operations.
///
/// Provides fallback implementations for all tensor operations
/// that hardware backends can override for performance.
pub mod defaults;

/// Stub implementations for performance improvements.
///
/// These are placeholders for future SIMD-accelerated or otherwise optimized
/// implementations of certain operations.
pub mod improvements;

/// Tensor definition and basic operations.
///
/// This module defines the `Tensor` struct and basic operations on tensors.
pub mod tensor;

// Re‑export the most important items for a convenient top‑level API.
// Backend module removed; QuantType not currently used.
pub use dtype::DType;
pub use graph::Graph;
pub use tensor::Tensor;

// The crate is deliberately minimal at this stage.  Additional functionality
// should be implemented in the appropriate sub‑module and then re‑exported
// above if it forms part of the public API.
