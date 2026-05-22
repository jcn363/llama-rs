//! Simple computation graph utilities for ggml.

use crate::tensor::Tensor;
use std::collections::HashMap;

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Identifier of the node.
    pub id: usize,
    /// Tensor produced by this node.
    pub tensor: Tensor,
    /// Optional list of input node ids.
    pub inputs: Vec<usize>,
}

/// A directed acyclic graph of tensor operations.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: HashMap<usize, Node>,
    next_id: usize,
}

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node producing `tensor` with given `inputs`.
    pub fn add_node(&mut self, tensor: Tensor, inputs: Vec<usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            Node {
                id,
                tensor,
                inputs,
            },
        );
        id
    }

    /// Retrieve a node by id.
    pub fn get(&self, id: usize) -> Option<&Node> {
        self.nodes.get(&id)
    }
}

