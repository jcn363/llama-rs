// Common test utilities for the llama-rs workspace

use std::path::PathBuf;
use std::fs;
use anyhow::Result;

/// Load a test model file from the `tests/models` directory.
pub fn load_test_model(name: &str) -> Result<PathBuf> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("models");
    path.push(name);
    if path.exists() {
        Ok(path)
    } else {
        Err(anyhow::anyhow!(format!("Test model not found: {}", path.display())))
    }
}

/// Helper to read a file into a string for test fixtures.
pub fn read_fixture(path: &PathBuf) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}
