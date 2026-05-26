//! Model management for llama-ui.
//!
//! Provides downloader, manifest, file scanning, and GGUF metadata extraction.

#![deny(missing_docs)]

use error::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A single model entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Human-readable model name.
    pub name: String,
    /// GGUF filename.
    pub filename: String,
    /// Absolute path to the GGUF file.
    pub path: PathBuf,
    /// Quantization string (e.g., "Q4_K_M").
    pub quantization: String,
    /// Download source URL.
    pub source_url: String,
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Architecture type (e.g., "llama", "mistral").
    pub architecture: String,
    /// Maximum context length supported.
    pub context_length: usize,
    /// ISO-8601 download timestamp.
    pub downloaded_at: Option<String>,
}

/// Model manifest: a list of known models.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    /// List of model entries in the manifest.
    pub models: Vec<ModelEntry>,
}

impl Manifest {
    /// Load manifest from a JSON file path.
    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(error::Error::Io)?;
        let manifest: Self =
            serde_json::from_str(&content).map_err(|e| error::Error::Parse(e.to_string()))?;
        Ok(manifest)
    }

    /// Save manifest to a JSON file path.
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content =
            serde_json::to_string_pretty(self).map_err(|e| error::Error::Parse(e.to_string()))?;
        std::fs::write(path, content).map_err(error::Error::Io)?;
        Ok(())
    }

    /// Scan a directory for `.gguf` files and reconcile with the manifest.
    /// Adds new files not yet in the manifest; removes entries whose files are missing.
    pub fn scan(models_dir: &PathBuf, manifest: &mut Self) -> Result<()> {
        if !models_dir.exists() {
            std::fs::create_dir_all(models_dir)?;
            return Ok(());
        }

        let mut seen_paths: Vec<PathBuf> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "gguf") {
                    seen_paths.push(path);
                }
            }
        }

        // Remove manifest entries whose files no longer exist
        manifest.models.retain(|m| seen_paths.contains(&m.path));

        // Add new files not yet in manifest (with minimal metadata)
        for path in &seen_paths {
            if !manifest.models.iter().any(|m| m.path == *path) {
                let filename = path
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_default();
                let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                manifest.models.push(ModelEntry {
                    name: filename.replace(".gguf", ""),
                    filename,
                    path: path.clone(),
                    quantization: String::new(),
                    source_url: String::new(),
                    file_size_bytes: file_size,
                    architecture: String::new(),
                    context_length: 0,
                    downloaded_at: None,
                });
            }
        }

        Ok(())
    }

    /// Extract GGUF metadata from a model file and update the manifest entry in place.
    pub fn extract_metadata(path: &PathBuf, entry: &mut ModelEntry) -> Result<()> {
        let reader =
            gguf::GgufReader::from_file(path).map_err(|e| error::Error::GgufMeta(e.to_string()))?;

        // Read architecture from GGUF metadata
        if let Some(gguf::GgufValue::Str(s)) = reader.get_kv("general.architecture") {
            entry.architecture = s.clone();
        }

        // Read context length (may be stored under various keys depending on architecture)
        for key in &[
            "llama.context_length",
            "llama.context_length",
            "mistral.context_length",
            "phi.context_length",
            "gemma.context_length",
            "qwen.context_length",
        ] {
            if let Ok(len) = reader.get_usize(key) {
                entry.context_length = len;
                break;
            }
        }

        // Extract quantization from filename if not already set
        if entry.quantization.is_empty() {
            entry.quantization = guess_quantization(&entry.filename);
        }

        Ok(())
    }
}

/// Guess quantization from a GGUF filename.
fn guess_quantization(filename: &str) -> String {
    let patterns = [
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S",
        "Q3_K_L", "Q2_K", "IQ4_NL", "IQ3_S", "IQ2_S", "F16", "F32",
    ];
    for pat in &patterns {
        if filename.contains(pat) {
            return pat.to_string();
        }
    }
    "unknown".into()
}

/// Download a GGUF model file from HuggingFace or other URL.
/// Calls `progress(downloaded, total)` for progress reporting.
pub async fn download_model(url: &str, dest: &PathBuf, progress: impl Fn(u64, u64)) -> Result<()> {
    let client = reqwest::Client::builder()
        .user_agent("llama-ui/0.1")
        .build()
        .map_err(|e| error::Error::Network(e.to_string()))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| error::Error::Network(e.to_string()))?;
    let total = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let mut file = std::fs::File::create(dest)?;
    let mut stream = response.bytes_stream();

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| error::Error::Network(e.to_string()))?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        progress(downloaded, total);
    }

    Ok(())
}

use std::io::Write;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guess_quantization() {
        assert_eq!(guess_quantization("model.Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(guess_quantization("model.F16.gguf"), "F16");
        assert_eq!(guess_quantization("model.gguf"), "unknown");
    }

    #[test]
    fn test_manifest_serde_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        let mut m = Manifest::default();
        m.models.push(ModelEntry {
            name: "test".into(),
            filename: "test.gguf".into(),
            path: PathBuf::from("/tmp/test.gguf"),
            quantization: "Q4_K_M".into(),
            source_url: "https://example.com/test.gguf".into(),
            file_size_bytes: 1000,
            architecture: "llama".into(),
            context_length: 4096,
            downloaded_at: Some("2026-05-25T12:00:00Z".into()),
        });
        m.save(&path.to_path_buf()).unwrap();
        let loaded = Manifest::load(&path.to_path_buf()).unwrap();
        assert_eq!(loaded.models.len(), 1);
        assert_eq!(loaded.models[0].name, "test");
    }

    #[test]
    fn test_manifest_scan_add_and_remove() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();

        // Create one gguf file and one non-gguf file
        let a = models_dir.join("a.gguf");
        std::fs::write(&a, b"gguf").unwrap();
        let _b = models_dir.join("ignored.txt");
        std::fs::write(&_b, b"nope").unwrap();

        // Start with a manifest that references a missing file
        let mut m = Manifest::default();
        m.models.push(ModelEntry {
            name: "old".into(),
            filename: "old.gguf".into(),
            path: models_dir.join("old.gguf"),
            quantization: String::new(),
            source_url: String::new(),
            file_size_bytes: 0,
            architecture: String::new(),
            context_length: 0,
            downloaded_at: None,
        });

        // Scan should remove the missing entry and add the existing a.gguf
        Manifest::scan(&models_dir, &mut m).unwrap();
        assert_eq!(m.models.len(), 1);
        assert_eq!(m.models[0].filename, "a.gguf");
    }

    #[test]
    fn test_extract_metadata_invalid_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"not a gguf").unwrap();

        let mut entry = ModelEntry {
            name: "bad".into(),
            filename: "bad.gguf".into(),
            path: path.clone(),
            quantization: String::new(),
            source_url: String::new(),
            file_size_bytes: 0,
            architecture: String::new(),
            context_length: 0,
            downloaded_at: None,
        };

        let res = Manifest::extract_metadata(&path, &mut entry);
        assert!(res.is_err());
        if let Err(e) = res {
            // Ensure the error originates from GGUF metadata extraction
            assert!(format!("{e}").to_lowercase().contains("gguf"));
        }
    }
}
