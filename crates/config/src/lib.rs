#![deny(missing_docs)]

//! Configuration module for the llama-rs workspace.
//!
//! Provides a unified `Config` struct loaded from environment variables,
//! and a `UiConfig` struct for GUI preferences (TOML-based).

use std::env;
use std::path::PathBuf;

use common::sampling::SamplingConfig;

/// Central configuration for the application (CLI/server).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Optional path to a model file.
    pub model_path: Option<PathBuf>,
    /// Number of threads to use for inference.
    pub num_threads: usize,
    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: None,
            num_threads: 1,
            verbose: false,
        }
    }
}

impl Config {
    /// Parse a thread count string, returning `None` on invalid input.
    fn parse_num_threads(s: &str) -> Option<usize> {
        s.parse::<usize>().ok()
    }

    /// Parse a verbose flag string (accepted: `"1"`, `"true"`, `"TRUE"`).
    fn parse_verbose(s: &str) -> bool {
        matches!(s, "1" | "true" | "TRUE")
    }

    /// Load configuration from environment variables.
    ///
    /// Recognised variables:
    /// - `LLAMA_MODEL_PATH`
    /// - `LLAMA_NUM_THREADS`
    /// - `LLAMA_VERBOSE`
    pub fn from_env() -> Self {
        let mut cfg = Config::default();
        if let Ok(p) = env::var("LLAMA_MODEL_PATH") {
            cfg.model_path = Some(PathBuf::from(p));
        }
        if let Ok(t) = env::var("LLAMA_NUM_THREADS") {
            if let Some(num) = Self::parse_num_threads(&t) {
                cfg.num_threads = num;
            }
        }
        if let Ok(v) = env::var("LLAMA_VERBOSE") {
            cfg.verbose = Self::parse_verbose(&v);
        }
        cfg
    }
}

/// UI preferences for the llama-ui desktop application.
///
/// Stored as TOML at `$XDG_CONFIG_HOME/llama-ui/prefs.toml`.
///
/// Sampling defaults (`temperature`, `top_k`, `top_p`) are inherited from
/// [`common::sampling::SamplingConfig`] via `#[serde(flatten)]` for
/// backward-compatible TOML serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UiConfig {
    /// Theme name ("dark", "light", "system").
    #[serde(default = "default_theme")]
    pub theme: String,
    /// Font size for chat text.
    #[serde(default = "default_font_size")]
    pub font_size: u16,
    /// Default max tokens for generation.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Whether to start maximized.
    #[serde(default)]
    pub start_maximized: bool,
    /// Sampling configuration (flattened for TOML compat).
    #[serde(flatten)]
    pub sampling: SamplingConfig,
}

fn default_theme() -> String {
    "dark".into()
}
fn default_font_size() -> u16 {
    14
}
fn default_max_tokens() -> usize {
    512
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: default_theme(),
            font_size: default_font_size(),
            max_tokens: default_max_tokens(),
            start_maximized: false,
            sampling: SamplingConfig::default(),
        }
    }
}

impl UiConfig {
    /// Path to the UI preferences file.
    fn path() -> PathBuf {
        let base = if let Ok(dir) = env::var("XDG_CONFIG_HOME") {
            PathBuf::from(dir)
        } else {
            let home = env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".config")
        };
        base.join("llama-ui").join("prefs.toml")
    }

    /// Load UI config from TOML file, or return defaults.
    pub fn load() -> Self {
        let path = Self::path();
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(cfg) => return cfg,
                    Err(e) => tracing::warn!("Failed to parse UiConfig: {e}"),
                },
                Err(e) => tracing::warn!("Failed to read UiConfig: {e}"),
            }
        }
        Self::default()
    }

    /// Save UI config to TOML file.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_should_be_reasonable() {
        let cfg = Config::default();
        assert!(cfg.num_threads >= 1);
        assert!(!cfg.verbose);
        assert!(cfg.model_path.is_none());
    }

    #[test]
    fn parse_num_threads_should_parse_valid_values() {
        assert_eq!(Config::parse_num_threads("1"), Some(1));
        assert_eq!(Config::parse_num_threads("8"), Some(8));
        assert_eq!(Config::parse_num_threads("64"), Some(64));
    }

    #[test]
    fn parse_num_threads_should_return_none_for_invalid() {
        assert_eq!(Config::parse_num_threads(""), None);
        assert_eq!(Config::parse_num_threads("abc"), None);
        assert_eq!(Config::parse_num_threads("-1"), None);
    }

    #[test]
    fn parse_verbose_should_accept_true_values() {
        assert!(Config::parse_verbose("1"));
        assert!(Config::parse_verbose("true"));
        assert!(Config::parse_verbose("TRUE"));
    }

    #[test]
    fn parse_verbose_should_reject_false_values() {
        assert!(!Config::parse_verbose("0"));
        assert!(!Config::parse_verbose("false"));
        assert!(!Config::parse_verbose("FALSE"));
        assert!(!Config::parse_verbose("yes"));
    }

    #[test]
    fn config_debug_and_clone_should_work() {
        let cfg = Config::default();
        let _cloned = cfg.clone();
        let _debug = format!("{cfg:?}");
    }

    #[test]
    fn ui_config_defaults() {
        let cfg = UiConfig::default();
        assert_eq!(cfg.theme, "dark");
        assert_eq!(cfg.font_size, 14);
        assert_eq!(cfg.max_tokens, 512);
        assert_eq!(cfg.sampling.temperature, 0.8);
        assert_eq!(cfg.sampling.top_k, 40);
        assert_eq!(cfg.sampling.top_p, 0.95);
    }

    #[test]
    fn ui_config_toml_roundtrip() {
        let cfg = UiConfig {
            theme: "light".into(),
            font_size: 16,
            max_tokens: 1024,
            start_maximized: true,
            sampling: SamplingConfig {
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                ..SamplingConfig::default()
            },
        };
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let parsed: UiConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.theme, "light");
        assert_eq!(parsed.font_size, 16);
        assert_eq!(parsed.max_tokens, 1024);
        assert!(parsed.start_maximized);
        assert_eq!(parsed.sampling.temperature, 0.7);
        assert_eq!(parsed.sampling.top_k, 50);
        assert_eq!(parsed.sampling.top_p, 0.9);
    }
}
