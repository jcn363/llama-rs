#![deny(missing_docs)]

//! Configuration module for the llama-rs workspace.
//!
//! Provides a unified `Config` struct that can be loaded from environment
//! variables or a configuration file. This is a minimal implementation that
//! can be expanded as needed.

use std::env;
use std::path::PathBuf;

/// Central configuration for the application.
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
}
