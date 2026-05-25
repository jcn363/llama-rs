use config::Config;

#[test]
fn integration_config_from_env_roundtrip() {
    // SAFETY: test-only, single-threaded env var manipulation
    unsafe {
        std::env::set_var("LLAMA_MODEL_PATH", "/tmp/integration_test.gguf");
        std::env::set_var("LLAMA_NUM_THREADS", "16");
        std::env::set_var("LLAMA_VERBOSE", "1");
    }

    let cfg = Config::from_env();
    assert_eq!(
        cfg.model_path,
        Some(std::path::PathBuf::from("/tmp/integration_test.gguf"))
    );
    assert_eq!(cfg.num_threads, 16);
    assert!(cfg.verbose);

    // Cleanup
    unsafe {
        std::env::remove_var("LLAMA_MODEL_PATH");
        std::env::remove_var("LLAMA_NUM_THREADS");
        std::env::remove_var("LLAMA_VERBOSE");
    }
}

#[test]
fn integration_config_default_is_sane() {
    let cfg = Config::default();
    // Default values must be reasonable for inference
    assert!(cfg.num_threads >= 1, "at least 1 thread");
    assert!(!cfg.verbose, "verbose defaults to false");
    assert!(cfg.model_path.is_none(), "no model path by default");
}
