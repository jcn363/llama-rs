//! llama-ui — Desktop GUI for llama-rs LLM inference.

#![deny(missing_docs)]

mod app;

use app::LlamaApp;
use tracing_subscriber;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
                .unwrap(),
        )
        .init();

    tracing::info!("llama-ui starting...");

    let _ = LlamaApp::run();
}