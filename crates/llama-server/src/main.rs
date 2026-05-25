//! HTTP server for llama inference.
//!
//! Usage: `llama-server -m model.gguf --host 0.0.0.0 --port 8080`

#![deny(missing_docs)]

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
};
use clap::Parser;
use futures::StreamExt;
use futures::stream::Stream;
use llama::{BackendType, CacheStrategy, InferenceContext, Model, ModelConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::CorsLayer;

/// Command-line arguments for llama-server.
#[derive(Parser, Debug)]
#[command(name = "llama-server", about = "LLaMA HTTP server")]
struct Args {
    /// Flattened common arguments.
    #[clap(flatten)]
    common: common::args::CommonArgs,

    /// Host to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on.
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Backend to use: auto, cpu, cuda (default: auto).
    #[arg(long, default_value_t = String::from("auto"))]
    backend: String,

    /// Sampling temperature (0.0 = greedy, default: 0.8).
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-k sampling (0 = disabled, default: 40).
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Top-p nucleus sampling (1.0 = disabled, default: 0.95).
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Random seed for reproducibility.
    #[arg(long)]
    seed: Option<u64>,
}

/// Shared server state.
#[derive(Clone)]
struct ServerState {
    model: Arc<Model>,
    config: ModelConfig,
}

/// Completion request body.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_temperature")]
    #[expect(dead_code)]
    temperature: f32,
}

fn default_max_tokens() -> usize {
    128
}

fn default_temperature() -> f32 {
    0.8
}

/// Completion response body.
#[derive(Serialize)]
struct CompletionResponse {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// Tokens generated per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_sec: Option<f64>,
}

/// Streaming chunk response.
#[derive(Serialize)]
struct StreamChunk {
    content: String,
    stop: bool,
}

/// Error response body.
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
                .unwrap(),
        )
        .init();

    let common = &args.common;
    tracing::info!("Loading model from: {}", common.model);
    let model = Model::load_from_gguf(&common.model, false)?;
    tracing::info!("{}", model.summary());

    let backend_type = match args.backend.as_str() {
        "cpu" => BackendType::Cpu,
        "cuda" => BackendType::Cuda,
        _ => BackendType::Auto,
    };

    // Load configuration: CLI args take precedence over environment variables.
    let cfg = config::Config::from_env();
    let n_threads = if common.threads > 0 {
        common.threads
    } else {
        cfg.num_threads
    };
    // Supported strategies: "full" (default), "prefix", "prefix_only"/"prefix-only".
    // "sliding_window" is available programmatically but requires a window size parameter.
    let cache_strategy = match common.cache_strategy.as_str() {
        "prefix" => CacheStrategy::Prefix,
        "prefix_only" | "prefix-only" => CacheStrategy::PrefixOnly,
        _ => CacheStrategy::Full,
    };
    let config = ModelConfig {
        n_threads,
        use_cuda: common.use_cuda,
        backend_type,
        n_ctx: common.ctx_size,
        n_batch: common.batch_size,
        cache_strategy,
        ..Default::default()
    };

    let state = ServerState {
        model: Arc::new(model),
        config,
    };

    // CORS: allow all origins for local development / API consumption.
    let cors = CorsLayer::permissive();

    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/completion", post(handle_completion))
        .route("/v1/models", get(handle_v1_models))
        .layer(cors)
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Health check endpoint.
async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// OpenAI-compatible model listing.
async fn handle_v1_models(State(state): State<ServerState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "default",
            "object": "model",
            "created": 0,
            "owned_by": "llama-rs",
            "description": state.model.summary(),
        }]
    }))
}

/// Handle completion (both streaming and non-streaming).
async fn handle_completion(
    State(state): State<ServerState>,
    Json(request): Json<CompletionRequest>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let max_tokens = request.max_tokens.min(4096);
    if request.prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "prompt must not be empty".into(),
            }),
        ));
    }

    tracing::info!(
        "Completion request: prompt_len={}, max_tokens={}, stream={}",
        request.prompt.len(),
        max_tokens,
        request.stream
    );

    if request.stream {
        return Ok(axum::response::IntoResponse::into_response(
            handle_streaming(state, request, max_tokens).await,
        ));
    }

    // Non-streaming: generate all tokens then return
    let gen_start = Instant::now();
    let model = Arc::clone(&state.model);

    let mut ctx = InferenceContext::new(model, state.config.clone());
    ctx.encode(&request.prompt);

    let generated = ctx.generate(&request.prompt, max_tokens).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Inference error: {e}"),
            }),
        )
    })?;

    let prompt_tokens = ctx.encode(&request.prompt).len();
    let output_tokens = generated.len().saturating_sub(prompt_tokens);
    let gen_time = gen_start.elapsed();
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        Some(output_tokens as f64 / gen_time.as_secs_f64())
    } else {
        None
    };

    let content = generated
        .iter()
        .map(|&id| ctx.decode_from_id(id))
        .collect::<String>();

    Ok(axum::response::IntoResponse::into_response(Json(
        CompletionResponse {
            content,
            model: Some(state.model.summary()),
            tokens_per_sec,
        },
    )))
}

/// Handle streaming completion via SSE.
async fn handle_streaming(
    state: ServerState,
    request: CompletionRequest,
    max_tokens: usize,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let model = Arc::clone(&state.model);
    let config = state.config.clone();
    let prompt = request.prompt;

    // Run inference in a blocking thread pool
    let stream = tokio::task::spawn_blocking(move || {
        let mut ctx = InferenceContext::new(model, config);
        ctx.encode(&prompt);

        let mut chunks = Vec::new();

        // Generate tokens one at a time
        for _ in 0..max_tokens {
            let generated = match ctx.generate(&prompt, 1) {
                Ok(tokens) => tokens,
                Err(_) => break,
            };

            if generated.is_empty() {
                break;
            }

            let token_id = generated[0];
            let content = ctx.decode_from_id(token_id);
            chunks.push(StreamChunk {
                content,
                stop: false,
            });
        }

        // Add final stop chunk
        chunks.push(StreamChunk {
            content: String::new(),
            stop: true,
        });

        chunks
    })
    .await
    .unwrap_or_default();

    let event_stream = futures::stream::iter(stream).map(|chunk| {
        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    Sse::new(event_stream)
}

/// Signal handler for graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { tracing::info!("Received Ctrl+C, shutting down..."); }
        _ = terminate => { tracing::info!("Received SIGTERM, shutting down..."); }
    }
}
