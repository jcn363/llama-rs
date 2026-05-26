//! Command-line interface for llama inference.
//!
//! Usage: `llama-cli -m model.gguf -p "Hello, world!"`

#![deny(missing_docs)]

use clap::Parser;
use llama::SamplingConfig;
use llama::{BackendType, CacheStrategy, InferenceContext, Model, ModelConfig};
use std::sync::Arc;
use std::time::Instant;

/// Command-line arguments for llama-cli.
#[derive(Parser, Debug)]
#[command(name = "llama-cli", about = "LLaMA inference CLI")]
struct Args {
    /// Flattened common arguments.
    #[clap(flatten)]
    common: common::args::CommonArgs,

    /// Prompt text.
    #[arg(short, long, default_value = "")]
    prompt: String,

    /// Maximum tokens to generate.
    #[arg(short = 'n', long, default_value_t = 128)]
    n_predict: usize,

    /// Backend to use: auto, cpu, cuda (default: auto).
    #[arg(long, default_value_t = String::from("auto"))]
    backend: String,

    /// Offload FFN weights to RAM (load on demand) to save VRAM.
    #[arg(long, default_value_t = false)]
    offload_ffn: bool,

    /// Size of thread-local memory pool for small temporary allocations (in bytes, 0 = disabled).
    #[arg(long, default_value_t = 0)]
    memory_pool_size: usize,

    /// Enable verbose logging.
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Sampling temperature (0.0 = greedy, default: 0.8).
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-k sampling (0 = disabled, default: 40).
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Top-p nucleus sampling (1.0 = disabled, default: 0.95).
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Random seed for reproducibility (default: random).
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
                .unwrap(),
        )
        .init();

    let args = Args::parse();
    let common = &args.common;
    tracing::info!(
        "model={}, prompt={}, n_predict={}",
        common.model,
        args.prompt,
        args.n_predict
    );

    tracing::info!("Loading model from: {}", common.model);
    let load_start = Instant::now();

    let model = Arc::new(Model::load_from_gguf(&common.model, args.offload_ffn)?);
    let load_time = load_start.elapsed();
    tracing::info!("Model loaded in {:.2}s", load_time.as_secs_f32());
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
        offload_ffn: args.offload_ffn,
        memory_pool_size: args.memory_pool_size,
        ..Default::default()
    };

    let sampling = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repeat_penalty: 1.1,
        seed: args.seed,
    };

    let mut ctx = InferenceContext::new(model, config).with_sampling(sampling);

    if args.prompt.is_empty() {
        println!("Interactive mode — type your prompt and press Enter:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();
        if input.is_empty() {
            eprintln!("No prompt provided. Exiting.");
            return Ok(());
        }
        println!("Prompt: {}", input);
        ctx.encode(&input);
        let generated = generate_with_progress(&mut ctx, &input, args.n_predict)?;
        print_generated(&ctx, &generated);
    } else {
        println!("Prompt: {}", args.prompt);
        let generated = generate_with_progress(&mut ctx, &args.prompt, args.n_predict)?;
        print_generated(&ctx, &generated);
    }

    Ok(())
}

/// Generate tokens with timing and progress feedback.
fn generate_with_progress(
    ctx: &mut InferenceContext,
    prompt: &str,
    n_predict: usize,
) -> anyhow::Result<Vec<usize>> {
    eprint!("Generating {} tokens...", n_predict);
    let gen_start = Instant::now();
    let generated = ctx.generate(prompt, n_predict)?;
    let gen_time = gen_start.elapsed();

    let prompt_tokens = ctx.encode(prompt).len();
    let output_tokens = generated.len().saturating_sub(prompt_tokens);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        output_tokens as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    eprintln!(
        " done! ({output_tokens} tokens in {:.2}s, {:.1} tok/s)",
        gen_time.as_secs_f32(),
        tokens_per_sec
    );

    Ok(generated)
}

/// Print generated token IDs as decoded text.
fn print_generated(ctx: &InferenceContext, generated: &[usize]) {
    for token_id in generated {
        print!("{}", ctx.decode_from_id(*token_id));
    }
    println!();
}
