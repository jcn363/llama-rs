# Security Considerations for llama‑rs

## Threat Model
- **Model Loading** – Untrusted GGUF files could contain malformed metadata that triggers panics or excessive memory allocation. The loader validates required keys and falls back to safe defaults.
- **GPU Memory Management** – CUDA off‑loading may allocate large buffers. Bounds checks ensure we never exceed the device's VRAM; the `cuda_vram_test` benchmark now asserts a minimum of 1.9 GiB.
- **KV‑Cache** – The cache holds intermediate key/value tensors. An attacker could craft prompts that cause the cache to grow without bound. The `CacheStrategy` (especially `SlidingWindow`) limits memory usage.
- **Network Exposure** – The `llama-server` binds to a configurable host/port. By default it binds to `127.0.0.1`. Production deployments should restrict access via firewalls or TLS termination.

## Mitigations
1. **Input Validation**
   - All numeric fields parsed from GGUF are checked for overflow.
   - Prompt strings are limited to `max_seq_len` tokens before inference.
2. **Memory Limits**
   - `KvCacheManager` enforces the selected `CacheStrategy` after each token generation.
   - CUDA allocations are wrapped in safe Rust abstractions; errors propagate as `Result`.
3. **Error Handling**
   - Functions return `Result<_, GgufError>` or `Result<_, anyhow::Error>`; panics are avoided.
   - The CLI and server exit gracefully with an error message on failure.
4. **Secure Defaults**
   - Server binds to `127.0.0.1` unless overridden.
   - `CacheStrategy::Full` is safe for most workloads; `SlidingWindow` can be enabled for tighter memory budgets.
5. **Auditing & Logging**
   - The `profile` module can export JSON logs (`ProfileResult::to_json`) for post‑mortem analysis.
   - Consider integrating with a logging framework for production deployments.

## Recommendations for Deployers
- Run the server behind a reverse proxy with TLS.
- Monitor GPU memory usage; adjust `CacheStrategy` if needed.
- Keep the Rust toolchain up‑to‑date to receive security patches.
- Review the `Cargo.lock` for vulnerable dependencies using `cargo audit`.

---
*For a full security review, see the `SECURITY.md` in the upstream LLaMA repository.*
