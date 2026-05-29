# High‑Level Architecture Overview

## System Description

The **llama‑rs** repository implements a Rust‑based LLM inference stack.  It provides:

* **llama‑cli** – a command‑line client for interactive prompting.
* **llama‑server** – an HTTP server exposing the model as a REST‑style API.
* **llama-ui** – a native desktop GUI (iced 0.13) for interactive LLM inference.
* **ggml** – the core tensor library handling low‑level math.
* **gguf** – a format library for loading GGUF model files.
* **common** – shared utilities and abstractions used across the crates.
* **media** – visual identity system showcasing the project's branding and design language.

These components are built as separate Cargo crates under the workspace and are linked together at compile time.

## Mermaid Diagram

```mermaid
graph LR
    subgraph Client
        CLI[llama-cli]
        UI[llama-ui (desktop)]
    end
    subgraph Server
        SERVER[llama-server]
    end
    subgraph Core
        GGML[ggml]
        GGUF[gguf]
        COMMON[common]
    end

    CLI -->|HTTP/JSON| SERVER
    UI -->|SSE/HTTP| SERVER
    SERVER -->|calls| GGML
    SERVER -->|loads| GGUF
    GGML -->|uses| COMMON
    GGUF -->|uses| COMMON
```

## Hardware Backend Plugin System

The project uses a trait-based plugin architecture for hardware acceleration:

| Component | Role |
|-----------|------|
| `ggml::backend::Backend` (trait) | Object-safe trait defining `mat_vec`, `add`, `mul` with CPU defaults |
| `ggml_cpu::CpuBackend` | CPU implementation — SIMD-accelerated via AVX/SSE4.2 + `std::thread::scope` |
| `ggml_cuda::CudaBackend` | CUDA implementation — cuBLAS `gemm` with transparent CPU fallback |
| `llama::create_backend` | Factory: `(&ModelConfig) -> Arc<dyn Backend>`, priority chain CUDA → CPU |

**Selection priority:**
1. CUDA (if `--backend cuda` or `auto` + CUDA available)
2. CPU (always available fallback)

**CLI:**
```bash
llama-cli -m model.gguf --backend cuda -p "Hello" -n 128
```

## Scaling

* **Horizontal scaling** – run multiple `llama‑server` instances behind a load balancer. Each instance loads its own model into memory; ensure the host has sufficient RAM/VRAM.
* **GPU scaling** – use the `ggml‑cuda` crate with the `cuda` feature enabled. Deploy on GPU‑enabled nodes and configure each instance to use a distinct GPU device.
* **Stateless design** – the server is stateless between requests, making it straightforward to add or remove instances.

## Monitoring

* Export Prometheus metrics from `llama‑server` (e.g., request latency, active connections).
* Log inference timings and errors using the `tracing` crate.
* Use system‑level monitoring (GPU utilization, memory usage) to trigger autoscaling policies.

---
*This document provides a concise overview for developers and operators.*
