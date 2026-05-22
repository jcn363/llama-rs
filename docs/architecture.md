# High‑Level Architecture Overview

## System Description

The **llama‑rs** repository implements a Rust‑based LLM inference stack.  It provides:

* **llama‑cli** – a command‑line client for interactive prompting.
* **llama‑server** – an HTTP server exposing the model as a REST‑style API.
* **ggml** – the core tensor library handling low‑level math.
* **gguf** – a format library for loading GGUF model files.
* **common** – shared utilities and abstractions used across the crates.
* **media** – visual identity system showcasing the project's branding and design language.

These components are built as separate Cargo crates under the workspace and are linked together at compile time.

## Mermaid Diagram

```mermaid
graph LR
    subgraph Client
        CLI[llama‑cli]
        UI[Web UI (optional)]
    end
    subgraph Server
        SERVER[llama‑server]
    end
    subgraph Core
        GGML[ggml]
        GGUF[gguf]
        COMMON[common]
    end

    CLI -->|HTTP/JSON| SERVER
    UI -->|WebSocket/HTTP| SERVER
    SERVER -->|calls| GGML
    SERVER -->|loads| GGUF
    GGML -->|uses| COMMON
    GGUF -->|uses| COMMON
```

## Deployment

* Build the binaries with `cargo build --release` (CUDA is enabled by default, requires CUDA toolkit).
* Deploy the `llama‑server` binary to a host with the desired model file (`model.gguf`).
* Optionally expose the server behind a reverse proxy (e.g., Nginx) for TLS termination.

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
