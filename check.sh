#!/bin/bash
cd /home/user/Desktop/llama-rs
echo "=== cargo check ==="
cargo check --workspace 2>&1 | tail -100
echo ""
echo "=== cargo fmt ==="
cargo fmt --all -- --check 2>&1 | tail -50
