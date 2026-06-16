#!/usr/bin/env bash
# Manually serve the Qwen3 extractor (the pipeline does this automatically via
# VLLMServer; use this only for --no-server runs or debugging).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct}"
PORT="${PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-32768}"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN"
