#!/usr/bin/env bash

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8900}"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-120b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-120b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Starts the GPT-OSS-120B judge on port ${PORT} with the Vision-OPD reproduction profile."
  exit 0
fi

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
