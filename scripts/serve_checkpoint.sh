#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/serve_checkpoint.sh MERGED_CHECKPOINT [SERVED_MODEL_NAME]"
  exit 0
fi
if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/serve_checkpoint.sh MERGED_CHECKPOINT [SERVED_MODEL_NAME]" >&2
  exit 2
fi

MODEL_PATH="$1"
SERVED_MODEL_NAME="${2:-VCD}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code
