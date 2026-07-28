#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_CODE="${PROJECT_ROOT}/evaluation/vision_opd_official"
DATA_DIR="${BENCHMARK_DATA_DIR:-${PROJECT_ROOT}/data/benchmarks}"
FAST_ONLY=0

if [[ "${1:-}" == "--fast-only" ]]; then
  FAST_ONLY=1
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/prepare_benchmarks.sh [--fast-only]"
  exit 0
elif [[ "$#" -gt 0 ]]; then
  echo "Unsupported argument: $1" >&2
  exit 2
fi

mkdir -p "${DATA_DIR}"
if [[ "${FAST_ONLY}" == "1" ]]; then
  BENCHMARKS=(vstar zoombench hrbench-4k hrbench-8k)
else
  BENCHMARKS=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
fi

for benchmark in "${BENCHMARKS[@]}"; do
  python3 "${EVAL_CODE}/prepare_data.py" \
    --benchmark "${benchmark}" \
    --data_dir "${DATA_DIR}"
done

python3 "${PROJECT_ROOT}/evaluation/prepare_fast_eval_data.py" \
  --eval-dir "${DATA_DIR}" \
  --benchmarks "vstar,zoombench,hrbench-4k,hrbench-8k" \
  --out "${DATA_DIR}/opd_fast_benchmarks.parquet"

echo "Prepared benchmark assets under ${DATA_DIR}"
