#!/bin/bash
# VLM-VAD-ROLLOUT-001 input-length filter (researcher decision of 2026-10-29 after DIRECTIVE-VLM-VAD-ROLLOUT-001-003).
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> [WORKERS=<processes>] \
#   scripts/vad_rollout/prompt_filter.sh <output_root>
# Runs prompt_filter.py with max model len 9216 (train_vad.sh MAX_MODEL_LEN) on train_vad.parquet and on the
# four-benchmark opd_fast_benchmarks.parquet. Writes train_vad_le9216.parquet and opd_fast_benchmarks_le9216.parquet
# next to the inputs, and PROMPT_FILTER.txt, PROMPT_FILTER.json, SKIPPED_train.jsonl, SKIPPED_val.jsonl under
# <output_root>. No GPU.
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${DATA_ROOT:?DATA_ROOT must name the data root}"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
workers="${WORKERS:-16}"
train="$data/vad/train_vad.parquet"
val="$data/vad/benchmarks/opd_fast_benchmarks.parquet"
python="$env_root/vad/bin/python"
mkdir -p "$out"
log="$out/PROMPT_FILTER.txt"
cd "$root" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""

{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== workers $workers max_model_len 9216"
} > "$log"
start=$(date +%s)
"$python" scripts/vad_rollout/prompt_filter.py "$data/models/Qwen3.5-4B" \
  chat_templates/perception_chat_template_qwen35.jinja 9216 "$workers" "$out/PROMPT_FILTER.json" \
  "$train" "$data/vad/train_vad_le9216.parquet" "$out/SKIPPED_train.jsonl" \
  "$val" "$data/vad/benchmarks/opd_fast_benchmarks_le9216.parquet" "$out/SKIPPED_val.jsonl" >> "$log" 2>&1
code=$?
echo "prompt_filter exit $code wall_seconds $(( $(date +%s) - start ))" >> "$log"
exit "$code"
