#!/bin/bash
# VLM-VAD-ROLLOUT-001 validation set without hrbench-8k (DIRECTIVE-VLM-VAD-ROLLOUT-001-003, items 1 and 2).
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> scripts/vad_rollout/val_no8k.sh <output_root>
# Builds $DATA_ROOT/vad/benchmarks/opd_fast_benchmarks_no8k.parquet (vstar, zoombench, hrbench-4k) with
# evaluation/prepare_fast_eval_data.py from the benchmark JSON files of data_download.sh, records its row count
# and SHA-256, then counts the prompt tokens of every row of the four-benchmark and the no8k parquet
# (val_prompt_tokens.py). No GPU. Records: VAL_NO8K.txt and VAL_PROMPT_TOKENS.json.
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${DATA_ROOT:?DATA_ROOT must name the data root}"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
bench="$data/vad/benchmarks"
full="$bench/opd_fast_benchmarks.parquet"
no8k="$bench/opd_fast_benchmarks_no8k.parquet"
python="$env_root/vad/bin/python"
mkdir -p "$out"
log="$out/VAL_NO8K.txt"
cd "$root" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""

{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== prepare_fast_eval_data.py --benchmarks vstar,zoombench,hrbench-4k"
} > "$log"
"$python" evaluation/prepare_fast_eval_data.py --eval-dir "$bench" --benchmarks vstar,zoombench,hrbench-4k \
  --out "$no8k" >> "$log" 2>&1
code=$?
echo "prepare exit $code" >> "$log"
[[ $code -eq 0 ]] || exit "$code"
{
  echo "== rows"
  "$python" - "$full" "$no8k" <<'EOF'
import sys

import pandas as pd

for path in sys.argv[1:]:
    frame = pd.read_parquet(path)
    print(path, len(frame), frame["data_source"].value_counts(sort=False).to_dict())
EOF
  echo "== sha256"
  sha256sum "$full" "$no8k"
  echo "== val_prompt_tokens.py"
} >> "$log" 2>&1

start=$(date +%s)
"$python" scripts/vad_rollout/val_prompt_tokens.py "$data/models/Qwen3.5-4B" \
  chat_templates/perception_chat_template_qwen35.jinja "$out/VAL_PROMPT_TOKENS.json" "$full" "$no8k" >> "$log" 2>&1
code=$?
echo "val_prompt_tokens exit $code wall_seconds $(( $(date +%s) - start ))" >> "$log"
exit "$code"
