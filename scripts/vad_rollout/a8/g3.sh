#!/bin/bash
# R9 G3 on one GPU: 16 step-0 trajectories (prompt positions 0 and 188 of the 375, 8 siblings each) from the
# Base model, then the gradient checks of a8/g3.py. It trains nothing and changes no A1 or A2 file.
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> CUDA_VISIBLE_DEVICES=<one GPU> \
#   g3.sh <A1 COMMAND.txt> <output root>
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
data="${DATA_ROOT:?DATA_ROOT must name the data root}"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
command_txt="$1"
out="$2"
mkdir -p "$out"
export PATH="$env_root/vad/bin:$PATH"
unset VAD_TOKSTATS_DIR
cd "$root" || exit 1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== git HEAD $(git rev-parse HEAD 2>/dev/null)"
  echo "== python $(command -v python3)"
  echo "== CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  grep -E "MemTotal|MemAvailable" /proc/meminfo
} > "$out/G3_ENV.txt" 2>&1
start=$(date +%s)
python3 scripts/vad_rollout/a8/generate.py "$root" "$command_txt" "$data/models/Qwen3.5-4B" "$out/rollouts_g3.jsonl" \
  0 188 375 8 > "$out/generate.log" 2>&1
status=$?
generation_seconds=$(( $(date +%s) - start ))
echo "generate exit $status seconds $generation_seconds" >> "$out/G3_ENV.txt"
[[ $status -eq 0 ]] || exit "$status"
sha256sum "$out/rollouts_g3.jsonl" >> "$out/G3_ENV.txt"
python3 scripts/vad_rollout/a8/g3.py "$root" "$command_txt" "$out/rollouts_g3.jsonl" "$out" "$generation_seconds" \
  2>&1 | tee "$out/g3.log"
status=${PIPESTATUS[0]}
echo "g3 exit $status" >> "$out/G3_ENV.txt"
exit "$status"
