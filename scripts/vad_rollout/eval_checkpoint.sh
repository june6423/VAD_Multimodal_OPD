#!/bin/bash
# VLM-VAD-ROLLOUT-001 official evaluation of one checkpoint (HANDOFF-VLM-VAD-ROLLOUT-001-R1, Part A item 7).
# Usage: GPU_UUID=<uuid of the task GPU> JUDGE_API_BASE=<judge endpoint> DATA_ROOT=<data root> \
#   ENV_ROOT=<environment root> scripts/vad_rollout/eval_checkpoint.sh <arm> <step> <merged model dir> <output_root>
# The eval_arms.sh procedure of 1f12b81d for one model: scripts/vad_repro/serve_model.sh (bf16, tensor
# parallel size 1, port 8000) on the task GPU, wait for /v1/models, bash scripts/evaluate_official.sh with
# the Mistral judge and PARALLEL_WORKERS=64, then stop the server. RUN_DIR is <output_root>/run, so a
# restarted task resumes from the answer files. The served name is <arm>_step<step>. The script copies
# summary.json, summary.md, the audits and the score, judge and audit logs to <output_root>/ and writes
# <output_root>/EVAL.txt with the stage times.
set -uo pipefail

arm="$1"
step="$2"
model="$3"
out="$4"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${DATA_ROOT:-$HOME/vad-rollout/data}"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
judge_model=mistralai/Mistral-Small-3.2-24B-Instruct-2506
served="${arm}_step${step}"
benchmarks=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
log="$out/EVAL.txt"
{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== arm $arm step $step model $model served $served"
  echo "== JUDGE_API_BASE ${JUDGE_API_BASE:-<unset>}"
  echo "== GPU_UUID ${GPU_UUID:-<unset>}"
  echo "== df -h DATA_ROOT"; df -h "$data"
} >> "$log" 2>&1

# models <name>: print the /v1/models ids of port 8000; exit 0 when <name> is among them.
models() {
  python3 - "$1" 2>/dev/null <<'EOF'
import json
import sys
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
    ids = [m["id"] for m in json.load(response)["data"]]
print("models", ids)
sys.exit(0 if sys.argv[1] in ids else 1)
EOF
}
stop_server() {
  kill -TERM -- -"$server_pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || return 0; sleep 2; done
  kill -KILL -- -"$server_pid" 2>/dev/null
}

if [[ -n "$(models "")" ]]; then
  echo "port 8000 already answers; stop" >> "$log"
  exit 3
fi

t0=$(date +%s)
setsid bash "$root/scripts/vad_repro/serve_model.sh" "$model" "$served" >> "$out/model_server.log" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 240); do
  kill -0 "$server_pid" 2>/dev/null || break
  models "$served" > /dev/null && { ready=1; break; }
  sleep 5
done
if [[ $ready -ne 1 ]]; then
  { echo "model server not ready"; tail -n 80 "$out/model_server.log"; } >> "$log"
  stop_server
  exit 2
fi
t1=$(date +%s)
{
  echo "== server ready after $((t1 - t0)) s at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  grep '^COMMAND' "$out/model_server.log" | tail -n 1
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
} >> "$log" 2>&1

MODEL_API_BASE=http://127.0.0.1:8000/v1 MODEL_ID="$served" MODEL_TAG="$served" MODEL_PATH="$model" \
  JUDGE_API_BASE="${JUDGE_API_BASE:-}" JUDGE_MODEL="$judge_model" \
  BENCHMARK_DATA_DIR="$data/vad/benchmarks" RUN_DIR="$out/run" PARALLEL_WORKERS=64 \
  bash scripts/evaluate_official.sh > "$out/evaluate_official.log" 2>&1
code=$?
t2=$(date +%s)
stop_server
{
  echo "evaluate_official exit $code wall_seconds $((t2 - t1))"
  echo "-- evaluate_official.log tail (progress lines removed)"
  grep -v -E 'it/s\]|s/it\]' "$out/evaluate_official.log" | tail -n 30
  echo "-- stage end times (log mtime, UTC)"
  for b in "${benchmarks[@]}"; do
    for stage in infer answer_audit judge audit score; do
      f="$out/run/logs/$b.$stage.log"
      [[ -f "$f" ]] && echo "$b $stage $(date -u -r "$f" +%Y-%m-%dT%H:%M:%SZ)"
    done
  done
} >> "$log" 2>&1
cp "$out/run/summary.json" "$out/run/summary.md" "$out/" 2>> "$log"
mkdir -p "$out/audits" "$out/logs"
cp "$out/run/audits/"*.json "$out/audits/" 2>> "$log"
for b in "${benchmarks[@]}"; do
  for stage in score judge audit answer_audit; do
    f="$out/run/logs/$b.$stage.log"
    [[ -f "$f" ]] && grep -v -E 'it/s\]|s/it\]' "$f" > "$out/logs/$b.$stage.log"
  done
done
{ echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $code"; df -h "$data"; } >> "$log" 2>&1
exit "$code"
