#!/bin/bash
# VLM-VAD-REPRO-001 R3 FULL task (HANDOFF-VLM-VAD-REPRO-001-R3, gate G4), container VAD_A100_001.
# Usage: GPU_UUID=<uuid> JUDGE_API_BASE=<G1 endpoint> scripts/vad_repro/eval_arms.sh <output_root>
# For each arm in order (base_qwen35_4b, then released_step80): start serve_model.sh, wait for
# /v1/models, run bash scripts/evaluate_official.sh with the Mistral judge and PARALLEL_WORKERS=64,
# stop the server. RUN_DIR is <output_root>/<arm>/run, so a restarted task resumes from the answer
# files. The script copies summary.json, summary.md, the audits and the score, judge and audit logs to
# <output_root>/<arm>/ and writes <output_root>/EVAL_ARMS.txt with the stage times.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
judge_model=mistralai/Mistral-Small-3.2-24B-Instruct-2506
arms=("base_qwen35_4b /workspace/harness/data/vision-opd/Qwen3.5-4B"
      "released_step80 /workspace/harness/data/vad/VAD_for_Qwen3.5-4b")
benchmarks=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
mkdir -p "$out"
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1
log="$out/EVAL_ARMS.txt"
{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== JUDGE_API_BASE ${JUDGE_API_BASE:-<unset>}"
  echo "== GPU_UUID ${GPU_UUID:-<unset>}"
  echo "== df -h /workspace"; df -h /workspace
} >> "$log" 2>&1

# models <name>: print the /v1/models ids of port 8000; exit 0 when <name> is among them.
models() {
  python - "$1" 2>/dev/null <<'EOF'
import json
import sys
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
    ids = [m["id"] for m in json.load(response)["data"]]
print("models", ids)
sys.exit(0 if sys.argv[1] in ids else 1)
EOF
}
start_server() {
  setsid bash "$root/scripts/vad_repro/serve_model.sh" "$1" "$2" >> "$3" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 240); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    models "$2" > /dev/null && return 0
    sleep 5
  done
  return 1
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

status=0
for entry in "${arms[@]}"; do
  read -r arm path <<< "$entry"
  mkdir -p "$out/$arm"
  t0=$(date +%s)
  if ! start_server "$path" "$arm" "$out/$arm/model_server.log"; then
    { echo "arm $arm: model server not ready"; tail -n 80 "$out/$arm/model_server.log"; } >> "$log"
    stop_server
    status=2
    break
  fi
  t1=$(date +%s)
  {
    echo "== arm $arm model $path server ready after $((t1 - t0)) s at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    grep '^COMMAND' "$out/$arm/model_server.log" | tail -n 1
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
  } >> "$log" 2>&1
  MODEL_API_BASE=http://127.0.0.1:8000/v1 MODEL_ID="$arm" MODEL_TAG="$arm" MODEL_PATH="$path" \
    JUDGE_API_BASE="${JUDGE_API_BASE:-}" JUDGE_MODEL="$judge_model" \
    BENCHMARK_DATA_DIR=/workspace/harness/data/vad/benchmarks RUN_DIR="$out/$arm/run" PARALLEL_WORKERS=64 \
    bash scripts/evaluate_official.sh > "$out/$arm/evaluate_official.log" 2>&1
  code=$?
  t2=$(date +%s)
  stop_server
  {
    echo "arm $arm evaluate_official exit $code wall_seconds $((t2 - t1))"
    echo "-- evaluate_official.log tail (progress lines removed)"
    grep -v -E 'it/s\]|s/it\]' "$out/$arm/evaluate_official.log" | tail -n 30
    echo "-- stage end times (log mtime, UTC)"
    for b in "${benchmarks[@]}"; do
      for stage in infer answer_audit judge audit score; do
        f="$out/$arm/run/logs/$b.$stage.log"
        [[ -f "$f" ]] && echo "$b $stage $(date -u -r "$f" +%Y-%m-%dT%H:%M:%SZ)"
      done
    done
  } >> "$log" 2>&1
  cp "$out/$arm/run/summary.json" "$out/$arm/run/summary.md" "$out/$arm/" 2>> "$log"
  mkdir -p "$out/$arm/audits" "$out/$arm/logs"
  cp "$out/$arm/run/audits/"*.json "$out/$arm/audits/" 2>> "$log"
  for b in "${benchmarks[@]}"; do
    for stage in score judge audit answer_audit; do
      f="$out/$arm/run/logs/$b.$stage.log"
      [[ -f "$f" ]] && grep -v -E 'it/s\]|s/it\]' "$f" > "$out/$arm/logs/$b.$stage.log"
    done
  done
  if [[ $code -ne 0 ]]; then status=1; break; fi
done
{ echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ) status $status"; df -h /workspace; } >> "$log" 2>&1
exit "$status"
