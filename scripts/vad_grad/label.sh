#!/bin/bash
# Part B item 5 (HANDOFF-VLM-VAD-GRAD-001-R1, gate G5): judge server, labels of the TASK-053 trajectories, judge stop.
# The judge (scripts/vad_rollout/judge_serve.sh, the TASK-VLM-VAD-ROLLOUT-001-011 settings) and label.py run in this
# task only; the STOP file ends the judge before the task ends, so the judge never runs with a gradient task.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> scripts/vad_grad/label.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?B200_ROOT must name the volume root}"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
data="$b200/harness/data"
r053="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-053"
mkdir -p "$out/judge"
cd "$root" || exit 1
log="$out/LABEL_ENV.txt"
export PYTHONDONTWRITEBYTECODE=1
"$env_root/vad/bin/python" scripts/vad_grad/map_command.py "$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" \
  "$out/COMMAND_B200.txt" "$root" "$b200/harness" > "$log" 2>&1 || exit 1
# Free-port probe (fallback table: port in use).
port=$("$env_root/vad/bin/python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
{ echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "== git HEAD $(git rev-parse HEAD)"; echo "== judge port $port"; } >> "$log"

CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 JUDGE_PORT="$port" \
  JUDGE_MODEL_DIR="$data/models/Mistral-Small-3.2-24B-Instruct-2506" JUDGE_PYTHON="$env_root/judge/bin/python" \
  bash scripts/vad_rollout/judge_serve.sh "$out/judge" &
judge_pid=$!
ready=0
for _ in $(seq 1 360); do
  [[ -e "$out/judge/READY.json" ]] && { ready=1; break; }
  kill -0 "$judge_pid" 2>/dev/null || break
  sleep 5
done
echo "== judge ready $ready at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
status=1
if [[ $ready -eq 1 ]]; then
  "$env_root/vad/bin/python" scripts/vad_grad/label.py "$root" "$out/COMMAND_B200.txt" "$out" "http://127.0.0.1:$port/v1" \
    "step0=$r053/step0/rollouts.jsonl" "step80=$r053/step80/rollouts.jsonl" "step130=$r053/step130/rollouts.jsonl" \
    > "$out/label.log" 2>&1
  status=$?
  echo "== label.py exit $status at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
fi
touch "$out/judge/STOP"
wait "$judge_pid"
echo "== judge exit $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
nvidia-smi --query-gpu=index,memory.used --format=csv >> "$log" 2>&1
tail -n 20 "$out/label.log" >> "$log"
exit "$status"
