#!/bin/bash
# PACKFIX-001 judge server: a copy of scripts/vad_rollout/judge_serve.sh (the A1 judge settings) with one change.
# The vLLM memory share is JUDGE_GPU_UTIL instead of 0.85, because the judge and the training share the one B200
# (HANDOFF-VLM-VAD-PACKFIX-001-R2 Definition 4). Model, weights, precision (vLLM default for the snapshot), tensor
# parallel size 1 and max model len 16384 stay as in A1. The prompt and the sampling settings come from the reward
# function of the training code, not from this script.
# Usage: CUDA_VISIBLE_DEVICES=<GPU> JUDGE_MODEL_DIR=<Mistral snapshot> JUDGE_PYTHON=<judge env python> \
#   JUDGE_GPU_UTIL=<share> [JUDGE_PORT=<port>] scripts/vad_packfix/judge_serve.sh <output_root>
# It writes <output_root>/READY.json when /v1/models lists the served name. Every 30 s it checks for
# <output_root>/STOP. When the file exists, it sends TERM to the server, waits up to 60 s (then KILL), and exits 0.
# Records: JUDGE_SERVE.txt and server.log.
set -uo pipefail

out="$1"
model_dir="${JUDGE_MODEL_DIR:?JUDGE_MODEL_DIR must name the Mistral snapshot directory}"
python="${JUDGE_PYTHON:?JUDGE_PYTHON must name the python of the judge environment}"
util="${JUDGE_GPU_UTIL:?JUDGE_GPU_UTIL must give the vLLM memory share of the judge}"
name=mistralai/Mistral-Small-3.2-24B-Instruct-2506
port="${JUDGE_PORT:-8900}"
mkdir -p "$out"
log="$out/JUDGE_SERVE.txt"
rm -f "$out/READY.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total --format=csv 2>&1
  echo "== host IP (hostname -I)"; hostname -I
  echo "== vllm version"; "$python" -c 'import vllm; print(vllm.__version__)'
} >> "$log" 2>&1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must name the judge GPU" >> "$log"
  exit 2
fi

bin_dir="$(dirname "$python")"
if [[ -x "$bin_dir/vllm" ]]; then serve=("$bin_dir/vllm" serve); else serve=("$python" -m vllm.entrypoints.cli.main serve); fi
cmd=("${serve[@]}" "$model_dir" --served-model-name "$name" --host 127.0.0.1 --port "$port"
  --tensor-parallel-size 1 --gpu-memory-utilization "$util" --max-model-len 16384)
{ echo "== command"; printf '%q ' "${cmd[@]}"; echo; } >> "$log"

start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"${cmd[@]}" >> "$out/server.log" 2>&1 &
pid=$!

# stop_server: TERM, wait up to 60 s, then KILL. The judge must not hold the GPU after a stop.
stop_server() {
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "server did not end within 60 s after TERM; KILL" >> "$log"
    kill -KILL "$pid" 2>/dev/null
  fi
  wait "$pid" 2>/dev/null
}

ready=0
for _ in $(seq 1 480); do
  kill -0 "$pid" 2>/dev/null || break
  if [[ -e "$out/STOP" ]]; then
    { echo "STOP file found before READY at $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } >> "$log"
    stop_server
    exit 0
  fi
  if "$python" - "$port" "$name" 2>/dev/null <<'EOF'
import json
import sys
from urllib.request import urlopen

with urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as response:
    sys.exit(0 if sys.argv[2] in [m["id"] for m in json.load(response)["data"]] else 1)
EOF
  then ready=1; break; fi
  sleep 5
done
if [[ $ready -ne 1 ]]; then
  { echo "judge not ready; server.log tail:"; tail -n 80 "$out/server.log"; } >> "$log"
  stop_server
  exit 1
fi

"$python" - "$out/READY.json" "$model_dir" "$name" "$port" "$start" "${CUDA_VISIBLE_DEVICES}" "$pid" "$util" <<'EOF'
import json
import sys
from datetime import datetime, timezone

path, model_dir, name, port, start, devices, pid, util = sys.argv[1:]
json.dump({"model_path": model_dir, "served_name": name, "port": int(port), "start_time": start,
           "ready_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "cuda_visible_devices": devices,
           "pid": int(pid), "tensor_parallel_size": 1, "gpu_memory_utilization": float(util),
           "max_model_len": 16384}, open(path, "w"), indent=1)
EOF
{ echo "== READY"; cat "$out/READY.json"; echo; nvidia-smi --query-gpu=index,uuid,memory.used --format=csv 2>&1; } >> "$log"

while kill -0 "$pid" 2>/dev/null; do
  if [[ -e "$out/STOP" ]]; then
    echo "STOP file found at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
    stop_server
    echo "server stopped at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
    exit 0
  fi
  sleep 30
done
wait "$pid"
status=$?
{ echo "server exit $status at $(date -u +%Y-%m-%dT%H:%M:%SZ)"; tail -n 40 "$out/server.log"; } >> "$log"
exit "$status"
