#!/bin/bash
# VLM-VAD-ROLLOUT-001 judge READY record (HANDOFF-VLM-VAD-ROLLOUT-001-R4 gate G3, REPORT-VLM-VAD-ROLLOUT-001-002
# proposal 2). Usage: ENV_ROOT=<environment root> [JUDGE_API_BASE=<endpoint>] [JUDGE_GPU=<index>] \
#   scripts/vad_rollout/judge_ready_read.sh <judge task output_root> <output_root>
# Prints READY.json of the running judge_serve.sh task, the /v1/models reply of the endpoint, and nvidia-smi
# of the judge GPU into <output_root>/JUDGE_READY.txt. No GPU allocation.
set -uo pipefail

judge_out="$1"
out="$2"
python="${ENV_ROOT:?ENV_ROOT must name the environment root}/vad/bin/python"
api="${JUDGE_API_BASE:-http://127.0.0.1:8900/v1}"
gpu="${JUDGE_GPU:-6}"
mkdir -p "$out"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== $judge_out/READY.json"
  cat "$judge_out/READY.json"
  echo
  echo "== $api/models"
  "$python" - "$api" <<'EOF'
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1] + "/models", timeout=10) as response:
    print(response.read().decode())
EOF
  echo "== nvidia-smi GPU $gpu"
  nvidia-smi -i "$gpu" --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv
} > "$out/JUDGE_READY.txt" 2>&1
grep -q '"served_name"' "$out/JUDGE_READY.txt"
