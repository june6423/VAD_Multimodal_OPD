#!/bin/bash
# VLM-VAD-ROLLOUT-001 check of the step-80 keep copy of a training arm (DIRECTIVE-VLM-VAD-ROLLOUT-001-007 item 6).
# Usage: scripts/vad_rollout/keep_check.sh <arm output_root> <output_root>
# Writes <output_root>/KEEP_CHECK.txt: keep/KEEP.json without its per-file SHA-256 map, the file count and bytes
# of keep/global_step_80, latest_checkpointed_iteration.txt, the size of every checkpoints/global_step_* folder,
# the last 20 lines of ckpt_keep.log, df -h of the arm output root, and the cgroup memory limit and usage of the
# host (DIRECTIVE-VLM-VAD-ROLLOUT-001-010, DT answer of 2026-10-30). It changes no file of the arm. No GPU.
# Exit 0 only when KEEP.json has status COPIED and verified true.
set -uo pipefail

arm="$1"
out="$2"
mkdir -p "$out"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== $arm/keep/KEEP.json (sha256 map omitted)"
  python3 - "$arm/keep/KEEP.json" <<'EOF'
import json
import sys

try:
    record = json.load(open(sys.argv[1]))
except FileNotFoundError:
    print("KEEP.json missing")
    sys.exit(0)
record.pop("sha256", None)
print(json.dumps(record, indent=1, sort_keys=True))
EOF
  echo "== keep/global_step_80 files $(find "$arm/keep/global_step_80" -type f 2>/dev/null | wc -l)" \
    "bytes $(du -sb "$arm/keep/global_step_80" 2>/dev/null | cut -f1)"
  echo "== latest_checkpointed_iteration $(cat "$arm/checkpoints/latest_checkpointed_iteration.txt")"
  echo "== checkpoints"
  du -sh "$arm"/checkpoints/global_step_*
  echo "== ckpt_keep.log tail"
  tail -n 20 "$arm/ckpt_keep.log"
  echo "== df -h"
  df -h "$arm"
  echo "== cgroup memory (v2 memory.max, memory.current; v1 memory.limit_in_bytes, memory.usage_in_bytes)"
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.limit_in_bytes \
    /sys/fs/cgroup/memory/memory.usage_in_bytes; do
    [[ -r "$f" ]] && echo "$f $(cat "$f")"
  done
  grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
} > "$out/KEEP_CHECK.txt" 2>&1
cat "$out/KEEP_CHECK.txt"
grep -q '"status": "COPIED"' "$out/KEEP_CHECK.txt" && grep -q '"verified": true' "$out/KEEP_CHECK.txt"
