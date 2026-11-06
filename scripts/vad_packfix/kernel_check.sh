#!/bin/bash
# Diagnosis of the G2 failure of TASK-VLM-VAD-PACKFIX-001-001: varlen kernel numerics against row boundaries.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> scripts/vad_packfix/kernel_check.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="${ENV_ROOT:?}/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/KERNEL_CHECK.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD)" >> "$log"
PYTHONPATH="$root" "$py" scripts/vad_packfix/kernel_check.py "${B200_ROOT:?}/harness/data/models/Qwen3.5-4B" \
  "$out/KERNEL_CHECK.json" >> "$log" 2>&1
code=$?
echo "kernel_check.py exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit "$code"
