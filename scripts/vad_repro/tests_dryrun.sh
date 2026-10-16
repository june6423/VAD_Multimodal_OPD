#!/bin/bash
# VLM-VAD-REPRO-001 Part B task 4 (HANDOFF-VLM-VAD-REPRO-001-R1, gate G5). No GPU work.
# Usage: scripts/vad_repro/tests_dryrun.sh <output_root>
# Runs the test suite and the 4b launcher dry run. The record is <output_root>/TESTS_DRYRUN.txt.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
mkdir -p "$out"
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1
log="$out/TESTS_DRYRUN.txt"

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== python -m pytest tests -q"
  python -m pytest tests -q -p no:cacheprovider
  t=$?
  echo "pytest exit $t"
  echo "== MODEL_PATH=/workspace/harness/data/vision-opd/Qwen3.5-4B bash scripts/train_vad.sh --profile 4b --dry-run"
  MODEL_PATH=/workspace/harness/data/vision-opd/Qwen3.5-4B bash scripts/train_vad.sh --profile 4b --dry-run
  d=$?
  echo "dry_run exit $d"
} > "$log" 2>&1

grep -q "^pytest exit 0$" "$log" && grep -q "^dry_run exit 0$" "$log"
