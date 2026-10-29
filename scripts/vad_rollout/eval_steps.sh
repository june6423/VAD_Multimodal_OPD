#!/bin/bash
# VLM-VAD-ROLLOUT-001 evaluation of one arm (HANDOFF-VLM-VAD-ROLLOUT-001-R4, Part A item 1, gate G6).
# Usage: GPU_UUID=<uuid of the evaluation GPU> JUDGE_API_BASE=<judge endpoint> DATA_ROOT=<data root> \
#   ENV_ROOT=<environment root> scripts/vad_rollout/eval_steps.sh <arm> <train output root> <output root>
# Runs scripts/vad_rollout/eval_checkpoint.sh for <train output root>/merged/step_80 and then
# merged/step_130, into <output root>/step_80 and <output root>/step_130. It stops at the first non-zero
# exit. A restarted task resumes through the RUN_DIR of each step. The record is <output root>/EVAL_STEPS.txt.
set -uo pipefail

arm="$1"
train="$2"
out="$3"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$out"
for step in 80 130; do
  start=$(date +%s)
  bash "$root/scripts/vad_rollout/eval_checkpoint.sh" "$arm" "$step" "$train/merged/step_$step" "$out/step_$step"
  code=$?
  echo "step $step eval_checkpoint exit $code wall_seconds $(( $(date +%s) - start )) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/EVAL_STEPS.txt"
  [[ $code -eq 0 ]] || exit "$code"
done
