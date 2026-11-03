#!/bin/bash
# Part B item 6 (HANDOFF-VLM-VAD-GRAD-001-R1, G10 inputs): stage 0b displacement.py and stage 0c adam_state.py on CPU.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> scripts/vad_grad/stage0.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${B200_ROOT:?}/harness/data"
py="${ENV_ROOT:?}/vad/bin/python"
r1="$data/rollout001"
a1="$r1/TASK-VLM-VAD-ROLLOUT-001-022" a2="$r1/TASK-VLM-VAD-ROLLOUT-001-036"
mkdir -p "$out"
cd "$root" || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
log="$out/STAGE0.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD)" > "$log"
"$py" scripts/vad_grad/displacement.py "$data/models/Qwen3.5-4B" "$r1/TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS.txt" "$out" \
  "$a1/keep/global_step_80/actor" "$a1/checkpoints/global_step_130/actor" "$a1/merged/step_80" "$a1/merged/step_130" \
  "$a2/keep/global_step_80/actor" "$a2/checkpoints/global_step_130/actor" "$a2/merged/step_80" "$a2/merged/step_130" \
  >> "$log" 2>&1
code=$?
echo "displacement exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
[[ $code -eq 0 ]] || exit 1
"$py" scripts/vad_grad/adam_state.py "$a1/keep/global_step_80/actor/huggingface" "$data/models/Qwen3.5-4B" \
  "$r1/TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS.txt" "$out" \
  "80=$a1/keep/global_step_80/actor=$a2/keep/global_step_80/actor" \
  "130=$a1/checkpoints/global_step_130/actor=$a2/checkpoints/global_step_130/actor" >> "$log" 2>&1
code=$?
echo "adam_state exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit "$code"
