#!/bin/bash
# DIRECTIVE-VLM-VAD-GRAD-003-005: fp32 part-sum diagnostic at M0 for the micro-batches of
# TASK-VLM-VAD-GRAD-003-004 (B8-f1 and B1-f1-j0, diag_partsum.pick_micro), with forward O (VAD_PACKFIX unset) and
# forward Fx (VAD_PACKFIX=1), one process each.
# Outputs: DIAG_FP32_O.json, DIAG_FP32_Fx.json, DIAG_FP32.txt (log), rows/ (batches.py row files of M0).
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> \
#   scripts/vad_grad3/diag_fp32.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}"
data="$b200/harness/data"
r1="$data/rollout001"
r053="$r1/TASK-VLM-VAD-ROLLOUT-001-053"
g3="$r1/TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS.txt"
py="$env_root/vad/bin/python"
mkdir -p "$out/rows"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR VAD_PACKFIX
log="$out/DIAG_FP32.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD)" >> "$log"
"$py" scripts/vad_grad3/map_command.py "$r1/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1
ro="$r053/step0/rollouts.jsonl" lab="$labels_root/step0/LABELS.jsonl"
"$py" scripts/vad_grad3/batches.py "$ro" "$lab" "$out/rows" >> "$log" 2>&1 || exit 1
args=("$root" "$out/COMMAND_B200.txt" "$data/models/Qwen3.5-4B" "$ro" "$out/rows" "$g3")
diag() {  # diag <mode>: all rows of the micro-batch; after a GPU out-of-memory error, 2 rows with both labels
  local mode="$1" fx=() code
  [[ "$mode" == Fx ]] && fx=(env VAD_PACKFIX=1)
  "${fx[@]}" "$py" scripts/vad_grad3/diag_fp32.py "${args[@]}" "$out/DIAG_FP32_$mode.json" "$mode" B8-f1 B1-f1-j0 \
    > "$out/diag_$mode.log" 2>&1
  code=$?
  cat "$out/diag_$mode.log" >> "$log"
  echo "diag $mode exit $code" >> "$log"
  if [[ $code -ne 0 ]] && grep -q OutOfMemoryError "$out/diag_$mode.log"; then
    MAX_ROWS=2 "${fx[@]}" "$py" scripts/vad_grad3/diag_fp32.py "${args[@]}" "$out/DIAG_FP32_$mode.json" "$mode" \
      B8-f1 B1-f1-j0 >> "$log" 2>&1
    echo "diag $mode MAX_ROWS=2 exit $?" >> "$log"
  fi
}
diag O
diag Fx
[[ -f "$out/DIAG_FP32_O.json" && -f "$out/DIAG_FP32_Fx.json" ]]
