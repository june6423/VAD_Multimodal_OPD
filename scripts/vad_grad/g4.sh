#!/bin/bash
# Part B item 4 (HANDOFF-VLM-VAD-GRAD-001-R1, gate G4): g4.py at steps 0, 80, 130 with FLASH_ATTENTION_DETERMINISTIC=1.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> scripts/vad_grad/g4.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}"
data="$b200/harness/data"
py="${ENV_ROOT:?}/vad/bin/python"
r053="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-053"
merged="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/merged"
mkdir -p "$out"
cd "$root" || exit 1
# verl from this checkout, not from the editable install of the vad env (DIRECTIVE-VLM-VAD-GRAD-001-003 Decision 4).
export PATH="${ENV_ROOT}/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR
log="$out/G4.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD)" > "$log"
"$py" scripts/vad_grad/map_command.py "$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1
status=0
for spec in "step0=$data/models/Qwen3.5-4B" "step80=$merged/step_80" "step130=$merged/step_130"; do
  ck="${spec%%=*}" model="${spec#*=}"
  "$py" scripts/vad_grad/g4.py "$root" "$out/COMMAND_B200.txt" "$model" "$r053/$ck/rollouts.jsonl" "$out/G4_$ck.json" \
    >> "$out/g4_$ck.log" 2>&1
  code=$?
  echo "$ck g4 exit $code" >> "$log"
  [[ $code -eq 0 ]] || status=1
done
exit "$status"
