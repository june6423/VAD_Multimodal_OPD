#!/bin/bash
# Part B items 7 and 9 (HANDOFF-VLM-VAD-GRAD-001-R1, gates G6 to G9): moment passes at steps 0, 80, 130 (in this order),
# the cross-check at step 80, and stats.py per checkpoint. One GPU. A restart resumes (moments.py progress files).
# PILOT=1: 4 prompts per checkpoint (pilot.py select), a save after 2 prompts, a forced stop after 3 prompts and one
# resume, 2 cross-check trials, and PROJECTION.json (pilot.py project).
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<label task output> ADAM_ROOT=<stage 0 output> \
#   [PILOT=1] scripts/vad_grad/run_grad.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}" adam_root="${ADAM_ROOT:?}"
pilot="${PILOT:-0}"
data="$b200/harness/data"
r053="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-053"
r056="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-056"
merged="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/merged"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
# verl from this checkout, not from the editable install of the vad env (DIRECTIVE-VLM-VAD-GRAD-001-003 Decision 4).
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR
log="$out/RUN_GRAD.txt"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD) PILOT $pilot"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  grep -E "MemTotal|MemAvailable" /proc/meminfo
  df -B1 "$out"
} >> "$log" 2>&1
"$py" scripts/vad_grad/map_command.py "$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1

model_of() { case "$1" in step0) echo "$data/models/Qwen3.5-4B" ;; step80) echo "$merged/step_80" ;; step130) echo "$merged/step_130" ;; esac; }
adam_of() { case "$1" in step130) echo "$adam_root/V_A1_130.safetensors" ;; *) echo "$adam_root/V_A1_80.safetensors" ;; esac; }

for ck in step0 step80 step130; do
  dir="$out/$ck"
  mkdir -p "$dir"
  args=("$root" "$out/COMMAND_B200.txt" "$(model_of $ck)" "$r053/$ck/rollouts.jsonl" "$labels_root/$ck/LABELS.jsonl"
        "$r056/$ck" "$(adam_of $ck)" "$dir" "$ck")
  if [[ "$pilot" == 1 ]]; then
    [[ -f "$dir/PILOT_PROMPTS.json" ]] || "$py" scripts/vad_grad/pilot.py select "$r053/$ck/rollouts.jsonl" \
      "$labels_root/$ck/LABELS.jsonl" "$dir/PILOT_PROMPTS.json" >> "$log" 2>&1 || exit 1
    args+=("$dir/PILOT_PROMPTS.json")
    if [[ ! -f "$dir/GRAM.json" ]]; then
      SAVE_EVERY=2 STOP_AFTER_PROMPTS=3 "$py" scripts/vad_grad/moments.py "${args[@]}" >> "$dir/moments.log" 2>&1
      code=$?
      echo "$ck pilot first run exit $code (75 = forced stop)" >> "$log"
      [[ $code -eq 75 || $code -eq 0 ]] || exit 1
    fi
  fi
  SAVE_EVERY=$([[ "$pilot" == 1 ]] && echo 2 || echo 25) "$py" scripts/vad_grad/moments.py "${args[@]}" >> "$dir/moments.log" 2>&1
  code=$?
  echo "$ck moments exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  [[ $code -eq 0 ]] || exit 1
  echo "$ck accumulator files left: $(find "$dir" -name '*.safetensors' | wc -l)" >> "$log"
done

trials=$([[ "$pilot" == 1 ]] && echo 2 || echo 16)
if [[ ! -f "$out/CROSSCHECK.json" ]]; then
  "$py" scripts/vad_grad/crosscheck.py "$root" "$out/COMMAND_B200.txt" "$(model_of step80)" "$r053/step80/rollouts.jsonl" \
    "$labels_root/step80/LABELS.jsonl" "$(adam_of step80)" "$out/step80/GRAM.json" "$out/CROSSCHECK.json" "$trials" \
    >> "$out/crosscheck.log" 2>&1 || { echo "crosscheck failed" >> "$log"; exit 1; }
fi
for ck in step0 step80 step130; do
  "$py" scripts/vad_grad/stats.py "$out/$ck/GRAM.json" "$out/STATS_$ck.json" >> "$log" 2>&1 || exit 1
done
if [[ "$pilot" == 1 ]]; then
  "$py" scripts/vad_grad/pilot.py project "$out" "$r053" "$labels_root" "$r056" "$out/PROJECTION.json" >> "$log" 2>&1 || exit 1
fi
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
