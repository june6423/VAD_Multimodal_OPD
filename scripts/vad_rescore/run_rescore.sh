#!/bin/bash
# Part B items 2 and 4 (HANDOFF-VLM-VAD-GRAD-002-R1, gates G3 to G6): score.py for the 5 models (S0 under M0, A1_80,
# A1_130, A2_80, A2_130; S80 under the 4 trained models), then stats.py. One GPU. A restart resumes (progress files).
# PILOT=1: 4 prompts per set (pilot.py select), a forced stop after 2 prompts of M0 and one resume, stats.py with 200
# resamples, PROJECTION.json (pilot.py project).
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> \
#   TRAJ_ROOT=<TASK-GRAD-001-013 output> [PILOT=1] scripts/vad_rescore/run_rescore.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}" traj_root="${TRAJ_ROOT:?}"
pilot="${PILOT:-0}"
data="$b200/harness/data"
r053="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-053"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
log="$out/RUN_RESCORE.txt"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD) PILOT $pilot"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  df -B1 "$out"
} >> "$log" 2>&1
"$py" scripts/vad_grad/map_command.py "$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1

spec() {  # <set name> <checkpoint>
  local s="$1=$r053/$2/rollouts.jsonl=$labels_root/$2/LABELS.jsonl=$traj_root/$2/ANSWER_POS.jsonl"
  [[ "$pilot" == 1 ]] && s="$s=$out/PILOT_PROMPTS_$1.json"
  echo "$s"
}
if [[ "$pilot" == 1 ]]; then
  for pair in S0:step0 S80:step80; do
    s="${pair%%:*}" ck="${pair##*:}"
    [[ -f "$out/PILOT_PROMPTS_$s.json" ]] || "$py" scripts/vad_rescore/pilot.py select "$r053/$ck/rollouts.jsonl" \
      "$labels_root/$ck/LABELS.jsonl" "$out/PILOT_PROMPTS_$s.json" >> "$log" 2>&1 || exit 1
  done
fi

a1="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/merged" a2="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-036/merged"
models=("M0=$data/models/Qwen3.5-4B" "A1_80=$a1/step_80" "A1_130=$a1/step_130" "A2_80=$a2/step_80" "A2_130=$a2/step_130")
for entry in "${models[@]}"; do
  name="${entry%%=*}" dir="${entry#*=}"
  sets=("$(spec S0 step0)")
  [[ "$name" != M0 ]] && sets+=("$(spec S80 step80)")
  if [[ "$pilot" == 1 && "$name" == M0 && ! -f "$out/SCORES_S0_M0.jsonl" ]]; then
    STOP_AFTER_PROMPTS=2 "$py" scripts/vad_rescore/score.py "$root" "$out/COMMAND_B200.txt" "$dir" "$name" "$out" \
      "${sets[@]}" >> "$out/score_$name.log" 2>&1
    code=$?
    echo "$name pilot first run exit $code (75 = forced stop) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
    [[ $code -eq 75 || $code -eq 0 ]] || exit 1
  fi
  "$py" scripts/vad_rescore/score.py "$root" "$out/COMMAND_B200.txt" "$dir" "$name" "$out" "${sets[@]}" \
    >> "$out/score_$name.log" 2>&1
  code=$?
  echo "$name score exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  [[ $code -eq 0 ]] || exit 1
done

resamples=$([[ "$pilot" == 1 ]] && echo 200 || echo 10000)
start=$(date +%s)
"$py" scripts/vad_rescore/stats.py "$out" "$traj_root/step0/TRAJ.jsonl" "$traj_root/step80/TRAJ.jsonl" \
  "$out/STATS_RESCORE.json" "$resamples" >> "$log" 2>&1 || { echo "stats failed" >> "$log"; exit 1; }
stats_s=$(( $(date +%s) - start ))
echo "stats exit 0, resamples $resamples, $stats_s s" >> "$log"
if [[ "$pilot" == 1 ]]; then
  "$py" scripts/vad_rescore/pilot.py project "$out" "$r053/step0/rollouts.jsonl" "$r053/step80/rollouts.jsonl" \
    "$stats_s" "$resamples" "$out/PROJECTION.json" >> "$log" 2>&1 || exit 1
fi
echo "temporary files removed: $(find "$out" -name '*.tmp' -print -delete | wc -l)" >> "$log"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
