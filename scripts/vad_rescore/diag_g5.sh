#!/bin/bash
# DIRECTIVE-VLM-VAD-GRAD-002-001: per-token diagnostic of the G5 failure of TASK-VLM-VAD-GRAD-002-005 (trajectory 2250,
# S80 prompt 281, model A1_80). Three score.py-path runs (one process each), one moments.py-path run, then compare.
# scripts/vad_grad/moments.py, scripts/vad_rollout/a8/ and verl/ are the same at 904097ac and at this commit.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> TRAJ_ROOT=<TASK-GRAD-001-013 output> \
#   FULL_ROOT=<TASK-GRAD-002-005 output> [PROMPT=281] scripts/vad_rescore/diag_g5.sh <output_root>
# Outputs: DIAG_G5.txt (log), DIAG_score_1.json, DIAG_score_2.json, DIAG_score_3.json, DIAG_moments.json, DIAG_G5.json.
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" traj_root="${TRAJ_ROOT:?}" full_root="${FULL_ROOT:?}"
prompt="${PROMPT:-281}"
data="$b200/harness/data"
rollouts="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-053/step80/rollouts.jsonl"
model="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/merged/step_80"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
log="$out/DIAG_G5.txt"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD) prompt $prompt"
  echo "== diff 904097ac..HEAD of the moments.py path: $(git diff --stat 904097ac HEAD -- scripts/vad_grad/moments.py \
    scripts/vad_grad/accum.py scripts/vad_rollout/a8 verl | tail -1)"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
} >> "$log" 2>&1
"$py" scripts/vad_grad/map_command.py "$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1
for k in 1 2 3; do
  "$py" scripts/vad_rescore/diag_prompt.py score "$root" "$out/COMMAND_B200.txt" "$model" "$rollouts" "$prompt" \
    "$out/DIAG_score_$k.json" >> "$log" 2>&1 || { echo "score run $k failed" >> "$log"; exit 1; }
  echo "score run $k exit 0 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
done
"$py" scripts/vad_rescore/diag_prompt.py moments "$root" "$out/COMMAND_B200.txt" "$model" "$rollouts" "$prompt" \
  "$out/DIAG_moments.json" >> "$log" 2>&1 || { echo "moments run failed" >> "$log"; exit 1; }
echo "moments run exit 0 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
"$py" scripts/vad_rescore/diag_prompt.py compare "$traj_root/step80/TRAJ.jsonl" "$full_root/SCORES_S80_A1_80.jsonl" \
  "$out/DIAG_G5.json" "$out"/DIAG_score_1.json "$out"/DIAG_score_2.json "$out"/DIAG_score_3.json "$out/DIAG_moments.json" \
  >> "$log" 2>&1 || { echo "compare failed" >> "$log"; exit 1; }
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
