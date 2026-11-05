#!/bin/bash
# Part B items 2 and 4 (HANDOFF-VLM-VAD-PACK-001-R1, gates G3 to G6): split.py, pack.py run for M0 (A1-S0-k, A2-S0-k
# on the S0 rollouts) and for A1_80 (A1-S80-k on the S80 rollouts), pack.py gram, stats.py. One GPU. A restart resumes
# (PACKROWS files, partial vectors and progress files).
# PILOT=1: 2 micro-batches of each group (pilot.py select), ONE_ROW_CHECK=1, a forced stop after 1 micro-batch of M0 and
# one resume, stats.py with 200 resamples, PILOT_CHECK.json (pilot.py check) and PROJECTION.json (pilot.py project).
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> [PILOT=1] \
#   scripts/vad_pack/run_pack.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}"
pilot="${PILOT:-0}"
commit_s="${COMMIT_SECONDS:-1800}"
data="$b200/harness/data"
r053="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-053"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export COMMIT_SECONDS="$commit_s"
log="$out/RUN_PACK.txt"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD) PILOT $pilot"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  df -B1 "$out"
} >> "$log" 2>&1
"$py" scripts/vad_grad/map_command.py "$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1
"$py" scripts/vad_pack/trainer_values.py "$root" "$out/TRAINER_VALUES.md" >> "$log" 2>&1 || exit 1
[[ -f "$out/split/SPLIT.json" ]] || "$py" scripts/vad_pack/split.py "$r053/step0/rollouts.jsonl" "$r053/step80/rollouts.jsonl" \
  "$labels_root/step0/LABELS.jsonl" "$labels_root/step80/LABELS.jsonl" "$out/split" >> "$log" 2>&1 || exit 1
if [[ "$pilot" == 1 ]]; then
  [[ -f "$out/PILOT_MICRO.json" ]] || "$py" scripts/vad_pack/pilot.py select "$out/split" "$out/PILOT_MICRO.json" >> "$log" 2>&1 || exit 1
  export PILOT_MICRO="$out/PILOT_MICRO.json" ONE_ROW_CHECK=1
fi

pack() {  # <name> <model dir> <rollouts> <batch>...
  local name="$1" dir="$2" rollouts="$3"
  shift 3
  "$py" scripts/vad_pack/pack.py run "$root" "$out/COMMAND_B200.txt" "$dir" "$rollouts" "$out/split" "$out" "$@" \
    >> "$out/pack_$name.log" 2>&1
}
m0="$data/models/Qwen3.5-4B" a1_80="$data/rollout001/TASK-VLM-VAD-ROLLOUT-001-022/merged/step_80"
s0_batches=(A1-S0-1 A1-S0-2 A1-S0-3 A2-S0-1 A2-S0-2 A2-S0-3)
s80_batches=(A1-S80-1 A1-S80-2 A1-S80-3)
if [[ "$pilot" == 1 && ! -f "$out/PACKROWS_A1-S0-1.jsonl" && ! -f "$out/runs.jsonl" ]]; then
  STOP_AFTER_MICRO=1 pack M0 "$m0" "$r053/step0/rollouts.jsonl" "${s0_batches[@]}"
  code=$?
  echo "M0 pilot first run exit $code (75 = forced stop) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  [[ $code -eq 75 || $code -eq 0 ]] || exit 1
fi
pack M0 "$m0" "$r053/step0/rollouts.jsonl" "${s0_batches[@]}"
code=$?
echo "M0 pack exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
[[ $code -eq 0 ]] || exit 1
pack A1_80 "$a1_80" "$r053/step80/rollouts.jsonl" "${s80_batches[@]}"
code=$?
echo "A1_80 pack exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
[[ $code -eq 0 ]] || exit 1
[[ -f "$out/GRAM_PACK.json" ]] || "$py" scripts/vad_pack/pack.py gram "$out" >> "$log" 2>&1 || { echo "gram failed" >> "$log"; exit 1; }
sha256sum -c "$out/GRAM_PACK.sha256" --quiet >> "$log" 2>&1 && echo "GRAM_PACK.sha256 check OK" >> "$log"

resamples=$([[ "$pilot" == 1 ]] && echo 200 || echo 10000)
start=$(date +%s)
"$py" scripts/vad_pack/stats.py "$out" "$out/STATS_PACK.json" "$resamples" >> "$log" 2>&1 || { echo "stats failed" >> "$log"; exit 1; }
stats_s=$(( $(date +%s) - start ))
echo "stats exit 0, resamples $resamples, $stats_s s" >> "$log"
if [[ "$pilot" == 1 ]]; then
  "$py" scripts/vad_pack/pilot.py check "$out" "$out/PILOT_CHECK.json" >> "$log" 2>&1 || exit 1
  "$py" scripts/vad_pack/pilot.py project "$out" "$out/split" "$stats_s" "$resamples" "$commit_s" "$out/PROJECTION.json" \
    >> "$log" 2>&1 || exit 1
fi
echo "temporary files removed: $(find "$out" -name '*.tmp' -print -delete | wc -l)" >> "$log"
echo "vector files left: $(find "$out/vectors" -type f 2>/dev/null | wc -l)" >> "$log"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
