#!/bin/bash
# Entry point of the pilot task and the FULL task of VLM-VAD-PACKFIX-001 (HANDOFF-VLM-VAD-PACKFIX-001-R2, Part B
# items 2 and 4): run_packfix.sh for arm F into <out>/F, then for arm B into <out>/B, then stats.py.
# PILOT=1 adds projection.py (PROJECTION.json) and the end cleanup (the pilot checkpoints are removed after the
# checks). The FULL task keeps the checkpoints of steps 10 and 20 on b200.
# A restarted task skips a finished arm (TRAIN_DONE) and resumes an unfinished arm from its last checkpoint.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [PILOT=1] [JUDGE_GPU_UTIL] [ROLLOUT_GPU_UTIL] \
#   scripts/vad_packfix/run_both.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pilot="${PILOT:-0}"
py="${ENV_ROOT:?}/vad/bin/python"
data="${B200_ROOT:?}/harness/data"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_BOTH.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) PILOT $pilot" >> "$log"
# Host memory against the container limit (DIRECTIVE-VLM-VAD-PACKFIX-001-003): memwatch.sh for the whole task.
bash scripts/vad_packfix/memwatch.sh "$out/memwatch.log" 400000 &
watch_pid=$!
trap 'kill "$watch_pid" 2>/dev/null' EXIT
for arm in F B; do
  start=$(date +%s)
  ARM="$arm" bash scripts/vad_packfix/run_packfix.sh "$out/$arm"
  code=$?
  echo "arm $arm exit $code wall_seconds $(( $(date +%s) - start )) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  [[ $code -eq 0 ]] || exit 1
done
"$py" scripts/vad_packfix/stats.py "$out/F" "$out/B" scripts/vad_packfix/sources/A1_STEP_METRICS_1_20.json \
  "$out/STATS_PACKFIX.json" >> "$log" 2>&1 || { echo "stats failed" >> "$log"; exit 1; }
echo "stats exit 0" >> "$log"
if [[ "$pilot" == 1 ]]; then
  rows=$("$py" scripts/vad_packfix/count_rows.py "$data/vad/benchmarks/opd_fast_benchmarks.parquet")
  "$py" scripts/vad_packfix/projection.py "$out" "$rows" "$out/PROJECTION.json" >> "$log" 2>&1 || exit 1
  for arm in F B; do
    echo "arm $arm checkpoints before cleanup: $(ls "$out/$arm/checkpoints" | tr '\n' ' ') $(du -sh "$out/$arm/checkpoints" | cut -f1)" >> "$log"
    rm -rf "$out/$arm/checkpoints"
  done
fi
echo "temporary files removed: $(find "$out" -name '*.tmp' -print -delete | wc -l)" >> "$log"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
