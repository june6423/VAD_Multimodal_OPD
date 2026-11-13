#!/bin/bash
# Entry point of the pilot task and the FULL task of VLM-VAD-VIEW-002 (HANDOFF-VLM-VAD-VIEW-002-R1 Part B items 2
# and 4): the host checks of scripts/vad_packfix/run_both.sh (container cgroup, oom_kill, orphan Ray and vLLM
# processes, memwatch.sh, offload_selftest.py), make_viewmap.sh, then run_view2.sh for the arms N1R, N1V8, B and T8
# in this order into <out>/<arm>, then stats.py (STATS_VIEW002.json) and steppeak.py (STEPPEAK.json).
# PILOT=1 (Part A item 7): pilot_rows.py first (16 prompts), stats.py on step 2 with 20 rows, projection.py
# (PROJECTION.json), then the end cleanup (the checkpoints and merged models of the pilot are removed after the checks).
# The FULL task keeps the checkpoints of steps 20 and 40 and the merged models on b200.
# A restarted task skips a finished arm (TRAIN_DONE) and resumes an unfinished arm from its last checkpoint.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [PILOT=1] [JUDGE_GPU_UTIL] [ROLLOUT_GPU_UTIL] \
#   scripts/vad_view2/run_all.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pilot="${PILOT:-0}"
py="${ENV_ROOT:?}/vad/bin/python"
b200="${B200_ROOT:?}"
data="$b200/harness/data"
view1="$b200/harness/outputs/VLM-VAD-VIEW-001/TASK-VLM-VAD-VIEW-001-006"
packfix_b="$b200/harness/outputs/VLM-VAD-PACKFIX-001/TASK-VLM-VAD-PACKFIX-001-026/B"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_ALL.txt"
t0=$(date +%s)
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) PILOT $pilot" >> "$log"
cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
while [[ "$cg" != /sys/fs/cgroup && ( ! -r "$cg/memory.max" || "$(cat "$cg/memory.max")" == max ) ]]; do
  cg="$(dirname "$cg")"
done
oom_kill() { awk '$1=="oom_kill"{print $2}' "$cg/memory.events"; }
echo "container cgroup $cg memory.max $(cat "$cg/memory.max") oom_kill before $(oom_kill)" >> "$log"
orphans=$(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^(ray|VLLM)::/' | wc -l)
echo "orphan ray and vllm processes before the run: $orphans" >> "$log"
[[ "$orphans" -eq 0 ]] || { ps -eo pid,ppid,etime,rss,args | awk '$2==1 && $5 ~ /^(ray|VLLM)::/' >> "$log"; exit 1; }
if [[ -f "$out/memwatch.log" ]]; then
  prev="$out/memwatch_before_$(date -u +%Y%m%dT%H%M%SZ).log"
  mv "$out/memwatch.log" "$prev"
  echo "earlier memwatch.log renamed to $prev sha256 $(sha256sum "$prev" | cut -d' ' -f1)" >> "$log"
fi
export VAD_CONTAINER_CGROUP="$cg" MEMWATCH_FAST_FLAG="$out/MEMWATCH_FAST"
rm -f "$MEMWATCH_FAST_FLAG"
bash scripts/vad_packfix/memwatch.sh "$out/memwatch.log" 400000 "$MEMWATCH_FAST_FLAG" &
watch_pid=$!
memory_summary() {
  sleep 10
  kill "$watch_pid" 2>/dev/null
  {
    echo "oom_kill after $(oom_kill)"
    echo "orphan ray and vllm processes after the run: $(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^(ray|VLLM)::/' | wc -l)"
  } >> "$log"
  PYTHONPATH="$root" "$py" scripts/vad_packfix/steppeak.py "$out" "$out/STEPPEAK.json" >> "$log" 2>&1
  echo "steppeak exit $?" >> "$log"
}
trap memory_summary EXIT
PYTHONPATH="$root" "$py" scripts/vad_packfix/offload_selftest.py "$out/OFFLOAD_SELFTEST.json" >> "$log" 2>&1
code=$?
echo "offload_selftest.py exit $code" >> "$log"
[[ $code -eq 0 ]] || exit 1
bash scripts/vad_view2/make_viewmap.sh "$out/viewmap" >> "$log" 2>&1 || { echo "make_viewmap.sh failed" >> "$log"; exit 1; }
echo "VIEWMAP.json sha256 $(sha256sum "$out/viewmap/VIEWMAP.json" | cut -d' ' -f1)" >> "$log"
export VAD_VIEW_MAP="$out/viewmap/VIEWMAP.json" HELDOUT_JSON="$view1/HELDOUT.json"
export VAL_STEP0_FROM="$packfix_b/validation/0.jsonl"
echo "HELDOUT.json sha256 $(sha256sum "$HELDOUT_JSON" | cut -d' ' -f1) step-0 validation $VAL_STEP0_FROM sha256 $(sha256sum "$VAL_STEP0_FROM" | cut -d' ' -f1)" >> "$log"
if [[ "$pilot" == 1 ]]; then
  vdir=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["views_dir"])' "$VAD_VIEW_MAP")
  [[ -f "$out/pilot_train.parquet" ]] || "$py" scripts/vad_view2/pilot_rows.py "$data/vad/train_vad.parquet" "$vdir" \
    "$out/pilot_train.parquet" "$out/PILOT_ROWS.json" >> "$log" 2>&1 || exit 1
  export PILOT_TRAIN_FILE="$out/pilot_train.parquet" HELDOUT_N=10
fi
for arm in N1R N1V8 B T8; do
  start=$(date +%s)
  ARM="$arm" bash scripts/vad_view2/run_view2.sh "$out/$arm"
  code=$?
  echo "arm $arm exit $code wall_seconds $(( $(date +%s) - start )) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  [[ $code -eq 0 ]] || exit 1
done
if [[ "$pilot" == 1 ]]; then
  "$py" scripts/vad_view2/stats.py "$out" "$out/STATS_VIEW002.json" 2 20 >> "$log" 2>&1 || { echo "stats failed" >> "$log"; exit 1; }
  echo "pilot wall_seconds $(( $(date +%s) - t0 ))" >> "$log"
  "$py" scripts/vad_view2/projection.py "$out" "$packfix_b" "$out/PROJECTION.json" >> "$log" 2>&1 || exit 1
  for arm in N1R N1V8 B T8; do
    echo "arm $arm checkpoints before cleanup: $(ls "$out/$arm/checkpoints" | tr '\n' ' ') $(du -sh "$out/$arm/checkpoints" "$out/$arm/merged" | cut -f1 | tr '\n' ' ')" >> "$log"
    rm -rf "$out/$arm/checkpoints" "$out/$arm/merged"
  done
else
  "$py" scripts/vad_view2/stats.py "$out" "$out/STATS_VIEW002.json" >> "$log" 2>&1 || { echo "stats failed" >> "$log"; exit 1; }
fi
echo "stats exit 0" >> "$log"
echo "temporary files removed: $(find "$out" -name '*.tmp' -print -delete | wc -l)" >> "$log"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ) wall_seconds $(( $(date +%s) - t0 ))" >> "$log"
