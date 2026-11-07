#!/bin/bash
# Entry point of the pilot task and the FULL task of VLM-VAD-PACKFIX-001 (HANDOFF-VLM-VAD-PACKFIX-001-R6, Part B
# items 2 and 4): run_packfix.sh for arm F into <out>/F, then for arm B into <out>/B, then stats.py. Each arm trains
# without the judge and then runs its judge session (DIRECTIVE-VLM-VAD-PACKFIX-001-007).
# PILOT=1 adds projection.py (PROJECTION.json) and the end cleanup (the pilot checkpoints are removed after the
# checks). The FULL task keeps the checkpoints of steps 10 and 20 on b200. Both write STEPPEAK.json (steppeak.py).
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
# DIRECTIVE-VLM-VAD-PACKFIX-001-004 and -005: no orphan Ray process before the run, oom_kill of the container cgroup
# before and after, and the memory and copy times of each step (steppeak.py, STEPPEAK.json; also after a failed arm).
cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
while [[ "$cg" != /sys/fs/cgroup && ( ! -r "$cg/memory.max" || "$(cat "$cg/memory.max")" == max ) ]]; do
  cg="$(dirname "$cg")"
done
oom_kill() { awk '$1=="oom_kill"{print $2}' "$cg/memory.events"; }
echo "container cgroup $cg memory.max $(cat "$cg/memory.max") oom_kill before $(oom_kill)" >> "$log"
# DIRECTIVE-VLM-VAD-PACKFIX-001-008: the check also counts orphan vLLM processes (TASK-026 left a VLLM::EngineCore).
orphans=$(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^(ray|VLLM)::/' | wc -l)
echo "orphan ray and vllm processes before the run: $orphans" >> "$log"
[[ "$orphans" -eq 0 ]] || { ps -eo pid,ppid,etime,rss,args | awk '$2==1 && $5 ~ /^(ray|VLLM)::/' >> "$log"; exit 1; }
# Host memory against the container limit (DIRECTIVE-VLM-VAD-PACKFIX-001-003): memwatch.sh for the whole task.
# memwatch.log has times of day without a date, so a restarted task first renames the log of the earlier run.
if [[ -f "$out/memwatch.log" ]]; then
  prev="$out/memwatch_before_$(date -u +%Y%m%dT%H%M%SZ).log"
  mv "$out/memwatch.log" "$prev"
  echo "earlier memwatch.log renamed to $prev sha256 $(sha256sum "$prev" | cut -d' ' -f1)" >> "$log"
fi
bash scripts/vad_packfix/memwatch.sh "$out/memwatch.log" 400000 &
watch_pid=$!
memory_summary() {
  sleep 10  # memwatch takes the samples after the end
  kill "$watch_pid" 2>/dev/null
  {
    echo "oom_kill after $(oom_kill)"
    echo "orphan ray and vllm processes after the run: $(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^(ray|VLLM)::/' | wc -l)"
  } >> "$log"
  PYTHONPATH="$root" "$py" scripts/vad_packfix/steppeak.py "$out" "$out/STEPPEAK.json" >> "$log" 2>&1
  echo "steppeak exit $?" >> "$log"
}
trap memory_summary EXIT
# The CUDA checks of hostmem.py (candidates 2, 4, 5, P1 and P2) before the training.
PYTHONPATH="$root" "$py" scripts/vad_packfix/offload_selftest.py "$out/OFFLOAD_SELFTEST.json" >> "$log" 2>&1
code=$?
echo "offload_selftest.py exit $code" >> "$log"
[[ $code -eq 0 ]] || exit 1
for arm in F B; do
  start=$(date +%s)
  # DIRECTIVE-VLM-VAD-PACKFIX-001-007 action 4: the step-0 model is equal in both arms. Arm F validates it once and
  # arm B copies that dump.
  step0=""
  [[ "$arm" == B ]] && step0="$out/F/validation/0.jsonl"
  ARM="$arm" VAL_STEP0_FROM="$step0" bash scripts/vad_packfix/run_packfix.sh "$out/$arm"
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
