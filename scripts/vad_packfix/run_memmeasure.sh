#!/bin/bash
# Memory measurement of step 1 (DIRECTIVE-VLM-VAD-PACKFIX-001-004 actions 2 to 5). Measurement only.
# 1. Stops with exit 1 if an orphan Ray process of an earlier task is alive (action 2; ET removes it by hand).
# 2. Records memory.events oom_kill of the container cgroup before and after the run.
# 3. Runs arm F as the pilot does (run_packfix.sh, PILOT=1) with MEASURE_RUN1_ONLY=1: validation at step 0, step 1
#    with the teacher forward and the first update, the checkpoint save, and the validation at step 1. Then it stops.
# 4. memwatch.sh logs the container charge every 5 s. The Ray object store size and use come from raylet.out.
# Outputs: RUN_MEMMEASURE.txt, MEMPEAK.json, memwatch.log, F/RUN_PACKFIX.txt, F/console_run1.log.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [JUDGE_GPU_UTIL] [ROLLOUT_GPU_UTIL] \
#   scripts/vad_packfix/run_memmeasure.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="${ENV_ROOT:?}/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_MEMMEASURE.txt"
cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
while [[ "$cg" != /sys/fs/cgroup && ( ! -r "$cg/memory.max" || "$(cat "$cg/memory.max")" == max ) ]]; do
  cg="$(dirname "$cg")"
done
oom_kill() { awk '$1=="oom_kill"{print $2}' "$cg/memory.events"; }
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) MEMMEASURE ARM F"
  echo "container cgroup $cg memory.max $(cat "$cg/memory.max") memory.current $(cat "$cg/memory.current")"
  echo "oom_kill before $(oom_kill)"
} >> "$log"
orphans=$(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^ray::/' | wc -l)
echo "orphan ray processes before the run: $orphans" >> "$log"
[[ "$orphans" -eq 0 ]] || { ps -eo pid,ppid,etime,rss,args | awk '$2==1 && $5 ~ /^ray::/' >> "$log"; exit 1; }

bash scripts/vad_packfix/memwatch.sh "$out/memwatch.log" 400000 &
watch_pid=$!
trap 'kill "$watch_pid" 2>/dev/null' EXIT
start=$(date +%s)
ARM=F PILOT=1 MEASURE_RUN1_ONLY=1 bash scripts/vad_packfix/run_packfix.sh "$out/F"
code=$?
sleep 10  # memwatch takes the samples after the end
{
  echo "arm F exit $code wall_seconds $(( $(date +%s) - start ))"
  echo "oom_kill after $(oom_kill)"
  echo "memory.peak $(cat "$cg/memory.peak" 2>/dev/null) (high-water mark since the container start)"
  grep -h "reference model removed" "$out/F/console_run1.log" | sed 's/^.*VAD_HOSTMEM/VAD_HOSTMEM/'
  raylet=$(ls -td /tmp/ray/session_*/logs/raylet.out 2>/dev/null | head -n 1)
  echo "raylet log $raylet"
  grep -h -o "object_store_memory: [0-9.e+]*" "$raylet" | head -n 1
  grep -h -E "Current usage:|bytes spillable|bytes in use" "$raylet" | sort | uniq -c | sort -k3 | tail -n 12
  echo "orphan ray processes after the run: $(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^ray::/' | wc -l)"
} >> "$log" 2>&1
"$py" scripts/vad_packfix/mempeak.py "$out/memwatch.log" "$out/MEMPEAK.json" >> "$log" 2>&1
echo "checkpoints removed: $(ls "$out/F/checkpoints" 2>/dev/null | tr '\n' ' ') $(du -sh "$out/F/checkpoints" 2>/dev/null | cut -f1)" >> "$log"
rm -rf "$out/F/checkpoints"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit 0
