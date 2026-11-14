#!/bin/bash
# Host-memory diagnosis of DIRECTIVE-VLM-VAD-VIEW-002-002 Decision 1 item 3 (failure of TASK-VLM-VAD-VIEW-002-005 in
# arm N1V8, step 3). Arms N1V8 and T8 with the full batch (96 prompts), DIAG_STEPS steps each (default 2) from M0,
# training only (run_view2.sh DIAG_STEPS), with memwatch.sh and memfix.py (<arm>/views_log/memfix.jsonl: driver RSS
# before and after the release of the previous step, after the build, and the teacher tensor bytes of each step).
# Then steppeak.py (STEPPEAK.json). The checkpoints of the diagnosis are removed after each arm.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [DIAG_STEPS=2] [ROLLOUT_GPU_UTIL] \
#   scripts/vad_view2/diag_mem.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="${ENV_ROOT:?}/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_ALL.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) DIAG_STEPS ${DIAG_STEPS:-2}" >> "$log"
cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
while [[ "$cg" != /sys/fs/cgroup && ( ! -r "$cg/memory.max" || "$(cat "$cg/memory.max")" == max ) ]]; do
  cg="$(dirname "$cg")"
done
oom_kill() { awk '$1=="oom_kill"{print $2}' "$cg/memory.events"; }
echo "container cgroup $cg memory.max $(cat "$cg/memory.max") oom_kill before $(oom_kill)" >> "$log"
orphans=$(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^(ray|VLLM)::/' | wc -l)
echo "orphan ray and vllm processes before the run: $orphans" >> "$log"
[[ "$orphans" -eq 0 ]] || exit 1
export VAD_CONTAINER_CGROUP="$cg"
bash scripts/vad_packfix/memwatch.sh "$out/memwatch.log" 100000 &
watch_pid=$!
bash scripts/vad_view2/make_viewmap.sh "$out/viewmap" >> "$log" 2>&1 || { echo "make_viewmap.sh failed" >> "$log"; exit 1; }
export VAD_VIEW_MAP="$out/viewmap/VIEWMAP.json"
echo "VIEWMAP.json sha256 $(sha256sum "$VAD_VIEW_MAP" | cut -d' ' -f1)" >> "$log"
code=0
for arm in N1V8 T8; do
  start=$(date +%s)
  ARM="$arm" DIAG_STEPS="${DIAG_STEPS:-2}" bash scripts/vad_view2/run_view2.sh "$out/$arm"
  c=$?
  echo "arm $arm exit $c wall_seconds $(( $(date +%s) - start )) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  echo "arm $arm checkpoints removed: $(du -sh "$out/$arm/checkpoints" 2>/dev/null | cut -f1)" >> "$log"
  rm -rf "$out/$arm/checkpoints"
  [[ $c -eq 0 ]] || code=1
done
sleep 10
kill "$watch_pid" 2>/dev/null
echo "oom_kill after $(oom_kill)" >> "$log"
echo "orphan ray and vllm processes after the run: $(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^(ray|VLLM)::/' | wc -l)" >> "$log"
PYTHONPATH="$root" "$py" scripts/vad_view2/steppeak.py "$out" "$out/STEPPEAK.json" >> "$log" 2>&1
echo "steppeak exit $?" >> "$log"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $code" >> "$log"
exit "$code"
