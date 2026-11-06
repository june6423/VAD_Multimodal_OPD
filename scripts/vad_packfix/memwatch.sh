#!/bin/bash
# Watch-only memory monitor for the PACKFIX memory diagnosis (copy of the ET scratch memwatch.sh used in TASK-009/010).
# Every 5 s for <seconds>: UTC time, the cgroup-root anon and file memory (GiB, as in TASK-009 to TASK-017), the
# container cgroup charge (GiB: memory.current, anon, shmem, file, file_mapped, kernel; memory.events oom_kill),
# the used bytes of /dev/shm (GiB, the Ray object store), host MemAvailable (GiB), GPU memory used (MiB) and
# utilization, and the 5 processes with the largest RSS (GiB).
# The container cgroup is the nearest ancestor of this process cgroup with a memory.max limit (TASK-017: the
# docker scope, 150 GiB). /sys/fs/cgroup/memory.stat is the cgroup root, not the container.
# Usage: bash memwatch.sh <log file> <seconds>
log="$1"
end=$(( $(date +%s) + $2 ))
cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
while [[ "$cg" != /sys/fs/cgroup && ( ! -r "$cg/memory.max" || "$(cat "$cg/memory.max")" == max ) ]]; do
  cg="$(dirname "$cg")"
done
echo "container cgroup $cg memory.max $(cat "$cg/memory.max" 2>/dev/null)" >> "$log"
while [ "$(date +%s)" -lt "$end" ]; do
  {
    printf '%s ' "$(date -u +%H:%M:%S)"
    awk '$1=="anon"{a=$2} $1=="file"{f=$2} END{printf "anon %.1f file %.1f ", a/2^30, f/2^30}' /sys/fs/cgroup/memory.stat
    awk '{printf "ctr_cur %.1f ", $1/2^30}' "$cg/memory.current"
    awk '$1~/^(anon|shmem|file|file_mapped|kernel)$/{printf "ctr_%s %.1f ", $1, $2/2^30}' "$cg/memory.stat"
    awk '$1=="oom_kill"{printf "ctr_oom_kill %s ", $2}' "$cg/memory.events"
    df -B1 --output=used /dev/shm | awk 'NR==2{printf "devshm_used %.1f ", $1/2^30}'  # the Ray object store
    awk '$1=="MemAvailable:"{printf "avail %.1f ", $2/2^20}' /proc/meminfo
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d '\n'
    echo
    ps -eo rss,pid,comm --sort=-rss | sed -n '2,6p' | awk '{printf "  rss %.1f pid %s %s\n", $1/2^20, $2, $3}'
    # DT ANSWER on TASK-VLM-VAD-PACKFIX-001-022 item 5: RSS sum (GiB) and process count of each group. An EngineCore
    # is named by its parent (ray::vLLMHttpSe = rollout, other = judge). RSS counts shared pages in each process.
    ps -eo rss=,pid=,ppid=,comm= | awk '
      { rss[$2] = $1; pp[$2] = $3; c[$2] = $4 }
      END {
        for (p in c) {
          g = c[p]
          if (g == "VLLM::EngineCor") g = "EngineCore(" c[pp[p]] ")"
          else if (g !~ /^ray::(WorkerDict|vLLMHttpSe|TaskRunner|AgentLoopW)$/) {
            if (g ~ /^ray::/) g = "ray::other"; else if (g ~ /^(raylet|gcs_server)$/) g = "ray_system"; else g = "other"
          }
          s[g] += rss[p]; n[g]++
        }
        printf "  groups"
        for (g in s) printf " %s %.1f/%d", g, s[g] / 2^20, n[g]
        printf "\n"
      }'
  } >> "$log" 2>&1
  sleep 5
done
