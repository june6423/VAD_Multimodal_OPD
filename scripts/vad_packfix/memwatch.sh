#!/bin/bash
# Watch-only memory monitor for the PACKFIX memory diagnosis (copy of the ET scratch memwatch.sh used in TASK-009/010).
# Every 5 s for <seconds>: UTC time, container cgroup anon and file memory (GiB), host MemAvailable (GiB),
# GPU memory used (MiB) and utilization, and the 5 processes with the largest RSS (GiB).
# Usage: bash memwatch.sh <log file> <seconds>
log="$1"
end=$(( $(date +%s) + $2 ))
while [ "$(date +%s)" -lt "$end" ]; do
  {
    printf '%s ' "$(date -u +%H:%M:%S)"
    awk '$1=="anon"{a=$2} $1=="file"{f=$2} END{printf "anon %.1f file %.1f ", a/2^30, f/2^30}' /sys/fs/cgroup/memory.stat
    awk '$1=="MemAvailable:"{printf "avail %.1f ", $2/2^20}' /proc/meminfo
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d '\n'
    echo
    ps -eo rss,pid,comm --sort=-rss | sed -n '2,6p' | awk '{printf "  rss %.1f pid %s %s\n", $1/2^20, $2, $3}'
  } >> "$log" 2>&1
  sleep 5
done
