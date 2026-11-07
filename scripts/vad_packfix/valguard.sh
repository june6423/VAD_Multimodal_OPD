#!/bin/bash
# Wall-time limit and worker-death guard of one val_only process (DIRECTIVE-VLM-VAD-PACKFIX-001-009 actions 3 and 4).
# TASK-VLM-VAD-PACKFIX-001-027: the OOM killer stopped a Ray worker in a val_only process, and the process then waited
# 7.5 h in ray.get. This script runs <command> under timeout(1) with <limit> seconds (TERM, KILL 120 s later).
# --foreground keeps the command in the process group of the task, so a harness cancel still stops it; the signal goes
# to the command (train_vad.sh execs the Ray driver), and the Ray processes stop with the driver. Every 5 s the script
# reads the oom_kill count of the container cgroup (memory.events) and the console log of the command. When the count
# rises or the console has a Ray worker death line, it sends TERM to timeout (which sends it to the command, and KILL
# 120 s later) and exits 125.
# Exit: the command exit code, 124 at the wall-time limit, 125 when a worker died.
# Usage: valguard.sh <limit seconds> <console log of the command> <container cgroup dir> <command> [args...]
limit="$1" console="$2" cg="$3"
shift 3
death='A worker died or was killed|actor died unexpectedly'
oom() { awk '$1=="oom_kill"{print $2}' "$cg/memory.events"; }
oom0=$(oom)
timeout --foreground -k 120 "$limit" "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  sleep 5
  reason=""
  if [[ "$(oom)" != "$oom0" ]]; then
    reason="oom_kill $oom0 -> $(oom)"
  elif grep -q -E "$death" "$console"; then
    reason="worker death line: $(grep -m1 -o -E "$death" "$console")"
  fi
  if [[ -n "$reason" ]]; then
    echo "VALGUARD stop: $reason utc $(date -u +%H:%M:%S)"
    kill -TERM "$pid"
    wait "$pid"
    exit 125
  fi
done
wait "$pid"
