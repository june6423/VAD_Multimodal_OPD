#!/bin/bash
# VLM-VAD-ROLLOUT-001 read-only host memory sampler (DIRECTIVE-VLM-VAD-ROLLOUT-001-010).
# Usage: [ENV_ROOT=<environment root>] scripts/vad_rollout/host_mem_sample.sh <training console.log> <duration s> \
#   <interval s> <output_root>
# Every <interval> seconds for <duration> seconds, one sample goes to <output_root>/HOSTMEM_H<hour>.txt (hour since
# the start) and to stdout: the UTC time, the training state from the console log (the last "step:N" number, or
# "validation after step N" when a validation batch line follows it), MemTotal, MemAvailable, MemFree, Cached, Shmem,
# AnonPages and SReclaimable of /proc/meminfo, the used bytes of /dev/shm, and the 15 processes with the largest RSS
# (PID, PPID, RSS kB, PSS kB from /proc/<pid>/smaps_rollup, name, the first 80 characters of the command line with
# key, token, secret and password values replaced by <redacted>). Every 5 minutes a sample also holds "ray status"
# when it ends within 10 s. Each sample also adds one line to <output_root>/HOSTMEM_SUMMARY.csv.
# Read only: the script sends no signal and writes only under <output_root>. No GPU.
set -uo pipefail

log="$1"
duration="$2"
interval="$3"
out="$4"
ray="${ENV_ROOT:-$HOME/vad-rollout/envs}/vad/bin/ray"
fields=(MemTotal MemAvailable MemFree Cached Shmem AnonPages SReclaimable)
mkdir -p "$out"
summary="$out/HOSTMEM_SUMMARY.csv"
echo "utc,state,$(IFS=,; echo "${fields[*]/%/_kB}"),shm_used_B" > "$summary"

redact() { sed -E 's/((key|token|secret|password)[A-Za-z0-9_.-]*[=: ]+)[^ ]+/\1<redacted>/Ig'; }

training_state() {
  local step_hit val_hit step_line val_line
  step_hit=$(grep -n -o -E 'step:[0-9]+ - ' "$log" 2>/dev/null | tail -n 1)
  val_hit=$(grep -n -o "'validate': True" "$log" 2>/dev/null | tail -n 1)
  step_line=${step_hit%%:*}
  val_line=${val_hit%%:*}
  local step
  step=$(sed -E 's/.*step:([0-9]+) - /\1/' <<< "$step_hit")
  if [[ ${val_line:-0} -gt ${step_line:-0} ]]; then
    echo "validation after step ${step:-none}"
  else
    echo "step ${step:-none}"
  fi
}

start=$(date +%s)
last_ray=0
samples=0
while [[ $(date +%s) -lt $((start + duration)) ]]; do
  now=$(date +%s)
  utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  state=$(training_state)
  mem=$(awk -v names="${fields[*]}" 'BEGIN { n = split(names, order, " ") }
    { key = $1; sub(":", "", key); value[key] = $2 }
    END { for (i = 1; i <= n; i++) printf "%s%s", value[order[i]], (i < n ? "," : "") }' /proc/meminfo)
  shm=$(df -B1 --output=used /dev/shm | tail -n 1 | tr -d ' ')
  echo "$utc,$state,$mem,$shm" >> "$summary"
  do_ray=0
  if [[ $((now - last_ray)) -ge 300 ]]; then
    do_ray=1
    last_ray=$now
  fi
  {
    echo "== $utc state=$state shm_used_B=$shm"
    echo "meminfo_kB $(paste -d= <(printf '%s\n' "${fields[@]}") <(tr ',' '\n' <<< "$mem") | tr '\n' ' ')"
    echo "pid ppid rss_kB pss_kB name cmdline"
    ps -eo pid=,ppid=,rss=,comm= --sort=-rss | head -n 15 | while read -r pid ppid rss name; do
      pss=$(awk '/^Pss:/ { print $2 }' "/proc/$pid/smaps_rollup" 2>/dev/null)
      cmd=$({ tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null | cut -c1-80 | redact)
      echo "$pid $ppid $rss ${pss:-na} $name $cmd"
    done
    if [[ $do_ray -eq 1 ]]; then
      echo "-- ray status"
      timeout 10 "$ray" status 2>&1 | redact
      echo "-- ray status exit ${PIPESTATUS[0]}"
    fi
  } | tee -a "$out/HOSTMEM_H$(((now - start) / 3600)).txt"
  samples=$((samples + 1))
  sleep "$interval"
done
echo "done samples $samples at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
