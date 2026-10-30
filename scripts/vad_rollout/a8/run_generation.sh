#!/bin/bash
# R9 Part B item 3 (G4): rollouts of each checkpoint on GPU 0-5 (6 shards per checkpoint), then one merged file
# and GEN_SUMMARY.json per checkpoint. It trains nothing and changes no A1 or A2 file.
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> \
#   run_generation.sh <A1 COMMAND.txt> <output root> <name>=<model dir> [<name>=<model dir> ...]
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
command_txt="$1"
out="$2"
shift 2
mkdir -p "$out"
export PATH="$env_root/vad/bin:$PATH"
unset VAD_TOKSTATS_DIR
cd "$root" || exit 1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== git HEAD $(git rev-parse HEAD 2>/dev/null)"
  echo "== checkpoints $*"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  grep -E "MemTotal|MemAvailable" /proc/meminfo
} > "$out/GEN_ENV.txt" 2>&1
jobs="$out/gen_jobs.tsv"
: > "$jobs"
for spec in "$@"; do
  name="${spec%%=*}"
  model="${spec#*=}"
  mkdir -p "$out/$name"
  for shard in 0 1 2 3 4 5; do
    printf '1\tgen_%s_%s\tpython3 scripts/vad_rollout/a8/generate.py %s %s %s %s %s 6\n' \
      "$name" "$shard" "$root" "$command_txt" "$model" "$out/$name/shard_$shard.jsonl" "$shard" >> "$jobs"
  done
done
python3 scripts/vad_rollout/a8/schedule.py "$jobs" 0,1,2,3,4,5 "$out/logs" 100
status=$?
echo "schedule exit $status" >> "$out/GEN_ENV.txt"
[[ $status -eq 0 ]] || exit "$status"
for spec in "$@"; do
  name="${spec%%=*}"
  python3 scripts/vad_rollout/a8/merge_rollouts.py "$out/$name/rollouts.jsonl" "$out/$name/GEN_SUMMARY.json" \
    "$out/$name"/shard_*.jsonl >> "$out/GEN_ENV.txt" 2>&1 || exit 1
done
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/GEN_ENV.txt"
