#!/bin/bash
# R9 Part B item 4 (G5): gradient passes of each checkpoint on GPU 0-5, then STATS.json per checkpoint.
# Phase 1: teacher check (6 shards per checkpoint). Phase 2: within-prompt per objective, trials per objective and
# pool (pooled; f1 and f0 with "split"), cross-objective. No parameter changes, no stored per-trajectory gradient.
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> \
#   run_grads.sh <A1 COMMAND.txt> <output root> <split|nosplit> <name>=<model dir>=<rollouts jsonl> [...]
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
command_txt="$1"
out="$2"
split="$3"
shift 3
mkdir -p "$out"
export PATH="$env_root/vad/bin:$PATH"
unset VAD_TOKSTATS_DIR
cd "$root" || exit 1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== git HEAD $(git rev-parse HEAD 2>/dev/null)"
  echo "== split $split checkpoints $*"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  grep -E "MemTotal|MemAvailable" /proc/meminfo
} > "$out/GRADS_ENV.txt" 2>&1
pools="pooled"
[[ "$split" == split ]] && pools="pooled f1 f0"
jobs="$out/grad_jobs.tsv"
: > "$jobs"
for spec in "$@"; do
  IFS='=' read -r name model rollouts <<< "$spec"
  mkdir -p "$out/$name"
  base="python3 scripts/vad_rollout/a8/grads.py $root $command_txt $model $rollouts $out/$name"
  for shard in 0 1 2 3 4 5; do
    printf '1\tteacher_%s_%s\t%s teacher %s 6\n' "$name" "$shard" "$base" "$shard" >> "$jobs"
    printf '2\tcross_%s_%s\t%s cross %s 6\n' "$name" "$shard" "$base" "$shard" >> "$jobs"
    for o in O1 O2 O3 O4; do
      printf '2\twithin_%s_%s_%s\t%s within %s %s 6\n' "$name" "$o" "$shard" "$base" "$o" "$shard" >> "$jobs"
      for pool in $pools; do
        printf '2\ttrials_%s_%s_%s_%s\t%s trials %s %s %s 6\n' "$name" "$o" "$pool" "$shard" "$base" "$o" "$pool" "$shard" >> "$jobs"
      done
    done
  done
done
python3 scripts/vad_rollout/a8/schedule.py "$jobs" 0,1,2,3,4,5 "$out/logs" 100
status=$?
echo "schedule exit $status" >> "$out/GRADS_ENV.txt"
[[ $status -eq 0 ]] || exit "$status"
for spec in "$@"; do
  IFS='=' read -r name model rollouts <<< "$spec"
  python3 scripts/vad_rollout/a8/grads.py "$root" "$command_txt" "$model" "$rollouts" "$out/$name" summarize \
    >> "$out/GRADS_ENV.txt" 2>&1 || exit 1
done
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/GRADS_ENV.txt"
