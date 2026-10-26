#!/bin/bash
# VLM-VAD-ROLLOUT-001 training arm (HANDOFF-VLM-VAD-ROLLOUT-001-R1, Part A item 3, gate G4).
# Usage: CUDA_VISIBLE_DEVICES=<6 training GPUs> DATA_ROOT=<data root> ENV_ROOT=<environment root> \
#   WANDB_ENV_FILE=<file that sets WANDB_API_KEY> [JUDGE_API_BASE=<judge endpoint>] \
#   scripts/vad_rollout/train_arm.sh <A0|A1|A2> <output_root> [--dry-run]
# Runs bash scripts/train_vad.sh --profile 4b with environment overrides only. The differences from the
# train_vad.sh defaults are these and nothing else: CUDA_VISIBLE_DEVICES, TRAINER_N_GPUS_PER_NODE=6,
# MODEL_PATH and TEACHER_MODEL_PATH (local Qwen3.5-4B), TASK_TRAIN_FILE and BENCHMARK_EVAL_FILE (local
# data), EXPERIMENT_NAME=<arm>, TRAINER_DEFAULT_LOCAL_DIR and TRAINER_ROLLOUT_DATA_DIR under <output_root>,
# TRAINER_MAX_ACTOR_CKPT_TO_KEEP=3, BENCHMARK_EVAL_JUDGE_API_BASE and BENCHMARK_EVAL_JUDGE_MODEL (Mistral
# judge), WANDB_API_KEY from $WANDB_ENV_FILE, VAD_TOKSTATS_DIR=<output_root>/tokstats, ROLLOUT_N=1 for A2.
# Arm A0 also takes TOTAL_TRAINING_STEPS and TRAINER_SAVE_FREQ from the environment (2 and 2 for the smoke).
# --dry-run passes --dry-run to train_vad.sh and prints the overrides (secret-named variables as <set> or
# <unset>), the resolved values, and the main_ppo argument list of train_vad.sh lines 242-357.
# A real run starts the keep hook (ckpt_keep.py watch), runs train_vad.sh, writes STEP_METRICS.json,
# runs scripts/merge_checkpoint.sh on step 80 and on the final step, then the cleanup (ckpt_keep.py cleanup).
# A restarted task continues through trainer.resume_mode=auto (train_vad.sh line 354).
set -uo pipefail

arm="$1"
out="$2"
dry_run=0
[[ "${3:-}" == "--dry-run" ]] && dry_run=1
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${DATA_ROOT:-$HOME/vad-rollout/data}"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
# SHA-256 of scripts/train_vad.sh at 1f12b81d. The renderer below reads it by line number.
train_vad_sha256=32a502341d9690e6b735747cd1a92b2d5c81a355dc9495f5ac510ed8443a6f6b
cd "$root" || exit 1

case "$arm" in
  A0) ;;
  A1|A2) unset TOTAL_TRAINING_STEPS TRAINER_SAVE_FREQ ;;
  *) echo "unknown arm $arm; expected A0, A1 or A2" >&2; exit 2 ;;
esac
unset ROLLOUT_N
[[ "$arm" == A2 ]] && export ROLLOUT_N=1

export PATH="$env_root/vad/bin:$PATH"
export TRAINER_N_GPUS_PER_NODE=6
export MODEL_PATH="$data/models/Qwen3.5-4B"
export TEACHER_MODEL_PATH="$MODEL_PATH"
export TASK_TRAIN_FILE="$data/vad/train_vad.parquet"
export BENCHMARK_EVAL_FILE="$data/vad/benchmarks/opd_fast_benchmarks.parquet"
export EXPERIMENT_NAME="$arm"
export TRAINER_DEFAULT_LOCAL_DIR="$out/checkpoints"
export TRAINER_ROLLOUT_DATA_DIR="$out/rollouts"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=3
export BENCHMARK_EVAL_JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:8900/v1}"
export BENCHMARK_EVAL_JUDGE_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
export VAD_TOKSTATS_DIR="$out/tokstats"
if [[ -n "${WANDB_ENV_FILE:-}" && -f "$WANDB_ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$WANDB_ENV_FILE"
  set +a
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

overrides=(CUDA_VISIBLE_DEVICES TRAINER_N_GPUS_PER_NODE MODEL_PATH TEACHER_MODEL_PATH TASK_TRAIN_FILE
  BENCHMARK_EVAL_FILE EXPERIMENT_NAME TRAINER_DEFAULT_LOCAL_DIR TRAINER_ROLLOUT_DATA_DIR
  TRAINER_MAX_ACTOR_CKPT_TO_KEEP BENCHMARK_EVAL_JUDGE_API_BASE BENCHMARK_EVAL_JUDGE_MODEL VAD_TOKSTATS_DIR
  ROLLOUT_N TOTAL_TRAINING_STEPS TRAINER_SAVE_FREQ)
print_overrides() {
  echo "== environment overrides of train_arm.sh $arm (secret-named variables as <set> or <unset>)"
  local name
  for name in "${overrides[@]}"; do echo "$name=${!name-<unset>}"; done
  for name in $(compgen -e | grep -E 'KEY|TOKEN|SECRET|PASSWORD' | sort); do
    if [[ -n "${!name}" ]]; then echo "$name=<set>"; else echo "$name=<unset>"; fi
  done
}

# render: run train_vad.sh lines 1-167 (argument parsing and defaults) and 219-241 (exports), print the
# resolved values, then print the main_ppo argument list of lines 242-357 instead of running it.
render() {
  if [[ "$(sha256sum scripts/train_vad.sh | cut -d' ' -f1)" != "$train_vad_sha256" ]]; then
    echo "scripts/train_vad.sh differs from 1f12b81d; the renderer stops" >&2
    return 2
  fi
  {
    sed -n '1,167p' scripts/train_vad.sh | sed "51s|^PROJECT_ROOT=.*|PROJECT_ROOT=\"$root\"|"
    sed -n '219,241p' scripts/train_vad.sh
    cat <<'EOF'
echo "== resolved values"
for name in ROLLOUT_N TRAINER_N_GPUS_PER_NODE TRAINER_MAX_ACTOR_CKPT_TO_KEEP TOTAL_TRAINING_STEPS \
  TRAINER_SAVE_FREQ TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE EXPERIMENT_NAME CUDA_VISIBLE_DEVICES; do
  echo "$name ${!name}"
done
echo "== main_ppo arguments (scripts/train_vad.sh lines 242-357, one per line)"
EOF
    sed -n '242,357p' scripts/train_vad.sh | sed '1s/^exec python3/printf "%s\\n" python3/'
  } | bash -s -- --profile 4b
}

if [[ $dry_run -eq 1 ]]; then
  print_overrides
  echo "== bash scripts/train_vad.sh --profile 4b --dry-run"
  bash scripts/train_vad.sh --profile 4b --dry-run
  render
  exit $?
fi

mkdir -p "$out"
if [[ $(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .) -ne 6 ]]; then
  echo "CUDA_VISIBLE_DEVICES must name the 6 training GPUs, got '${CUDA_VISIBLE_DEVICES}'" >&2
  exit 2
fi
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  print_overrides
  echo "== python $(command -v python3)"
  echo "VERL_FILE $(python3 -c 'import verl; print(verl.__file__)')"
  echo "== nvidia-smi"
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total --format=csv,noheader
  render
} >> "$out/COMMAND.txt" 2>&1

python3 scripts/vad_rollout/ckpt_keep.py watch "$out/checkpoints" "$out/keep" --step 80 >> "$out/ckpt_keep.log" 2>&1 &
keep_pid=$!

start=$(date +%s)
bash scripts/train_vad.sh --profile 4b 2>&1 | tee -a "$out/console.log"
train_status=${PIPESTATUS[0]}
end=$(date +%s)
echo "train exit $train_status wall_seconds $((end - start)) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/TRAIN_ARM.txt"

# The keep hook ends by itself after the copy. It waits for its next poll when step 80 exists.
latest=$(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [[ "$latest" -ge 80 ]]; then wait "$keep_pid"; else kill "$keep_pid" 2>/dev/null; wait "$keep_pid" 2>/dev/null; fi
echo "latest_checkpointed_iteration $latest keep $(cat "$out/keep/KEEP.json" 2>/dev/null || echo none)" >> "$out/TRAIN_ARM.txt"

# Step wall times and metrics from the console log, as scripts/vad_repro/train_smoke.sh.
python3 - "$out" <<'EOF'
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
steps = {}
for line in (out / "console.log").read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"step:(\d+) - (.*)$", line)
    if not match:
        continue
    metrics = {}
    for item in match.group(2).split(" - "):
        key, sep, value = item.rpartition(":")
        if sep:
            try:
                metrics[key.strip()] = float(value)
            except ValueError:
                pass
    steps[int(match.group(1))] = metrics
(out / "STEP_METRICS.json").write_text(json.dumps(steps, indent=1, sort_keys=True) + "\n")
print("STEP_METRICS steps", len(steps), "timing_s/step", {k: v.get("timing_s/step") for k, v in sorted(steps.items())})
EOF
[[ $train_status -eq 0 ]] || exit "$train_status"

# merge <source global_step dir> <target dir>: merge into <target>.tmp, then rename. A merged target is kept.
merge() {
  local source="$1" target="$2" code
  if [[ -d "$target" ]]; then echo "merge $target exists; skipped" >> "$out/TRAIN_ARM.txt"; return 0; fi
  rm -rf "$target.tmp"
  bash scripts/merge_checkpoint.sh "$source" "$target.tmp" >> "$out/merge.log" 2>&1
  code=$?
  [[ $code -eq 0 ]] && mv "$target.tmp" "$target"
  echo "merge $source -> $target exit $code" >> "$out/TRAIN_ARM.txt"
  return "$code"
}
merge_status=0
mkdir -p "$out/merged"
if [[ -d "$out/keep/global_step_80" ]]; then
  merge "$out/keep/global_step_80" "$out/merged/step_80" || merge_status=1
fi
merge "$out/checkpoints/global_step_$latest" "$out/merged/step_$latest" || merge_status=1
[[ $merge_status -eq 0 ]] || exit 1

python3 scripts/vad_rollout/ckpt_keep.py cleanup "$out/checkpoints" --final-step "$latest" \
  --record "$out/CLEANUP.json" >> "$out/ckpt_keep.log" 2>&1
code=$?
echo "cleanup exit $code" >> "$out/TRAIN_ARM.txt"
exit "$code"
