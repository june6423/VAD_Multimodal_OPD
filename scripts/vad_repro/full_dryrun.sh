#!/bin/bash
# VLM-VAD-REPRO-001 R2 Part B task 4 (HANDOFF-VLM-VAD-REPRO-001-R2, gates G4 and G5). No GPU work.
# Usage: scripts/vad_repro/full_dryrun.sh <output_root>
# Exports the G4 override list and runs bash scripts/train_vad.sh --profile 4b --dry-run.
# BENCHMARK_EVAL_JUDGE_API_BASE is a placeholder: the full run names the judge endpoint.
# The directories point inside <output_root>; the dry run creates none of them.
# Then it runs python -m pytest tests -q on the branch (G5).
# The record is <output_root>/FULL_DRYRUN.txt.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
data=/workspace/harness/data
mkdir -p "$out"
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1
log="$out/FULL_DRYRUN.txt"

overrides=(
  "MODEL_PATH=$data/vision-opd/Qwen3.5-4B"
  "TEACHER_MODEL_PATH=$data/vision-opd/Qwen3.5-4B"
  "TASK_TRAIN_FILE=$data/vad/train_vad.parquet"
  "BENCHMARK_EVAL_FILE=$data/vad/benchmarks/opd_fast_benchmarks.parquet"
  "CUDA_VISIBLE_DEVICES=4,5,6,7"
  "TRAINER_N_GPUS_PER_NODE=4"
  "TRAINER_MAX_ACTOR_CKPT_TO_KEEP=1"
  "TRAINER_DEFAULT_LOCAL_DIR=$out/checkpoints"
  "TRAINER_ROLLOUT_DATA_DIR=$out/rollouts"
  "WANDB_DIR=$out/wandb"
  "BENCHMARK_EVAL_JUDGE_API_BASE=http://judge-placeholder:8900/v1"
  "BENCHMARK_EVAL_JUDGE_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  "EXPERIMENT_NAME=VAD-qwen35-4b-VLM-VAD-REPRO-001"
)

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD"; git -c safe.directory="$root" rev-parse HEAD
  echo "== override list (environment of the dry run)"
  printf '%s\n' "${overrides[@]}"
  echo "== env $(printf '%q ' "${overrides[@]}")bash scripts/train_vad.sh --profile 4b --dry-run"
  env "${overrides[@]}" bash scripts/train_vad.sh --profile 4b --dry-run
  echo "dry_run exit $?"
  echo "== python -m pytest tests -q"
  python -m pytest tests -q -p no:cacheprovider
  echo "pytest exit $?"
} > "$log" 2>&1

grep -q "^dry_run exit 0$" "$log" && grep -q "^pytest exit 0$" "$log"
