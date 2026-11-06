#!/bin/bash
# Data-equality check of the host-memory patch (DIRECTIVE-VLM-VAD-PACKFIX-001-003 action 3). Check only, no training.
# Starts arm F as run_packfix.sh does (same environment, judge next to training, same overrides), with:
#   1. trainer.val_before_train=False and TOTAL_TRAINING_STEPS=1 (no validation, no update).
#   2. The Ray worker hook scripts/vad_packfix/hostmem_check.py in place of hostmem.py (it installs hostmem.py too).
#      Each agent worker runs its first HOSTMEM_CHECK_GROUPS prompt groups of the step-1 batch with and without the
#      patch and compares the sha256 of each field of each row. Then the hook stops the run by design.
#   3. memwatch.sh as in run_both.sh.
# hostmem_check.py <out> then writes HOSTMEM_CHECK.json. Exit 0 only when no hash differs.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [JUDGE_GPU_UTIL] [ROLLOUT_GPU_UTIL] \
#   [HOSTMEM_CHECK_GROUPS=4] scripts/vad_packfix/run_hostmem_check.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}"
judge_util="${JUDGE_GPU_UTIL:-0.40}" rollout_util="${ROLLOUT_GPU_UTIL:-0.25}"
groups="${HOSTMEM_CHECK_GROUPS:-4}"
data="$b200/harness/data"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_HOSTMEM_CHECK.txt"
export PYTHONDONTWRITEBYTECODE=1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) HOSTMEM_CHECK ARM F groups $groups"
  echo "judge_gpu_util $judge_util rollout_gpu_util $rollout_util"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
} >> "$log" 2>&1

bash scripts/vad_packfix/memwatch.sh "$out/memwatch.log" 3000 &
watch_pid=$!
port=$("$py" scripts/vad_packfix/free_port.py)
rm -f "$out/judge/STOP"
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 JUDGE_PORT="$port" JUDGE_GPU_UTIL="$judge_util" \
  JUDGE_MODEL_DIR="$data/models/Mistral-Small-3.2-24B-Instruct-2506" JUDGE_PYTHON="$env_root/judge/bin/python" \
  bash scripts/vad_packfix/judge_serve.sh "$out/judge" &
judge_pid=$!
finish() {
  touch "$out/judge/STOP"
  wait "$judge_pid" 2>/dev/null
  echo "judge exit $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  kill "$watch_pid" 2>/dev/null
}
trap finish EXIT
ready=0
for _ in $(seq 1 360); do
  [[ -e "$out/judge/READY.json" ]] && { ready=1; break; }
  kill -0 "$judge_pid" 2>/dev/null || break
  sleep 5
done
echo "judge ready $ready port $port at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
[[ $ready -eq 1 ]] || exit 1

export PATH="$env_root/vad/bin:$PATH"
unset ROLLOUT_N
export CUDA_VISIBLE_DEVICES=0 TRAINER_N_GPUS_PER_NODE=1
export MODEL_PATH="$data/models/Qwen3.5-4B"
export TEACHER_MODEL_PATH="$MODEL_PATH"
export TASK_TRAIN_FILE="$data/vad/train_vad.parquet"
export BENCHMARK_EVAL_FILE="$data/vad/benchmarks/opd_fast_benchmarks.parquet"
export MAX_MODEL_LEN=17920
export EXPERIMENT_NAME="PACKFIX-HOSTMEM-CHECK"
export TRAINER_DEFAULT_LOCAL_DIR="$out/checkpoints"
export TRAINER_ROLLOUT_DATA_DIR="$out/rollouts"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=3
export BENCHMARK_EVAL_JUDGE_API_BASE="http://127.0.0.1:$port/v1"
export BENCHMARK_EVAL_JUDGE_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
export VAD_TOKSTATS_DIR="$out/tokstats"
export ROLLOUT_GPU_MEMORY_UTILIZATION="$rollout_util"
export TRAINER_LOGGER='["console","file"]'
export WANDB_MODE=disabled WANDB_DIR="$out/wandb"
export VAD_PACKFIX=1
export ROLLOUT_AGENT_NUM_WORKERS=8
export RAY_DEDUP_LOGS=0  # each worker prints its lines (logging only)
start=$(date +%s)
echo "== train start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
VERL_FILE_LOGGER_PATH="$out/metrics.jsonl" TOTAL_TRAINING_STEPS=1 TRAINER_SAVE_FREQ=1 \
  bash scripts/train_vad.sh --profile 4b -- \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_PACKFIX='1'" \
  ++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=scripts.vad_packfix.hostmem_check.setup \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
  actor_rollout_ref.actor.calculate_entropy=True \
  actor_rollout_ref.actor.entropy_coeff=0 \
  "++trainer.validation_data_dir=$out/validation" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$out/metrics.jsonl'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.HOSTMEM_CHECK_OUT='$out'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.HOSTMEM_CHECK_GROUPS='$groups'" \
  trainer.val_before_train=False \
  > "$out/console.log" 2>&1
code=$?
echo "train exit $code (a nonzero exit is expected: the hook stops the run) wall_seconds $(( $(date +%s) - start ))" >> "$log"
grep -c "HOSTMEM_CHECK hook installed" "$out/console.log" | sed 's/^/hook install lines: /' >> "$log"
grep -E "VAD_HOSTMEM pid=[0-9]+ reference model removed|HOSTMEM_CHECK pid=" "$out/console.log" | head -n 12 >> "$log"
"$py" scripts/vad_packfix/hostmem_check.py "$out" "$out/HOSTMEM_CHECK.json" >> "$log" 2>&1
check=$?
echo "check exit $check" >> "$log"
rm -rf "$out/checkpoints"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit "$check"
