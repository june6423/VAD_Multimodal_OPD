#!/bin/bash
# Host-memory diagnosis of VLM-VAD-PACKFIX-001 (DIRECTIVE-VLM-VAD-PACKFIX-001-002). Diagnosis only, no training.
# Starts arm F as run_packfix.sh does at f9d0b09 (same environment, judge next to training, same overrides), with:
#   1. trainer.val_before_train=False and TOTAL_TRAINING_STEPS=1 (no validation, no update).
#   2. The Ray worker hook scripts/vad_packfix/memdiag.py (MEMDIAG_OUT=<out>). It measures the WorkerDict CPU tensors
#      after init_model and the AgentLoopWorker trajectory fields of the step-1 batch, then stops the run by design.
#   3. memwatch.sh: cgroup memory and the largest processes every 5 s.
# memdiag_summary.py then writes MEMDIAG.json. Exit 0 only when all measurement files exist.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [JUDGE_GPU_UTIL] [ROLLOUT_GPU_UTIL] \
#   scripts/vad_packfix/run_memdiag.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}"
judge_util="${JUDGE_GPU_UTIL:-0.40}" rollout_util="${ROLLOUT_GPU_UTIL:-0.25}"
data="$b200/harness/data"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_MEMDIAG.txt"
export PYTHONDONTWRITEBYTECODE=1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) MEMDIAG ARM F"
  echo "judge_gpu_util $judge_util rollout_gpu_util $rollout_util"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
} >> "$log" 2>&1
PYTHONPATH="$root" "$py" -c "import scripts.vad_packfix.memdiag as m; print('hook module', m.__file__)" >> "$log" 2>&1 \
  || { echo "hook module import failed" >> "$log"; exit 1; }

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
export EXPERIMENT_NAME="PACKFIX-MEMDIAG"
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
export ROLLOUT_AGENT_NUM_WORKERS=2
start=$(date +%s)
echo "== train start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
VERL_FILE_LOGGER_PATH="$out/metrics.jsonl" TOTAL_TRAINING_STEPS=1 TRAINER_SAVE_FREQ=1 \
  bash scripts/train_vad.sh --profile 4b -- \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_PACKFIX='1'" \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
  actor_rollout_ref.actor.calculate_entropy=True \
  actor_rollout_ref.actor.entropy_coeff=0 \
  "++trainer.validation_data_dir=$out/validation" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$out/metrics.jsonl'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.MEMDIAG_OUT='$out'" \
  ++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=scripts.vad_packfix.memdiag.setup \
  trainer.val_before_train=False \
  > "$out/console.log" 2>&1
code=$?
echo "train exit $code (a nonzero exit is expected: the hook stops the run) wall_seconds $(( $(date +%s) - start ))" >> "$log"
grep -c "MEMDIAG hook installed" "$out/console.log" | sed 's/^/hook install lines: /' >> "$log"
grep -E "MEMDIAG (workerdict|agent loop)|MEMDIAG: measurement done" "$out/console.log" | head -n 6 >> "$log"
"$py" scripts/vad_packfix/memdiag_summary.py "$out" "$out/MEMDIAG.json" >> "$log" 2>&1 \
  || { echo "summary failed" >> "$log"; exit 1; }
rm -rf "$out/checkpoints"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
