#!/bin/bash
# One training arm of VLM-VAD-PACKFIX-001 (HANDOFF-VLM-VAD-PACKFIX-001-R2, Part A items 4 and 5).
# ARM=F: VAD_PACKFIX=1 (the fix). ARM=B: VAD_PACKFIX=0 (the A1 code at 2627743a).
# Runs scripts/train_vad.sh --profile 4b on the one B200 with the A1 environment overrides of
# scripts/vad_rollout/train_arm.sh and the Definition 4 changes (both arms equal):
#   1. TRAINER_N_GPUS_PER_NODE=1, CUDA_VISIBLE_DEVICES=0.
#   2. TOTAL_TRAINING_STEPS 20 and TRAINER_SAVE_FREQ 10 (pilot: 1, then 2 after a resume; save each step).
#   3. vLLM memory share of the rollout engine ROLLOUT_GPU_UTIL and of the Mistral judge JUDGE_GPU_UTIL. The judge
#      runs next to training (the A1 reward calls it for each rule miss) with scripts/vad_packfix/judge_serve.sh.
#   4. actor.calculate_entropy=True with actor.entropy_coeff=0 (the loss does not change).
#   5. trainer.logger console and file (VERL_FILE_LOGGER_PATH=<out>/metrics.jsonl), rollout dumps in <out>/rollouts
#      (as in A1), validation dumps in <out>/validation (per-row judge records), W&B off.
#   6. Fast validation at steps 0, 10 and 20: trainer.val_before_train=True after the train_vad.sh value, and the
#      A1 test_freq 10. Pilot: 20 rows (val_subset.py) at steps 0, 1 and 2.
#   7. Host memory below the container limit of b200 (DIRECTIVE-VLM-VAD-PACKFIX-001-001): rollout.agent.num_workers 2
#      (A1: 8) and the vLLM multimodal processor cache off (mm_processor_cache_gb 0). Throughput settings only.
# Kept from A1: the train data, model, teacher, batch 96 x 8, mini-batch 96, ppo_max_token_len_per_gpu 9216, offload,
# seeds 42, MAX_MODEL_LEN 17920, the reward function and its judge settings.
# A restarted task resumes through trainer.resume_mode=auto (train_vad.sh). A finished arm is not run again.
# Usage: ARM=<F|B> [PILOT=1] B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 \
#   [JUDGE_GPU_UTIL=0.30] [ROLLOUT_GPU_UTIL=0.40] scripts/vad_packfix/run_packfix.sh <arm output root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arm="${ARM:?ARM must be F or B}"
pilot="${PILOT:-0}"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}"
judge_util="${JUDGE_GPU_UTIL:-0.30}" rollout_util="${ROLLOUT_GPU_UTIL:-0.40}"
case "$arm" in F) flag=1 ;; B) flag=0 ;; *) echo "ARM must be F or B" >&2; exit 2 ;; esac
data="$b200/harness/data"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_PACKFIX.txt"
export PYTHONDONTWRITEBYTECODE=1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) ARM $arm VAD_PACKFIX $flag PILOT $pilot"
  echo "judge_gpu_util $judge_util rollout_gpu_util $rollout_util"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  df -B1 "$out"
} >> "$log" 2>&1

if [[ "$pilot" == 1 ]]; then steps=2; else steps=20; fi
latest=$(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [[ "$latest" -ge "$steps" && -f "$out/TRAIN_DONE" ]]; then
  echo "arm $arm already finished (latest checkpoint $latest)" >> "$log"
  exit 0
fi

# Judge next to training (Definition 4).
port=$("$py" scripts/vad_packfix/free_port.py)
rm -f "$out/judge/STOP"
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 JUDGE_PORT="$port" JUDGE_GPU_UTIL="$judge_util" \
  JUDGE_MODEL_DIR="$data/models/Mistral-Small-3.2-24B-Instruct-2506" JUDGE_PYTHON="$env_root/judge/bin/python" \
  bash scripts/vad_packfix/judge_serve.sh "$out/judge" &
judge_pid=$!
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 10000 \
  >> "$out/gpu_mem.csv" 2>&1 &
smi_pid=$!
finish() {
  touch "$out/judge/STOP"
  wait "$judge_pid" 2>/dev/null
  echo "judge exit $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  kill "$smi_pid" 2>/dev/null
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

if [[ "$pilot" == 1 ]]; then
  val_file="$out/val_subset.parquet"
  [[ -f "$val_file" ]] || "$py" scripts/vad_packfix/val_subset.py "$data/vad/benchmarks/opd_fast_benchmarks.parquet" \
    "$val_file" "$out/VAL_SUBSET.json" >> "$log" 2>&1 || exit 1
else
  val_file="$data/vad/benchmarks/opd_fast_benchmarks.parquet"
fi

export PATH="$env_root/vad/bin:$PATH"
unset ROLLOUT_N
export CUDA_VISIBLE_DEVICES=0 TRAINER_N_GPUS_PER_NODE=1
export MODEL_PATH="$data/models/Qwen3.5-4B"
export TEACHER_MODEL_PATH="$MODEL_PATH"
export TASK_TRAIN_FILE="$data/vad/train_vad.parquet"
export BENCHMARK_EVAL_FILE="$val_file"
export MAX_MODEL_LEN=17920
export EXPERIMENT_NAME="PACKFIX-$arm"
export TRAINER_DEFAULT_LOCAL_DIR="$out/checkpoints"
export TRAINER_ROLLOUT_DATA_DIR="$out/rollouts"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=3
export BENCHMARK_EVAL_JUDGE_API_BASE="http://127.0.0.1:$port/v1"
export BENCHMARK_EVAL_JUDGE_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
export VAD_TOKSTATS_DIR="$out/tokstats"
export ROLLOUT_GPU_MEMORY_UTILIZATION="$rollout_util"
export TRAINER_LOGGER='["console","file"]'
export WANDB_MODE=disabled WANDB_DIR="$out/wandb"
export VAD_PACKFIX="$flag"
export ROLLOUT_AGENT_NUM_WORKERS=2
common=(
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_PACKFIX='$flag'"
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
  actor_rollout_ref.actor.calculate_entropy=True
  actor_rollout_ref.actor.entropy_coeff=0
  "++trainer.validation_data_dir=$out/validation"
)

# train <total steps> <save freq> <val_before_train> <tag>. Each run writes its own file-logger file, because the
# verl FileLogger truncates its file at the start (metrics.py reads them in time order; a later run wins a step).
train() {
  local total="$1" save="$2" vbt="$3" tag="$4" start code metrics_file
  start=$(date +%s)
  metrics_file="$out/metrics_${tag}_$start.jsonl"
  echo "== train $tag: total $total save $save val_before_train $vbt metrics $metrics_file start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  VERL_FILE_LOGGER_PATH="$metrics_file" TOTAL_TRAINING_STEPS="$total" TRAINER_SAVE_FREQ="$save" \
    bash scripts/train_vad.sh --profile 4b -- "${common[@]}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$metrics_file'" trainer.val_before_train="$vbt" \
    > "$out/console_$tag.log" 2>&1
  code=$?
  cat "$out/console_$tag.log" >> "$out/console.log"
  echo "train $tag exit $code wall_seconds $(( $(date +%s) - start )) latest $(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)" >> "$log"
  return "$code"
}

if [[ "$pilot" == 1 ]]; then
  if [[ ! -d "$out/checkpoints/global_step_1" ]]; then
    train 1 1 True run1 || exit 1
  fi
  [[ -d "$out/checkpoints/global_step_1" ]] || { echo "no checkpoint after step 1" >> "$log"; exit 1; }
  echo "stop after step 1; resume" >> "$log"
  train 2 1 False run2 || exit 1
  grep -n -i -E "resum|load(ing)? (from )?checkpoint|global_step_1" "$out/console_run2.log" | head -n 20 \
    | sed 's/^/resume evidence: /' >> "$log"
else
  train 20 10 True full || exit 1
fi
latest=$(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
[[ "$latest" -ge "$steps" ]] || { echo "latest checkpoint $latest < $steps" >> "$log"; exit 1; }
echo "flag lines: $(grep -c "VAD_PACKFIX=$flag pid=" "$out/console.log") with VAD_PACKFIX=$flag, $(grep -c 'VAD_PACKFIX=' "$out/console.log") in all; fix lines $(grep -c 'Packed linear-attention fix applied' "$out/console.log")" >> "$log"
grep -o -E "VAD_PACKFIX=[^ ]* pid=[0-9]+" "$out/console.log" | sort | uniq -c | sed 's/^/flag: /' >> "$log"

"$py" scripts/vad_packfix/metrics.py "$out" "$steps" >> "$log" 2>&1 || { echo "metrics check failed" >> "$log"; exit 1; }
"$py" scripts/vad_packfix/val_acc.py "$out" >> "$log" 2>&1 || { echo "validation records failed" >> "$log"; exit 1; }
touch "$out/TRAIN_DONE"
echo "== arm $arm done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
