#!/bin/bash
# One training arm of VLM-VAD-PACKFIX-001 (HANDOFF-VLM-VAD-PACKFIX-001-R6, Part A items 4 and 5).
# ARM=F: VAD_PACKFIX=1 (the fix). ARM=B: VAD_PACKFIX=0 (the A1 code at 2627743a).
# Runs scripts/train_vad.sh --profile 4b on the one B200 with the A1 environment overrides of
# scripts/vad_rollout/train_arm.sh and the Definition 4 changes (both arms equal):
#   1. TRAINER_N_GPUS_PER_NODE=1, CUDA_VISIBLE_DEVICES=0.
#   2. TOTAL_TRAINING_STEPS 20 and TRAINER_SAVE_FREQ 10. Pilot: steps 1 and 2 in run1 with a save after step 2, then
#      step 3 in run2 after a resume, with a save after step 3 (the last step).
#   3. DIRECTIVE-VLM-VAD-PACKFIX-001-007: the training runs with no judge on the GPU, no in-training validation
#      (trainer.test_freq=-1, trainer.val_before_train=False) and actor optimizer_offload=False. The A1 training calls
#      no reward function (ray_trainer.py 654-667 and 2792-2793: zero reward for vopd with teacher_always_on and a
#      teacher_image_key). The judge API base of the training points to judge_stub.py (CPU): it lists the judge model
#      for the endpoint check of train_vad.sh and refuses each judge call (HTTP 503), so an unexpected judge call
#      fails. The vLLM memory share of the rollout engine is ROLLOUT_GPU_UTIL.
#   4. actor.calculate_entropy=True with actor.entropy_coeff=0 (the loss does not change).
#   5. trainer.logger console and file (VERL_FILE_LOGGER_PATH=<out>/metrics_<run>_<start>.jsonl), rollout dumps in
#      <out>/rollouts (as in A1), validation dumps in <out>/validation (per-row judge records), W&B off.
#   6. After the training, one judge session (scripts/vad_packfix/judge_serve.sh, JUDGE_GPU_UTIL) runs:
#      a. the fast validation of Definition 6 in separate val_only processes of train_vad.sh with the A1 validation
#         pipeline: step 0 from the initial weights (trainer.resume_mode=disable), and each checkpoint (pilot 3,
#         FULL 10 and 20) with trainer.resume_mode=resume_path. These processes use optimizer_offload=True (the
#         optimizer is not used). Pilot: 20 rows (val_subset.py).
#         The step-0 model is equal in both arms: with VAL_STEP0_FROM=<file> the arm copies that step-0 dump and
#         does not validate step 0 again.
#      b. rollout_acc.py: the rollout accuracy of each step (R6 Definition 8) from the rollout dumps.
#      The judge stops before this script ends, so it is not on the GPU during the training of the next arm.
#   7. Host memory below the container limit of b200: the Ray worker hook scripts/vad_packfix/hostmem.py (R6
#      Definition 4 items a to h, and item 9: data_source and extra_info in the rollout dumps, logging only), the
#      vLLM multimodal processor cache off (mm_processor_cache_gb 0), and rollout.agent.num_workers 8 as in A1.
# Kept from A1: the train data, model, teacher, batch 96 x 8, mini-batch 96, ppo_max_token_len_per_gpu 9216,
# param offload, seeds 42, MAX_MODEL_LEN 17920, the reward function and its judge settings.
# A restarted task resumes through trainer.resume_mode=auto (train_vad.sh), skips a finished training, and skips a
# step that has a validation dump or a rollout accuracy. A finished arm (TRAIN_DONE) is not run again.
# Usage: ARM=<F|B> [PILOT=1] [VAL_STEP0_FROM=<0.jsonl>] B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 \
#   [JUDGE_GPU_UTIL=0.30] [ROLLOUT_GPU_UTIL=0.40] scripts/vad_packfix/run_packfix.sh <arm output root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arm="${ARM:?ARM must be F or B}"
pilot="${PILOT:-0}"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}"
judge_util="${JUDGE_GPU_UTIL:-0.30}" rollout_util="${ROLLOUT_GPU_UTIL:-0.40}"
step0_from="${VAL_STEP0_FROM:-}"
case "$arm" in F) flag=1 ;; B) flag=0 ;; *) echo "ARM must be F or B" >&2; exit 2 ;; esac
data="$b200/harness/data"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_PACKFIX.txt"
export PYTHONDONTWRITEBYTECODE=1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) ARM $arm VAD_PACKFIX $flag PILOT $pilot"
  echo "judge_gpu_util $judge_util rollout_gpu_util $rollout_util actor optimizer_offload False (training) True (validation)"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  df -B1 "$out"
} >> "$log" 2>&1

if [[ "$pilot" == 1 ]]; then steps=3; ckpt_steps=(3); else steps=20; ckpt_steps=(10 20); fi
if [[ -f "$out/TRAIN_DONE" ]]; then
  echo "arm $arm already finished" >> "$log"
  exit 0
fi

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 10000 \
  >> "$out/gpu_mem.csv" 2>&1 &
smi_pid=$!
judge_pid=""
stub_pid=""
stop_stub() {
  [[ -n "$stub_pid" ]] || return 0
  kill "$stub_pid" 2>/dev/null
  wait "$stub_pid" 2>/dev/null
  stub_pid=""
  echo "judge stub stopped: $(grep -c ' served$' "$out/judge_stub.log" 2>/dev/null) model-list requests, $(grep -c ' refused$' "$out/judge_stub.log" 2>/dev/null) refused judge calls" >> "$log"
}
stop_judge() {
  [[ -n "$judge_pid" ]] || return 0
  touch "$out/judge/STOP"
  wait "$judge_pid" 2>/dev/null
  echo "judge exit $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  judge_pid=""
}
finish() {
  stop_stub
  stop_judge
  kill "$smi_pid" 2>/dev/null
}
trap finish EXIT

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
export BENCHMARK_EVAL_JUDGE_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
export VAD_TOKSTATS_DIR="$out/tokstats"
export ROLLOUT_GPU_MEMORY_UTILIZATION="$rollout_util"
export TRAINER_LOGGER='["console","file"]'
export WANDB_MODE=disabled WANDB_DIR="$out/wandb"
export VAD_PACKFIX="$flag"
export ROLLOUT_AGENT_NUM_WORKERS=8
export RAY_DEDUP_LOGS=0  # each worker prints its VAD_HOSTMEM lines (logging only)
common=(
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_PACKFIX='$flag'"
  ++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=scripts.vad_packfix.hostmem.setup
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
  actor_rollout_ref.actor.calculate_entropy=True
  actor_rollout_ref.actor.entropy_coeff=0
  "++trainer.validation_data_dir=$out/validation"
)

# train <total steps> <save freq> <tag>. Each run writes its own file-logger file, because the verl FileLogger
# truncates its file at the start (metrics.py reads them in time order; a later run wins a step).
train() {
  local total="$1" save="$2" tag="$3" start code metrics_file
  start=$(date +%s)
  metrics_file="$out/metrics_${tag}_$start.jsonl"
  echo "== train $tag: total $total save $save val_before_train False test_freq -1 metrics $metrics_file start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  VERL_FILE_LOGGER_PATH="$metrics_file" TOTAL_TRAINING_STEPS="$total" TRAINER_SAVE_FREQ="$save" \
    ACTOR_OPTIMIZER_OFFLOAD=False \
    bash scripts/train_vad.sh --profile 4b -- "${common[@]}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$metrics_file'" \
    trainer.val_before_train=False trainer.test_freq=-1 \
    > "$out/console_$tag.log" 2>&1
  code=$?
  cat "$out/console_$tag.log" >> "$out/console.log"
  echo "train $tag exit $code wall_seconds $(( $(date +%s) - start )) latest $(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)" >> "$log"
  return "$code"
}

# validate <tag> <step>: the fast validation of one model in a val_only process (step 0: initial weights).
validate() {
  local tag="$1" step="$2" start code resume
  start=$(date +%s)
  if [[ "$step" == 0 ]]; then
    resume=(trainer.resume_mode=disable)
  else
    resume=(trainer.resume_mode=resume_path "++trainer.resume_from_path=$out/checkpoints/global_step_$step")
  fi
  echo "== validate $tag: step $step ${resume[*]} start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  VERL_FILE_LOGGER_PATH="$out/val_${tag}_$start.jsonl" TOTAL_TRAINING_STEPS="$steps" TRAINER_SAVE_FREQ="$steps" \
    ACTOR_OPTIMIZER_OFFLOAD=True \
    bash scripts/train_vad.sh --profile 4b -- "${common[@]}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$out/val_${tag}_$start.jsonl'" \
    trainer.val_before_train=True ++trainer.val_only=True trainer.test_freq=-1 "${resume[@]}" \
    > "$out/console_$tag.log" 2>&1
  code=$?
  echo "validate $tag exit $code wall_seconds $(( $(date +%s) - start )) dump $(ls "$out/validation/$step.jsonl" 2>/dev/null || echo none)" >> "$log"
  [[ $code -eq 0 && -f "$out/validation/$step.jsonl" ]]
}

# Training (no judge on the GPU). The endpoint check of train_vad.sh needs a judge model list: judge_stub.py (CPU)
# serves it and refuses each judge call (HTTP 503), so an unexpected judge call fails.
latest=$(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [[ "$latest" -lt "$steps" ]]; then
  stub_port=$("$py" scripts/vad_packfix/free_port.py)
  "$py" scripts/vad_packfix/judge_stub.py "$stub_port" "$BENCHMARK_EVAL_JUDGE_MODEL" "$out/judge_stub.log" &
  stub_pid=$!
  export BENCHMARK_EVAL_JUDGE_API_BASE="http://127.0.0.1:$stub_port/v1"
  for _ in $(seq 1 50); do
    curl -fsS --max-time 2 "$BENCHMARK_EVAL_JUDGE_API_BASE/models" > /dev/null 2>&1 && break
    sleep 0.2
  done
  echo "judge stub (training) pid $stub_pid port $stub_port" >> "$log"
  if [[ "$pilot" == 1 ]]; then
    if [[ ! -d "$out/checkpoints/global_step_2" ]]; then
      train 2 2 run1 || exit 1
    fi
    [[ -d "$out/checkpoints/global_step_2" ]] || { echo "no checkpoint after step 2" >> "$log"; exit 1; }
    # Memory measurement of DIRECTIVE-VLM-VAD-PACKFIX-001-004 (run_memmeasure.sh): run1 only.
    [[ "${MEASURE_RUN1_ONLY:-0}" == 1 ]] && { echo "measurement: stop after run1" >> "$log"; exit 0; }
    echo "stop after step 2; resume" >> "$log"
    train 3 2 run2 || exit 1
    grep -n -i -E "resum|load(ing)? (from )?checkpoint|global_step_2" "$out/console_run2.log" | head -n 20 \
      | sed 's/^/resume evidence: /' >> "$log"
  else
    train 20 10 full || exit 1
  fi
  stop_stub
fi
latest=$(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
[[ "$latest" -ge "$steps" ]] || { echo "latest checkpoint $latest < $steps" >> "$log"; exit 1; }
for k in "${ckpt_steps[@]}"; do
  [[ -d "$out/checkpoints/global_step_$k" ]] || { echo "no checkpoint of step $k" >> "$log"; exit 1; }
done
echo "flag lines: $(grep -c "VAD_PACKFIX=$flag pid=" "$out/console.log") with VAD_PACKFIX=$flag, $(grep -c 'VAD_PACKFIX=' "$out/console.log") in all; fix lines $(grep -c 'Packed linear-attention fix applied' "$out/console.log")" >> "$log"
grep -o -E "VAD_PACKFIX=[^ ]* pid=[0-9]+" "$out/console.log" | sort | uniq -c | sed 's/^/flag: /' >> "$log"

# Judge session: fast validation and rollout accuracy.
port=$("$py" scripts/vad_packfix/free_port.py)
rm -f "$out/judge/STOP" "$out/judge/READY.json"
judge_start=$(date +%s)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 JUDGE_PORT="$port" JUDGE_GPU_UTIL="$judge_util" \
  JUDGE_MODEL_DIR="$data/models/Mistral-Small-3.2-24B-Instruct-2506" JUDGE_PYTHON="$env_root/judge/bin/python" \
  bash scripts/vad_packfix/judge_serve.sh "$out/judge" &
judge_pid=$!
ready=0
for _ in $(seq 1 360); do
  [[ -e "$out/judge/READY.json" ]] && { ready=1; break; }
  kill -0 "$judge_pid" 2>/dev/null || break
  sleep 5
done
echo "judge ready $ready port $port at $(date -u +%Y-%m-%dT%H:%M:%SZ) judge_start_seconds $(( $(date +%s) - judge_start ))" >> "$log"
[[ $ready -eq 1 ]] || exit 1
export BENCHMARK_EVAL_JUDGE_API_BASE="http://127.0.0.1:$port/v1"

mkdir -p "$out/validation"
if [[ -n "$step0_from" ]]; then
  cp -f "$step0_from" "$out/validation/0.jsonl" || exit 1
  echo "step 0 validation copied from $step0_from sha256 $(sha256sum "$out/validation/0.jsonl" | cut -d' ' -f1) (the step-0 model is equal in both arms)" >> "$log"
elif [[ ! -f "$out/validation/0.jsonl" ]]; then
  validate val0 0 || exit 1
fi
for k in "${ckpt_steps[@]}"; do
  [[ -f "$out/validation/$k.jsonl" ]] || validate "val$k" "$k" || exit 1
done
echo "== score rollouts: start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
PYTHONPATH="$root" "$py" scripts/vad_packfix/rollout_acc.py "$out" "$steps" "$BENCHMARK_EVAL_JUDGE_API_BASE" \
  >> "$log" 2>&1 || { echo "rollout accuracy failed" >> "$log"; exit 1; }
stop_judge

"$py" scripts/vad_packfix/metrics.py "$out" "$steps" >> "$log" 2>&1 || { echo "metrics check failed" >> "$log"; exit 1; }
"$py" scripts/vad_packfix/val_acc.py "$out" >> "$log" 2>&1 || { echo "validation records failed" >> "$log"; exit 1; }
touch "$out/TRAIN_DONE"
echo "== arm $arm done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
