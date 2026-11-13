#!/bin/bash
# One training arm of VLM-VAD-VIEW-002 (HANDOFF-VLM-VAD-VIEW-002-R1 Part A items 4 and 7, Definitions 1 to 6).
# The base run is arm B of VLM-VAD-PACKFIX-001 (scripts/vad_packfix/run_packfix.sh with ARM=B, VAD_PACKFIX=0) with
# each of its settings: one B200, no judge during the training (judge_stub.py), actor optimizer_offload False in the
# training, entropy logging, file logger, rollout dumps, the Ray hook scripts/vad_packfix/hostmem.setup,
# mm_processor_cache_gb 0, the validation of saved checkpoints after the training in 4 val_only processes
# (val_split.py, val_concat.py, valtrim.setup, valguard.sh, val_batch_size 32), rollout_acc.py.
# Changes (Definition 1): TOTAL_TRAINING_STEPS 40, checkpoints of steps 20 and 40 (TRAINER_SAVE_FREQ 20), and
# Definition 3: VAD_VIEW_ARM=<ARM> for the trainer (scripts/vad_view2/hook.py through ray_trainer._vad_view2_hook,
# given to the Ray workers by runtime_env env_vars with VAD_VIEW_MAP and VAD_VIEW_LOG_DIR=<out>/views_log), and
# ROLLOUT_N=1 for N1R (rollout.n 8 for B, T8 and N1V8; N1V8 generates 1 response and copies it 8 times).
# Step 0 of the fast validation is the copy of VAL_STEP0_FROM (Definition 5, TASK-VLM-VAD-PACKFIX-001-029).
# After the judge session: scripts/merge_checkpoint.sh of each checkpoint step into <out>/merged/step_<k>, then
# heldout.py (Definition 6) on it into <out>/heldout/<k>/. Then metrics.py and val_acc.py (scripts/vad_packfix).
# PILOT=1 (Part A item 7): TRAIN_BATCH_SIZE and PPO_MINI_BATCH_SIZE 16, the train file PILOT_TRAIN_FILE (16 rows of
#   pilot_rows.py, written 2 times: 2 steps in one epoch), step 1 with a save in run1, stop, resume for step 2 in run2 (save at step 2); validation of step 2
#   on 20 rows (val_subset.py); held-out on HELDOUT_N=10 items.
# CHECK=1 (check_view2.sh, gates G2 and G3): step 1 only, training only (no judge session, no validation). Here
#   CODE_ROOT may name another code tree (the PACKFIX archive), WORKER_HOOK replaces the Ray hook, G3_PY is put on
#   PYTHONPATH, and VAD_G3_REPLAY and VAD_G3_GRAD go to the workers.
# A restarted task resumes through trainer.resume_mode=auto, skips a finished training, a validated step, a merged
# checkpoint and a finished held-out step. A finished arm (TRAIN_DONE) is not run again.
# Usage: ARM=<B|T8|N1V8|N1R> VAD_VIEW_MAP=<VIEWMAP.json> HELDOUT_JSON=<HELDOUT.json> [PILOT=1] [CHECK=1]
#   [VAL_STEP0_FROM=<0.jsonl>] B200_ROOT=<root> ENV_ROOT=<env root> VAD_CONTAINER_CGROUP=<cgroup> \
#   [JUDGE_GPU_UTIL=0.40] [ROLLOUT_GPU_UTIL=0.25] scripts/vad_view2/run_view2.sh <arm output root>
set -uo pipefail
out="$1"
own_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arm="${ARM:?ARM must be B, T8, N1V8 or N1R}"
pilot="${PILOT:-0}" check="${CHECK:-0}"
root="$own_root"
if [[ -n "${CODE_ROOT:-}" ]]; then
  [[ "$check" == 1 ]] || { echo "CODE_ROOT needs CHECK=1" >&2; exit 2; }
  root="$CODE_ROOT"
fi
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" viewmap="${VAD_VIEW_MAP:?}"
judge_util="${JUDGE_GPU_UTIL:-0.40}" rollout_util="${ROLLOUT_GPU_UTIL:-0.25}"
step0_from="${VAL_STEP0_FROM:-}"
val_bs="${VAL_BATCH_SIZE:-32}" val_limit=10800
case "$arm" in B|T8|N1V8|N1R) ;; *) echo "ARM must be B, T8, N1V8 or N1R" >&2; exit 2 ;; esac
data="$b200/harness/data"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_VIEW2.txt"
export PYTHONDONTWRITEBYTECODE=1
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) code $root git HEAD $(git -c safe.directory="$root" rev-parse HEAD 2>/dev/null || echo none) script HEAD $(git -c safe.directory="$own_root" -C "$own_root" rev-parse HEAD) ARM $arm PILOT $pilot CHECK $check VAD_PACKFIX 0"
  echo "judge_gpu_util $judge_util rollout_gpu_util $rollout_util view map $viewmap sha256 $(sha256sum "$viewmap" | cut -d' ' -f1)"
  echo "validation: val_batch_size $val_bs wall limit $val_limit s for each val_only process"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  df -B1 "$out"
} >> "$log" 2>&1

if [[ "$pilot" == 1 ]]; then steps=2; ckpt_steps=(2)
elif [[ "$check" == 1 ]]; then steps=1; ckpt_steps=()
else steps=40; ckpt_steps=(20 40); fi
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
  [[ -f "$val_file" ]] || "$py" "$own_root/scripts/vad_packfix/val_subset.py" \
    "$data/vad/benchmarks/opd_fast_benchmarks.parquet" "$val_file" "$out/VAL_SUBSET.json" >> "$log" 2>&1 || exit 1
else
  val_file="$data/vad/benchmarks/opd_fast_benchmarks.parquet"
fi
split_dir="$out/val_split"
if [[ "$check" != 1 ]]; then
  [[ -f "$split_dir/VAL_SPLIT.json" ]] || "$py" scripts/vad_packfix/val_split.py \
    "$data/vad/benchmarks/opd_fast_benchmarks.parquet" "$split_dir" 4 >> "$log" 2>&1 || exit 1
  "$py" scripts/vad_packfix/val_batches.py "$split_dir" "$val_bs" "$split_dir/VAL_BATCHES.json" >> "$log" 2>&1 || exit 1
fi

export PATH="$env_root/vad/bin:$PATH"
unset ROLLOUT_N
[[ "$arm" == N1R ]] && export ROLLOUT_N=1
export CUDA_VISIBLE_DEVICES=0 TRAINER_N_GPUS_PER_NODE=1
export MODEL_PATH="$data/models/Qwen3.5-4B"
export TEACHER_MODEL_PATH="$MODEL_PATH"
export TASK_TRAIN_FILE="$data/vad/train_vad.parquet"
if [[ "$pilot" == 1 ]]; then
  export TASK_TRAIN_FILE="${PILOT_TRAIN_FILE:?}" TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16
fi
export BENCHMARK_EVAL_FILE="$val_file"
export MAX_MODEL_LEN=17920
export EXPERIMENT_NAME="VIEW2-$arm"
export TRAINER_DEFAULT_LOCAL_DIR="$out/checkpoints"
export TRAINER_ROLLOUT_DATA_DIR="$out/rollouts"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=3
export BENCHMARK_EVAL_JUDGE_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
export VAD_TOKSTATS_DIR="$out/tokstats"
export ROLLOUT_GPU_MEMORY_UTILIZATION="$rollout_util"
export TRAINER_LOGGER='["console","file"]'
export WANDB_MODE=disabled WANDB_DIR="$out/wandb"
export VAD_PACKFIX=0
export VAD_VIEW_ARM="$arm" VAD_VIEW_MAP="$viewmap" VAD_VIEW_LOG_DIR="$out/views_log"
export ROLLOUT_AGENT_NUM_WORKERS=8
export RAY_DEDUP_LOGS=0
hook="${WORKER_HOOK:-scripts.vad_packfix.hostmem.setup}"
common=(
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_PACKFIX='0'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_VIEW_ARM='$arm'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_VIEW_MAP='$viewmap'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_VIEW_LOG_DIR='$out/views_log'"
  "++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=$hook"
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
  actor_rollout_ref.actor.calculate_entropy=True
  actor_rollout_ref.actor.entropy_coeff=0
  "++trainer.validation_data_dir=$out/validation"
)
if [[ "$check" == 1 ]]; then
  export PYTHONPATH="${G3_PY:?}:$root${PYTHONPATH:+:$PYTHONPATH}"
  common+=("++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$PYTHONPATH'")
  for v in VAD_G3_REPLAY VAD_G3_GRAD; do
    [[ -n "${!v:-}" ]] && common+=("++ray_kwargs.ray_init.runtime_env.env_vars.$v='${!v}'")
  done
fi

# train <total steps> <save freq> <tag>: one run, its own file-logger file (the verl FileLogger truncates its file).
train() {
  local total="$1" save="$2" tag="$3" start code metrics_file
  start=$(date +%s)
  metrics_file="$out/metrics_${tag}_$start.jsonl"
  echo "== train $tag: total $total save $save rollout.n ${ROLLOUT_N:-8} metrics $metrics_file start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
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

val_common=()
for c in "${common[@]}"; do
  case "$c" in
    ++trainer.validation_data_dir=*) ;;
    *=scripts.vad_packfix.hostmem.setup) val_common+=("${c%hostmem.setup}valtrim.setup") ;;
    *) val_common+=("$c") ;;
  esac
done

# validate <tag> <step> <val parquet> <dump dir>: the fast validation of one checkpoint in a val_only process.
validate() {
  local tag="$1" step="$2" file="$3" vdir="$4" start code
  start=$(date +%s)
  echo "== validate $tag: step $step val_batch_size $val_bs start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  [[ -n "${MEMWATCH_FAST_FLAG:-}" ]] && touch "$MEMWATCH_FAST_FLAG"
  VERL_FILE_LOGGER_PATH="$out/val_${tag}_$start.jsonl" TOTAL_TRAINING_STEPS="$steps" TRAINER_SAVE_FREQ="$steps" \
    ACTOR_OPTIMIZER_OFFLOAD=True BENCHMARK_EVAL_FILE="$file" BENCHMARK_EVAL_BATCH_SIZE="$val_bs" \
    bash scripts/vad_packfix/valguard.sh "$val_limit" "$out/console_$tag.log" "${VAD_CONTAINER_CGROUP:?}" \
    bash scripts/train_vad.sh --profile 4b -- "${val_common[@]}" "++trainer.validation_data_dir=$vdir" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VAD_VALTRIM='1'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$out/val_${tag}_$start.jsonl'" \
    trainer.val_before_train=True ++trainer.val_only=True trainer.test_freq=-1 \
    trainer.resume_mode=resume_path "++trainer.resume_from_path=$out/checkpoints/global_step_$step" \
    > "$out/console_$tag.log" 2>&1
  code=$?
  [[ -n "${MEMWATCH_FAST_FLAG:-}" ]] && rm -f "$MEMWATCH_FAST_FLAG"
  echo "validate $tag exit $code wall_seconds $(( $(date +%s) - start )) dump $(ls "$vdir/$step.jsonl" 2>/dev/null || echo none)" >> "$log"
  [[ $code -eq 0 && -f "$vdir/$step.jsonl" ]]
}

# validate_full <tag> <step>: the full set in 4 val_only processes (tags <tag>_p0 .. _p3), then one dump.
validate_full() {
  local tag="$1" step="$2" k parts=()
  for k in 0 1 2 3; do
    parts+=("$out/validation_parts/$step/$k/$step.jsonl")
    [[ -f "${parts[$k]}" ]] || validate "${tag}_p$k" "$step" "$split_dir/part_$k.parquet" \
      "$out/validation_parts/$step/$k" || return 1
  done
  "$py" scripts/vad_packfix/val_concat.py "$split_dir/VAL_SPLIT.json" "$out/validation/$step.jsonl" "${parts[@]}" \
    >> "$log" 2>&1
}

# Training (no judge on the GPU; judge_stub.py serves the model list and refuses each judge call).
latest=$(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [[ "$latest" -lt "$steps" ]]; then
  stub_port=$("$py" "$own_root/scripts/vad_packfix/free_port.py")
  "$py" "$own_root/scripts/vad_packfix/judge_stub.py" "$stub_port" "$BENCHMARK_EVAL_JUDGE_MODEL" "$out/judge_stub.log" &
  stub_pid=$!
  export BENCHMARK_EVAL_JUDGE_API_BASE="http://127.0.0.1:$stub_port/v1"
  for _ in $(seq 1 50); do
    curl -fsS --max-time 2 "$BENCHMARK_EVAL_JUDGE_API_BASE/models" > /dev/null 2>&1 && break
    sleep 0.2
  done
  echo "judge stub (training) pid $stub_pid port $stub_port" >> "$log"
  if [[ "$pilot" == 1 ]]; then
    if [[ ! -d "$out/checkpoints/global_step_1" ]]; then
      train 1 1 run1 || exit 1
    fi
    [[ -d "$out/checkpoints/global_step_1" ]] || { echo "no checkpoint after step 1" >> "$log"; exit 1; }
    echo "stop after step 1; resume" >> "$log"
    train 2 1 run2 || exit 1
    grep -n -i -E "resum|load(ing)? (from )?checkpoint|global_step_1" "$out/console_run2.log" | head -n 20 \
      | sed 's/^/resume evidence: /' >> "$log"
  elif [[ "$check" == 1 ]]; then
    train 1 1 check || exit 1
  else
    train 40 20 full || exit 1
  fi
  stop_stub
fi
latest=$(cat "$out/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
[[ "$latest" -ge "$steps" ]] || { echo "latest checkpoint $latest < $steps" >> "$log"; exit 1; }
for k in "${ckpt_steps[@]}"; do
  [[ -d "$out/checkpoints/global_step_$k" ]] || { echo "no checkpoint of step $k" >> "$log"; exit 1; }
done
echo "view lines: assign $(ls "$out"/views_log/assign_*.jsonl 2>/dev/null | wc -l) teacher $(ls "$out"/views_log/teacher_*.jsonl 2>/dev/null | wc -l) files" >> "$log"
if [[ "$check" == 1 ]]; then
  echo "== check run done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  exit 0
fi

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
if [[ ! -f "$out/validation/0.jsonl" ]]; then
  cp -f "${step0_from:?VAL_STEP0_FROM is needed}" "$out/validation/0.jsonl" || exit 1
  echo "step 0 validation copied from $step0_from sha256 $(sha256sum "$out/validation/0.jsonl" | cut -d' ' -f1) (Definition 5)" >> "$log"
fi
for k in "${ckpt_steps[@]}"; do
  [[ -f "$out/validation/$k.jsonl" ]] && continue
  if [[ "$pilot" == 1 ]]; then
    validate "val$k" "$k" "$val_file" "$out/validation" || exit 1
  else
    validate_full "val$k" "$k" || exit 1
  fi
done
echo "== score rollouts: start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
PYTHONPATH="$root" "$py" scripts/vad_packfix/rollout_acc.py "$out" "$steps" "$BENCHMARK_EVAL_JUDGE_API_BASE" \
  >> "$log" 2>&1 || { echo "rollout accuracy failed" >> "$log"; exit 1; }
stop_judge

# Held-out set B (Definition 6) on each checkpoint step, after the merge into a Hugging Face directory.
for k in "${ckpt_steps[@]}"; do
  [[ -f "$out/heldout/$k/HELDOUT_B.json" ]] && continue
  m="$out/merged/step_$k"
  if [[ ! -d "$m" ]]; then
    rm -rf "$m.tmp"
    start=$(date +%s)
    bash scripts/merge_checkpoint.sh "$out/checkpoints/global_step_$k" "$m.tmp" >> "$out/merge.log" 2>&1 \
      && mv "$m.tmp" "$m" || { echo "merge of step $k failed" >> "$log"; exit 1; }
    echo "merge step $k wall_seconds $(( $(date +%s) - start ))" >> "$log"
  fi
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 "$py" scripts/vad_view2/heldout.py "$root" "$m" "${HELDOUT_JSON:?}" \
    "$data/vad/benchmarks" "$out/heldout/$k" >> "$out/heldout_$k.log" 2>&1 \
    || { echo "heldout step $k failed" >> "$log"; exit 1; }
  echo "heldout step $k exit 0 wall_seconds $(( $(date +%s) - start ))" >> "$log"
done

"$py" scripts/vad_packfix/metrics.py "$out" "$steps" >> "$log" 2>&1 || { echo "metrics check failed" >> "$log"; exit 1; }
"$py" scripts/vad_packfix/val_acc.py "$out" >> "$log" 2>&1 || { echo "validation records failed" >> "$log"; exit 1; }
touch "$out/TRAIN_DONE"
echo "== arm $arm done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
