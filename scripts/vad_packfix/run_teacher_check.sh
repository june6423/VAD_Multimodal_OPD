#!/bin/bash
# Bit-equal check of candidate 2 of hostmem.py (DIRECTIVE-VLM-VAD-PACKFIX-001-005 actions 5 and 6). Check only, no
# update. Steps:
#   1. Stops with exit 1 if an orphan Ray process of an earlier task is alive (DIRECTIVE-VLM-VAD-PACKFIX-001-004).
#   2. scripts/vad_packfix/offload_selftest.py with CUDA (OFFLOAD_SELFTEST.json): the unit checks of candidates 2, 4
#      and 5 on the GPU (the b200 env has no pytest). Exit 1 at a failed check.
#   3. Starts arm F as run_hostmem_check.sh does (same environment, judge next to training, same overrides as
#      run_packfix.sh), with trainer.val_before_train=False and TOTAL_TRAINING_STEPS=1, and the Ray worker hook
#      scripts/vad_packfix/teacher_check.py in place of hostmem.py (it installs hostmem.py without the teacher cast).
#      At the first update_policy call the hook compares the teacher forwards of the first TEACHER_CHECK_GROUPS prompt
#      groups with the fp32 teacher and with the cast bf16 teacher (TEACHER_CHECK.json). Then it stops the run.
#   4. memwatch.sh logs the container charge; oom_kill before and after.
# Exit 0 only when each self-test check passes and each compared teacher tensor is equal.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [JUDGE_GPU_UTIL] [ROLLOUT_GPU_UTIL] \
#   [TEACHER_CHECK_GROUPS=4] scripts/vad_packfix/run_teacher_check.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}"
judge_util="${JUDGE_GPU_UTIL:-0.40}" rollout_util="${ROLLOUT_GPU_UTIL:-0.25}"
groups="${TEACHER_CHECK_GROUPS:-4}"
data="$b200/harness/data"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_TEACHER_CHECK.txt"
export PYTHONDONTWRITEBYTECODE=1
cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
while [[ "$cg" != /sys/fs/cgroup && ( ! -r "$cg/memory.max" || "$(cat "$cg/memory.max")" == max ) ]]; do
  cg="$(dirname "$cg")"
done
oom_kill() { awk '$1=="oom_kill"{print $2}' "$cg/memory.events"; }
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) TEACHER_CHECK ARM F groups $groups"
  echo "judge_gpu_util $judge_util rollout_gpu_util $rollout_util"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  echo "container cgroup $cg memory.max $(cat "$cg/memory.max") memory.current $(cat "$cg/memory.current")"
  echo "oom_kill before $(oom_kill)"
} >> "$log" 2>&1
orphans=$(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^ray::/' | wc -l)
echo "orphan ray processes before the run: $orphans" >> "$log"
[[ "$orphans" -eq 0 ]] || { ps -eo pid,ppid,etime,rss,args | awk '$2==1 && $5 ~ /^ray::/' >> "$log"; exit 1; }

PYTHONPATH="$root" "$py" scripts/vad_packfix/offload_selftest.py "$out/OFFLOAD_SELFTEST.json" >> "$log" 2>&1
code=$?
echo "offload_selftest.py exit $code" >> "$log"
[[ $code -eq 0 ]] || exit 1

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
export EXPERIMENT_NAME="PACKFIX-TEACHER-CHECK"
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
  ++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=scripts.vad_packfix.teacher_check.setup \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
  actor_rollout_ref.actor.calculate_entropy=True \
  actor_rollout_ref.actor.entropy_coeff=0 \
  "++trainer.validation_data_dir=$out/validation" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$out/metrics.jsonl'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.TEACHER_CHECK_OUT='$out'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.TEACHER_CHECK_GROUPS='$groups'" \
  trainer.val_before_train=False \
  > "$out/console.log" 2>&1
code=$?
sleep 10  # memwatch takes the samples after the end
{
  echo "train exit $code (a nonzero exit is expected: the hook stops the run) wall_seconds $(( $(date +%s) - start ))"
  echo "oom_kill after $(oom_kill)"
  echo "hook install lines: $(grep -c 'TEACHER_CHECK hook installed' "$out/console.log")"
  grep -h -E "VAD_HOSTMEM pid=[0-9]+ (reference model removed|teacher flat params|copy )|TEACHER_CHECK pid=" "$out/console.log" \
    | sed 's/^.*\(VAD_HOSTMEM\|TEACHER_CHECK\)/\1/' | head -n 40
  echo "orphan ray processes after the run: $(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^ray::/' | wc -l)"
} >> "$log" 2>&1
"$py" scripts/vad_packfix/mempeak.py "$out/memwatch.log" "$out/MEMPEAK.json" >> "$log" 2>&1
PYTHONPATH="$root" "$py" scripts/vad_packfix/teacher_check.py "$out/TEACHER_CHECK.json" >> "$log" 2>&1
check=$?
echo "check exit $check" >> "$log"
rm -rf "$out/checkpoints"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit "$check"
