#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_vad.sh [--profile 4b|9b] [--dry-run] [-- HYDRA_OVERRIDE ...]

Required before a real launch:
  1. data/train_vad.parquet
  2. data/benchmarks/opd_fast_benchmarks.parquet
  3. WANDB_API_KEY when W&B logging is enabled
  4. A reachable benchmark judge endpoint

All profiles keep in-training evaluation enabled for:
  vstar, zoombench, hrbench-4k, hrbench-8k
at a fixed frequency of every 10 optimizer steps.
EOF
}

PROFILE="${VAD_PROFILE:-4b}"
DRY_RUN=0
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:?--profile requires 4b or 9b}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_FILE="${PROJECT_ROOT}/configs/vad_${PROFILE}.env"
if [[ ! -f "${PROFILE_FILE}" ]]; then
  echo "Unsupported profile ${PROFILE!r}; expected 4b or 9b." >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${PROFILE_FILE}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-VAD-${PROFILE_LABEL}-${RUN_STAMP}}"
PROJECT_NAME="${PROJECT_NAME:-VAD}"

TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${PROJECT_ROOT}/data/train_vad.parquet}"
BENCHMARK_EVAL_FILE="${BENCHMARK_EVAL_FILE:-${PROJECT_ROOT}/data/benchmarks/opd_fast_benchmarks.parquet}"
CUSTOM_CHAT_TEMPLATE_FILE="${CUSTOM_CHAT_TEMPLATE_FILE:-${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-8}"
TRAINER_NNODES="${TRAINER_NNODES:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
RAY_ADDRESS="${RAY_ADDRESS:-local}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-True}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-8}"

DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.5}"
VC_PROJECTION_RIDGE="${VC_PROJECTION_RIDGE:-1e-3}"
VC_STEP_ALPHA="${VC_STEP_ALPHA:-1.0}"
VC_MAX_LOGIT_SHIFT="${VC_MAX_LOGIT_SHIFT:-20.0}"
VC_MAX_SHIFT_NORM="${VC_MAX_SHIFT_NORM:-null}"
VC_SUPPORT_MODE="${VC_SUPPORT_MODE:-student_topk}"
VC_DIRECTION_MODE="${VC_DIRECTION_MODE:-single}"

# These are release invariants. They intentionally cannot be disabled by a
# profile file or an inherited shell environment.
BENCHMARK_EVAL_ENABLE="True"
BENCHMARK_EVAL_FREQ="10"
BENCHMARK_EVAL_BENCHMARKS="vstar,zoombench,hrbench-4k,hrbench-8k"
BENCHMARK_EVAL_SCORER="judge_qwenlm"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-128}"
BENCHMARK_EVAL_JUDGE_API_BASE="${BENCHMARK_EVAL_JUDGE_API_BASE:-${JUDGE_API_BASE:-http://127.0.0.1:8900/v1}}"
BENCHMARK_EVAL_JUDGE_API_KEY="${BENCHMARK_EVAL_JUDGE_API_KEY:-${JUDGE_API_KEY:-EMPTY}}"
BENCHMARK_EVAL_JUDGE_MODEL="${BENCHMARK_EVAL_JUDGE_MODEL:-${JUDGE_MODEL:-openai/gpt-oss-120b}}"
BENCHMARK_EVAL_JUDGE_MAX_TOKENS="${BENCHMARK_EVAL_JUDGE_MAX_TOKENS:-2048}"
BENCHMARK_EVAL_JUDGE_PARALLEL_WORKERS="${BENCHMARK_EVAL_JUDGE_PARALLEL_WORKERS:-8}"
BENCHMARK_EVAL_JUDGE_REQUEST_TIMEOUT="${BENCHMARK_EVAL_JUDGE_REQUEST_TIMEOUT:-600}"
BENCHMARK_EVAL_JUDGE_MIN_REQUEST_INTERVAL="${BENCHMARK_EVAL_JUDGE_MIN_REQUEST_INTERVAL:-0.0}"
BENCHMARK_EVAL_JUDGE_ENABLE_THINKING="False"

TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
TRAINER_ROLLOUT_DATA_DIR="${TRAINER_ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}}"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-null}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"

if (( ${#EXPERIMENT_NAME} > 120 )); then
  echo "EXPERIMENT_NAME is ${#EXPERIMENT_NAME} characters; keep it at or below 120." >&2
  exit 2
fi

print_configuration() {
  echo "VAD training preflight"
  echo "  profile: ${PROFILE}"
  echo "  experiment: ${EXPERIMENT_NAME}"
  echo "  W&B project: ${PROJECT_NAME}"
  echo "  model: ${MODEL_PATH}"
  echo "  fixed teacher: ${TEACHER_MODEL_PATH}"
  echo "  training data: ${TASK_TRAIN_FILE}"
  echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "  train batch / mini batch / rollouts: ${TRAIN_BATCH_SIZE} / ${PPO_MINI_BATCH_SIZE} / ${ROLLOUT_N}"
  echo "  target: budgeted_asvc"
  echo "  positive cap / weak anchor: ${VC_POSITIVE_BUDGET_CAP} / ${VC_NONVISUAL_ANCHOR_ETA}"
  echo "  benchmark eval enabled: ${BENCHMARK_EVAL_ENABLE}"
  echo "  benchmark eval frequency: ${BENCHMARK_EVAL_FREQ}"
  echo "  benchmark eval set: ${BENCHMARK_EVAL_BENCHMARKS}"
  echo "  benchmark judge: ${BENCHMARK_EVAL_JUDGE_MODEL} at ${BENCHMARK_EVAL_JUDGE_API_BASE}"
  echo "  total epochs / steps: ${TRAINER_TOTAL_EPOCHS} / ${TOTAL_TRAINING_STEPS}"
}

path_or_hub_id() {
  local value="$1"
  if [[ "${value}" == /* || "${value}" == ./* || "${value}" == ../* ]]; then
    [[ -e "${value}" ]]
  else
    [[ "${value}" == */* ]]
  fi
}

check_judge_endpoint() {
  local response
  response="$(
    curl -fsS --max-time 20 \
      -H "Authorization: Bearer ${BENCHMARK_EVAL_JUDGE_API_KEY}" \
      "${BENCHMARK_EVAL_JUDGE_API_BASE%/}/models"
  )"
  if [[ "${response}" != *"\"${BENCHMARK_EVAL_JUDGE_MODEL}\""* ]]; then
    echo "Judge endpoint is reachable but does not advertise ${BENCHMARK_EVAL_JUDGE_MODEL}." >&2
    exit 1
  fi
}

print_configuration
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete; filesystem, W&B, and endpoint checks were skipped."
  exit 0
fi

for required_file in "${TASK_TRAIN_FILE}" "${BENCHMARK_EVAL_FILE}" "${CUSTOM_CHAT_TEMPLATE_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file is missing: ${required_file}" >&2
    exit 1
  fi
done
if ! path_or_hub_id "${MODEL_PATH}"; then
  echo "MODEL_PATH is neither a local path nor a Hugging Face repository ID: ${MODEL_PATH}" >&2
  exit 1
fi
if ! path_or_hub_id "${TEACHER_MODEL_PATH}"; then
  echo "TEACHER_MODEL_PATH is neither a local path nor a Hugging Face repository ID: ${TEACHER_MODEL_PATH}" >&2
  exit 1
fi
if [[ "${TRAINER_LOGGER}" == *wandb* && -z "${WANDB_API_KEY:-}" ]]; then
  echo "W&B logging is enabled but WANDB_API_KEY is not exported." >&2
  exit 1
fi
check_judge_endpoint

mkdir -p \
  "${TRAINER_DEFAULT_LOCAL_DIR}" \
  "${TRAINER_ROLLOUT_DATA_DIR}" \
  "${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}" \
  "${WANDB_DIR}"

MANIFEST="${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}/launch.env"
{
  echo "profile=${PROFILE}"
  echo "experiment_name=${EXPERIMENT_NAME}"
  echo "project_name=${PROJECT_NAME}"
  echo "model_path=${MODEL_PATH}"
  echo "teacher_model_path=${TEACHER_MODEL_PATH}"
  echo "task_train_file=${TASK_TRAIN_FILE}"
  echo "benchmark_eval_enable=${BENCHMARK_EVAL_ENABLE}"
  echo "benchmark_eval_freq=${BENCHMARK_EVAL_FREQ}"
  echo "benchmark_eval_benchmarks=${BENCHMARK_EVAL_BENCHMARKS}"
  echo "benchmark_eval_file=${BENCHMARK_EVAL_FILE}"
  echo "benchmark_eval_judge_api_base=${BENCHMARK_EVAL_JUDGE_API_BASE}"
  echo "benchmark_eval_judge_model=${BENCHMARK_EVAL_JUDGE_MODEL}"
  echo "benchmark_eval_judge_max_tokens=${BENCHMARK_EVAL_JUDGE_MAX_TOKENS}"
  echo "seed=42"
  echo "created_at=$(date --iso-8601=seconds)"
} >"${MANIFEST}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export RAY_ADDRESS
export WANDB_DIR
export WANDB_MODE="${WANDB_MODE:-online}"
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

export OPD_BENCH_JUDGE_API_BASE="${BENCHMARK_EVAL_JUDGE_API_BASE}"
export OPD_BENCH_JUDGE_API_KEY="${BENCHMARK_EVAL_JUDGE_API_KEY}"
export OPD_BENCH_JUDGE_MODEL="${BENCHMARK_EVAL_JUDGE_MODEL}"
export OPD_BENCH_JUDGE_MAX_TOKENS="${BENCHMARK_EVAL_JUDGE_MAX_TOKENS}"
export OPD_BENCH_JUDGE_PARALLEL_WORKERS="${BENCHMARK_EVAL_JUDGE_PARALLEL_WORKERS}"
export OPD_BENCH_JUDGE_REQUEST_TIMEOUT="${BENCHMARK_EVAL_JUDGE_REQUEST_TIMEOUT}"
export OPD_BENCH_JUDGE_MIN_REQUEST_INTERVAL="${BENCHMARK_EVAL_JUDGE_MIN_REQUEST_INTERVAL}"
export OPD_BENCH_JUDGE_ENABLE_THINKING="${BENCHMARK_EVAL_JUDGE_ENABLE_THINKING}"
export BENCHMARK_EVAL_ENABLE
export BENCHMARK_EVAL_FREQ
export BENCHMARK_EVAL_BENCHMARKS

cd "${PROJECT_ROOT}"
exec python3 -m verl.trainer.main_ppo --config-name vopd \
  data.train_files="[\"${TASK_TRAIN_FILE}\"]" \
  data.val_files="[\"${BENCHMARK_EVAL_FILE}\"]" \
  data.filter_overlong_prompts=False \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.truncation=error \
  data.shuffle=True \
  data.trust_remote_code=True \
  data.return_multi_modal_inputs=True \
  data.image_key=images \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${BENCHMARK_EVAL_BATCH_SIZE}" \
  data.dataloader_num_workers="${DATA_DATALOADER_NUM_WORKERS}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.custom_chat_template_file="${CUSTOM_CHAT_TEMPLATE_FILE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${ACTOR_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.seed=42 \
  actor_rollout_ref.actor.data_loader_seed=42 \
  actor_rollout_ref.actor.clip_ratio_high=0.3 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
  actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
  actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
  actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
  actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
  actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
  actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed \
  actor_rollout_ref.actor.self_distillation.teacher_model_path="${TEACHER_MODEL_PATH}" \
  actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
  actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
  actor_rollout_ref.actor.self_distillation.counterfactual_teacher_image_key=degraded_bbox_images \
  actor_rollout_ref.actor.self_distillation.target_mode=vc \
  actor_rollout_ref.actor.self_distillation.vc_target_variant=budgeted_asvc \
  actor_rollout_ref.actor.self_distillation.vc_positive_budget_cap="${VC_POSITIVE_BUDGET_CAP}" \
  actor_rollout_ref.actor.self_distillation.vc_projection_ridge="${VC_PROJECTION_RIDGE}" \
  actor_rollout_ref.actor.self_distillation.vc_step_alpha="${VC_STEP_ALPHA}" \
  actor_rollout_ref.actor.self_distillation.vc_max_logit_shift="${VC_MAX_LOGIT_SHIFT}" \
  actor_rollout_ref.actor.self_distillation.vc_max_shift_norm="${VC_MAX_SHIFT_NORM}" \
  actor_rollout_ref.actor.self_distillation.vc_use_rho_gate=False \
  actor_rollout_ref.actor.self_distillation.vc_nonvisual_anchor_eta="${VC_NONVISUAL_ANCHOR_ETA}" \
  actor_rollout_ref.actor.self_distillation.vc_support_mode="${VC_SUPPORT_MODE}" \
  actor_rollout_ref.actor.self_distillation.vc_direction_mode="${VC_DIRECTION_MODE}" \
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
  actor_rollout_ref.actor.self_distillation.alpha="${DISTILLATION_ALPHA}" \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  actor_rollout_ref.ref.fsdp_config.seed=42 \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  reward_model.use_reward_loop=False \
  reward_model.reward_manager=batch \
  reward_manager.name=batch \
  custom_reward_function.path="${PROJECT_ROOT}/verl/utils/reward_score/opd_benchmark_judge.py" \
  custom_reward_function.name=compute_score_batched \
  ++custom_reward_function.reward_kwargs.judge_api_base="'${BENCHMARK_EVAL_JUDGE_API_BASE}'" \
  ++custom_reward_function.reward_kwargs.judge_model="${BENCHMARK_EVAL_JUDGE_MODEL}" \
  ++custom_reward_function.reward_kwargs.judge_max_tokens="${BENCHMARK_EVAL_JUDGE_MAX_TOKENS}" \
  ++custom_reward_function.reward_kwargs.parallel_workers="${BENCHMARK_EVAL_JUDGE_PARALLEL_WORKERS}" \
  ++custom_reward_function.reward_kwargs.request_timeout="${BENCHMARK_EVAL_JUDGE_REQUEST_TIMEOUT}" \
  ++custom_reward_function.reward_kwargs.min_request_interval="${BENCHMARK_EVAL_JUDGE_MIN_REQUEST_INTERVAL}" \
  ++custom_reward_function.reward_kwargs.enable_thinking=False \
  ++custom_reward_function.reward_kwargs.thinking_extra_body_style=direct \
  ++custom_reward_function.reward_kwargs.show_progress=False \
  critic.model.path="${MODEL_PATH}" \
  critic.data_loader_seed=42 \
  critic.model.fsdp_config.seed=42 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.group_name="${EXPERIMENT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
  trainer.nnodes="${TRAINER_NNODES}" \
  trainer.save_freq="${TRAINER_SAVE_FREQ}" \
  trainer.test_freq="${BENCHMARK_EVAL_FREQ}" \
  trainer.val_before_train=False \
  trainer.max_actor_ckpt_to_keep="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${TRAINER_DEFAULT_LOCAL_DIR}" \
  trainer.rollout_data_dir="${TRAINER_ROLLOUT_DATA_DIR}" \
  trainer.resume_mode=auto \
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
  +ray_kwargs.ray_init.include_dashboard="${RAY_INCLUDE_DASHBOARD}" \
  "${EXTRA_ARGS[@]}"
