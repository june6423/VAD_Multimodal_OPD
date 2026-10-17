#!/bin/bash
# VLM-VAD-REPRO-001 Part B task 6 (HANDOFF-VLM-VAD-REPRO-001-R1, gate G6). 2-step training smoke.
# Usage: CUDA_VISIBLE_DEVICES=<4 container GPU indices> scripts/vad_repro/train_smoke.sh <output_root>
# Runs the main_ppo command of scripts/train_vad.sh lines 242-357 after configs/vad_4b.env, with
# the handoff Part A item 6 differences only. It bypasses the launcher checks (judge endpoint, W&B,
# input files). A full run of a later revision uses scripts/train_vad.sh.
# After the run: scripts/merge_checkpoint.sh on global_step_2, then the metric and disk summaries.
# Records: COMMAND.txt, console.log, STEP_METRICS.json, SMOKE_SUMMARY.txt, ROLLOUT_STEP1_SAMPLE.json,
# merged_step_2/config.json, gpu_disk_samples.csv, all under <output_root>.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
data=/workspace/harness/data
mkdir -p "$out"
cd "$root" || exit 1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must name the 4 GPUs of the task" >&2
  exit 2
fi

# Handoff Part A item 6 differences (1 to 5).
export MODEL_PATH="$data/vision-opd/Qwen3.5-4B"
export TEACHER_MODEL_PATH="$MODEL_PATH"
TASK_TRAIN_FILE="$data/vad/train_vad.parquet"
BENCHMARK_EVAL_FILE="$data/vad/benchmarks/opd_fast_benchmarks.parquet"
TRAINER_N_GPUS_PER_NODE=4
export TOTAL_TRAINING_STEPS=2 TRAINER_SAVE_FREQ=2
TRAINER_LOGGER='["console"]'
TRAINER_DEFAULT_LOCAL_DIR="$out/checkpoints"
TRAINER_ROLLOUT_DATA_DIR="$out/rollouts"
export HF_HUB_OFFLINE=1

# shellcheck source=/dev/null
source configs/vad_4b.env

# The remaining values are the train_vad.sh defaults (lines 61-120).
PROFILE=4b
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="VAD-${PROFILE_LABEL}-${RUN_STAMP}"
PROJECT_NAME="VAD"
CUSTOM_CHAT_TEMPLATE_FILE="${root}/chat_templates/perception_chat_template_qwen35.jinja"
TRAINER_NNODES=1
RAY_NUM_CPUS=32
RAY_INCLUDE_DASHBOARD=False
RAY_ADDRESS=local
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${MAX_MODEL_LEN}"
PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_USE_DYNAMIC_BSZ=True
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
ROLLOUT_AGENT_NUM_WORKERS=8
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
DATA_DATALOADER_NUM_WORKERS=8
DISTILLATION_ALPHA=0.5
VC_PROJECTION_RIDGE=1e-3
VC_STEP_ALPHA=1.0
VC_MAX_LOGIT_SHIFT=20.0
VC_MAX_SHIFT_NORM=null
VC_SUPPORT_MODE=student_topk
VC_DIRECTION_MODE=single
BENCHMARK_EVAL_ENABLE="True"
BENCHMARK_EVAL_FREQ="10"
BENCHMARK_EVAL_BENCHMARKS="vstar,zoombench,hrbench-4k,hrbench-8k"
BENCHMARK_EVAL_BATCH_SIZE=128
BENCHMARK_EVAL_JUDGE_API_BASE="http://127.0.0.1:8900/v1"
BENCHMARK_EVAL_JUDGE_API_KEY="EMPTY"
BENCHMARK_EVAL_JUDGE_MODEL="openai/gpt-oss-120b"
BENCHMARK_EVAL_JUDGE_MAX_TOKENS=2048
BENCHMARK_EVAL_JUDGE_PARALLEL_WORKERS=8
BENCHMARK_EVAL_JUDGE_REQUEST_TIMEOUT=600
BENCHMARK_EVAL_JUDGE_MIN_REQUEST_INTERVAL=0.0
BENCHMARK_EVAL_JUDGE_ENABLE_THINKING="False"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
WANDB_DIR="${root}/wandb"

# Environment of train_vad.sh lines 219-239, with PYTHONPATH = the clone root (difference 5).
export PYTHONPATH="${root}"
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

# train_vad.sh lines 242-357. Changed overrides: total_training_steps (2, both), test_freq (-1),
# save_freq (2), logger (console), default_local_dir and rollout_data_dir (output root).
cmd=(python3 -m verl.trainer.main_ppo --config-name vopd
  data.train_files="[\"${TASK_TRAIN_FILE}\"]"
  data.val_files="[\"${BENCHMARK_EVAL_FILE}\"]"
  data.filter_overlong_prompts=False
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.truncation=error
  data.shuffle=True
  data.trust_remote_code=True
  data.return_multi_modal_inputs=True
  data.image_key=images
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size="${BENCHMARK_EVAL_BATCH_SIZE}"
  data.dataloader_num_workers="${DATA_DATALOADER_NUM_WORKERS}"
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.custom_chat_template_file="${CUSTOM_CHAT_TEMPLATE_FILE}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  actor_rollout_ref.actor.optim.total_training_steps="${TOTAL_TRAINING_STEPS}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.use_dynamic_bsz="${ACTOR_USE_DYNAMIC_BSZ}"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.seed=42
  actor_rollout_ref.actor.data_loader_seed=42
  actor_rollout_ref.actor.clip_ratio_high=0.3
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.calculate_entropy=False
  actor_rollout_ref.actor.policy_loss.loss_mode=vopd
  actor_rollout_ref.actor.self_distillation.full_logit_distillation=True
  actor_rollout_ref.actor.self_distillation.distillation_topk=100
  actor_rollout_ref.actor.self_distillation.distillation_add_tail=True
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240
  actor_rollout_ref.actor.self_distillation.is_clip=2.0
  actor_rollout_ref.actor.self_distillation.teacher_always_on=True
  actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed
  actor_rollout_ref.actor.self_distillation.teacher_model_path="${TEACHER_MODEL_PATH}"
  actor_rollout_ref.actor.self_distillation.teacher_regularization=ema
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05
  actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images
  actor_rollout_ref.actor.self_distillation.counterfactual_teacher_image_key=degraded_bbox_images
  actor_rollout_ref.actor.self_distillation.target_mode=vc
  actor_rollout_ref.actor.self_distillation.vc_target_variant=budgeted_asvc
  actor_rollout_ref.actor.self_distillation.vc_positive_budget_cap="${VC_POSITIVE_BUDGET_CAP}"
  actor_rollout_ref.actor.self_distillation.vc_projection_ridge="${VC_PROJECTION_RIDGE}"
  actor_rollout_ref.actor.self_distillation.vc_step_alpha="${VC_STEP_ALPHA}"
  actor_rollout_ref.actor.self_distillation.vc_max_logit_shift="${VC_MAX_LOGIT_SHIFT}"
  actor_rollout_ref.actor.self_distillation.vc_max_shift_norm="${VC_MAX_SHIFT_NORM}"
  actor_rollout_ref.actor.self_distillation.vc_use_rho_gate=False
  actor_rollout_ref.actor.self_distillation.vc_nonvisual_anchor_eta="${VC_NONVISUAL_ANCHOR_ETA}"
  actor_rollout_ref.actor.self_distillation.vc_support_mode="${VC_SUPPORT_MODE}"
  actor_rollout_ref.actor.self_distillation.vc_direction_mode="${VC_DIRECTION_MODE}"
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True
  actor_rollout_ref.actor.self_distillation.alpha="${DISTILLATION_ALPHA}"
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"
  actor_rollout_ref.ref.fsdp_config.seed=42
  algorithm.rollout_correction.rollout_is=token
  algorithm.rollout_correction.rollout_is_threshold=2.0
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=False
  algorithm.use_kl_in_reward=False
  reward_model.enable=False
  reward_model.use_reward_loop=False
  reward_model.reward_manager=batch
  reward_manager.name=batch
  custom_reward_function.path="${root}/verl/utils/reward_score/opd_benchmark_judge.py"
  custom_reward_function.name=compute_score_batched
  ++custom_reward_function.reward_kwargs.judge_api_base="'${BENCHMARK_EVAL_JUDGE_API_BASE}'"
  ++custom_reward_function.reward_kwargs.judge_model="${BENCHMARK_EVAL_JUDGE_MODEL}"
  ++custom_reward_function.reward_kwargs.judge_max_tokens="${BENCHMARK_EVAL_JUDGE_MAX_TOKENS}"
  ++custom_reward_function.reward_kwargs.parallel_workers="${BENCHMARK_EVAL_JUDGE_PARALLEL_WORKERS}"
  ++custom_reward_function.reward_kwargs.request_timeout="${BENCHMARK_EVAL_JUDGE_REQUEST_TIMEOUT}"
  ++custom_reward_function.reward_kwargs.min_request_interval="${BENCHMARK_EVAL_JUDGE_MIN_REQUEST_INTERVAL}"
  ++custom_reward_function.reward_kwargs.enable_thinking=False
  ++custom_reward_function.reward_kwargs.thinking_extra_body_style=direct
  ++custom_reward_function.reward_kwargs.show_progress=False
  critic.model.path="${MODEL_PATH}"
  critic.data_loader_seed=42
  critic.model.fsdp_config.seed=42
  trainer.project_name="${PROJECT_NAME}"
  trainer.group_name="${EXPERIMENT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.logger="${TRAINER_LOGGER}"
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}"
  trainer.nnodes="${TRAINER_NNODES}"
  trainer.save_freq="${TRAINER_SAVE_FREQ}"
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.max_actor_ckpt_to_keep="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP}"
  trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.default_local_dir="${TRAINER_DEFAULT_LOCAL_DIR}"
  trainer.rollout_data_dir="${TRAINER_ROLLOUT_DATA_DIR}"
  trainer.resume_mode=auto
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
  +ray_kwargs.ray_init.include_dashboard="${RAY_INCLUDE_DASHBOARD}"
)

{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd $(pwd)"
  echo "git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== environment (names with KEY, TOKEN, SECRET or PASSWORD show <set> or <unset>)"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|PYTHONPATH|HF_HUB_OFFLINE|VLLM_USE_V1|RAY_ADDRESS|WANDB_MODE|WANDB_DIR|PYTHONUNBUFFERED|OPD_BENCH_JUDGE_[A-Z_]*|BENCHMARK_EVAL_[A-Z]*)=' \
    | grep -Ev '^[A-Z_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*=' | sort
  for name in $(compgen -e | grep -E 'KEY|TOKEN|SECRET|PASSWORD' | sort); do
    if [[ -n "${!name}" ]]; then echo "$name=<set>"; else echo "$name=<unset>"; fi
  done
  echo "== verl location (python3 -c, same cwd and environment as the launch)"
  echo "VERL_FILE $(python3 -c 'import verl; print(verl.__file__)')"
  echo "== nvidia-smi (container indices)"
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total --format=csv,noheader
  echo "== command (one argument per line)"
  printf '%q\n' "${cmd[@]}"
} > "$out/COMMAND.txt"

# GPU memory and disk sampler: every 15 s, with the count of finished steps in the console log.
sampler() {
  echo "epoch,steps_done,gpu_index,gpu_uuid,memory_used_mib,workspace_avail_kib,output_root_bytes"
  while true; do
    local now steps avail used
    now=$(date +%s)
    steps=$(grep -c 'step:[0-9]* - ' "$out/console.log" 2>/dev/null)
    avail=$(df -k --output=avail /workspace | tail -n 1 | tr -d ' ')
    used=$(du -sb "$out" 2>/dev/null | cut -f1)
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
      while IFS=', ' read -r idx uuid mem; do echo "$now,${steps:-0},$idx,$uuid,$mem,$avail,$used"; done
    sleep 15
  done
}
sampler > "$out/gpu_disk_samples.csv" 2>/dev/null &
sampler_pid=$!

start=$(date +%s)
"${cmd[@]}" 2>&1 | tee "$out/console.log"
train_status=${PIPESTATUS[0]}
end=$(date +%s)
sleep 20
kill "$sampler_pid" 2>/dev/null

merge_status=-1
if [[ -d "$out/checkpoints/global_step_2" ]]; then
  bash scripts/merge_checkpoint.sh "$out/checkpoints/global_step_2" "$out/merged_step_2" > "$out/merge.log" 2>&1
  merge_status=$?
fi

{
  echo "train exit $train_status wall_seconds $((end - start))"
  echo "merge exit $merge_status"
  echo "== verl location"
  grep -m 1 '^VERL_FILE' "$out/COMMAND.txt"
  echo "console.log lines with /VAD_Multimodal_OPD/verl/: $(grep -c '/VAD_Multimodal_OPD/verl/' "$out/console.log")"
  echo "console.log lines with /vision-opd/verl/: $(grep -c '/vision-opd/verl/' "$out/console.log")"
  grep -m 3 '/vision-opd/verl/' "$out/console.log"
  echo "== error lines in console.log (first 40)"
  grep -n -E 'OutOfMemory|out of memory|cudaErrorCallRequiresNewerDriver|CUDA_ERROR_UNSUPPORTED_PTX_VERSION|driver too old|Traceback|Error' "$out/console.log" | head -n 40
  echo "== merge log tail"; tail -n 20 "$out/merge.log" 2>/dev/null
  echo "== merged config.json"; cat "$out/merged_step_2/config.json" 2>/dev/null
  echo "== merged files (bytes path)"
  (cd "$out" && find merged_step_2 -type f -printf '%s %p\n' 2>/dev/null | sort -k2)
  echo "== checkpoint sizes"
  du -sb "$out/checkpoints/global_step_2" "$out/checkpoints/global_step_2/actor" "$out/merged_step_2" 2>/dev/null
  (cd "$out" && find checkpoints -maxdepth 3 -printf '%s %p\n' 2>/dev/null | sort -k2)
  echo "== df -h /workspace after"; df -h /workspace
  echo "== metrics"
  python - "$out" <<'EOF'
import csv
import json
import re
import sys
from collections import defaultdict
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
keys = ["self_distillation/raw_jsd_token_mean", "self_distillation/vc_rho_single.mean",
        "self_distillation/basvc_alpha_pos.mean", "response_length/mean", "response_length/max",
        "response_length/clip_ratio", "timing_s/step", "timing_s/gen", "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb", "perf/cpu_memory_used_gb"]
for step in sorted(steps):
    print("step", step)
    for key in keys:
        print(f"  {key} {steps[step].get(key)}")
peak = defaultdict(int)
disk = {}
with open(out / "gpu_disk_samples.csv") as handle:
    for row in csv.DictReader(handle):
        try:
            peak[row["gpu_index"]] = max(peak[row["gpu_index"]], int(row["memory_used_mib"]))
            disk.setdefault(int(row["steps_done"]), (row["workspace_avail_kib"], row["output_root_bytes"]))
        except (KeyError, ValueError):
            continue
print("nvidia-smi peak memory_used_mib per container GPU index", dict(peak))
for done, (avail, used) in sorted(disk.items()):
    print(f"first sample with steps_done={done}: workspace_avail_kib {avail} output_root_bytes {used}")
EOF
} > "$out/SMOKE_SUMMARY.txt" 2>&1

python - "$out" <<'EOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
path = out / "rollouts" / "1.jsonl"
records = []
if path.exists():
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records.append({k: (v[:4000] if isinstance(v, str) else v) for k, v in row.items()})
            if len(records) == 3:
                break
(out / "ROLLOUT_STEP1_SAMPLE.json").write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
EOF

[[ $train_status -eq 0 && $merge_status -eq 0 ]]
