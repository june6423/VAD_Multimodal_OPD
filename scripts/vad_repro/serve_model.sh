#!/bin/bash
# VLM-VAD-REPRO-001 R3 model server (HANDOFF-VLM-VAD-REPRO-001-R3, gates G2 to G4), container VAD_A100_001.
# Usage: GPU_UUID=<uuid of the task GPU> scripts/vad_repro/serve_model.sh <model path> <served model name>
# Runs the vllm serve command of scripts/serve_checkpoint.sh (host 0.0.0.0, port 8000, max_model_len
# 32768, gpu_memory_utilization 0.85, trust_remote_code) with tensor parallel size 1 and
# --dtype bfloat16 (researcher decision "(ii)로 가자"). It prints the full command, then execs vllm.
# The GPU_UUID maps to the PCI-ordered container index.
set -euo pipefail

model="$1"
served="$2"
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1
index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="${GPU_UUID:-}" '$2 == u {print $1}')
if [[ -z "$index" ]]; then
  echo "GPU_UUID ${GPU_UUID:-<unset>} does not match a GPU in this container" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$index"
cmd=(vllm serve "$model" --host 0.0.0.0 --port 8000 --served-model-name "$served"
  --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.85 --trust-remote-code
  --dtype bfloat16)
echo "GPU_UUID $GPU_UUID -> CUDA_VISIBLE_DEVICES $index"
echo "COMMAND $(printf '%q ' "${cmd[@]}")"
exec "${cmd[@]}"
