#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: bash scripts/merge_checkpoint.sh GLOBAL_STEP_DIR [OUTPUT_DIR]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  usage >&2
  exit 2
fi

SOURCE_DIR="${1%/}"
ACTOR_DIR="${SOURCE_DIR}/actor"
OUTPUT_DIR="${2:-${SOURCE_DIR}_merged}"

if [[ ! -d "${ACTOR_DIR}" ]]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists; choose a fresh path: ${OUTPUT_DIR}" >&2
  exit 1
fi

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${OUTPUT_DIR}"

echo "Merged checkpoint: ${OUTPUT_DIR}"
