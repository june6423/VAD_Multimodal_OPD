#!/bin/bash
# VLM-VAD-REPRO-001 Part B task 1 (HANDOFF-VLM-VAD-REPRO-001-R1, gates G1 and G2). No GPU work.
# Usage: scripts/vad_repro/env_check.sh <output_root>
# Records the clone state, runs the editable install (the only package command), and records
# the verl import path and the pip freeze diff against the image freeze.
# Writes <output_root>/ENV_CHECK.txt and <output_root>/pip-freeze.txt.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
mkdir -p "$out"
cd "$root" || exit 1
g() { git -c safe.directory="$root" "$@"; }
export PIP_NO_CACHE_DIR=1

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== container identity (hostname, cgroup)"; hostname; head -n 3 /proc/self/cgroup
  echo "== image environment"; env | grep -E '^(CUDA_VERSION|NV_CUDA_LIB_VERSION|NVIDIA_REQUIRE_CUDA|CONDA_DEFAULT_ENV|PATH)=' | sort
  echo "== clone state (before install)"
  echo "-- git remote -v"; g remote -v
  echo "-- git rev-parse main"; g rev-parse main
  echo "-- git rev-parse HEAD"; g rev-parse HEAD
  echo "-- git branch --show-current"; g branch --show-current
  echo "-- git status --short (begin)"; g status --short; echo "-- git status --short (end)"
  echo "== python"; command -v python; python --version
  echo "== pip install -e . --no-deps"
  pip install -e . --no-deps
  echo "pip install exit $?"
  echo "== verl import path (cwd = clone root)"
  python -c "import verl; print(verl.__file__)"
  echo "== verl import path (cwd = /, PYTHONPATH unset), diagnostic"
  (cd / && env -u PYTHONPATH python -c "import verl; print(verl.__file__)")
  echo "== pip show vad-opd"; pip show vad-opd
  echo "== pip freeze diff (/opt/vision-opd/pip-freeze.txt -> now)"
  pip freeze > "$out/pip-freeze.txt"
  diff /opt/vision-opd/pip-freeze.txt "$out/pip-freeze.txt"
  echo "diff exit $?"
  echo "== git status --short after install (begin)"; g status --short; echo "-- (end)"
  echo "== nvidia-smi"; nvidia-smi
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total --format=csv
  echo "== df -h /workspace"; df -h /workspace
  echo "== Hugging Face variables"
  echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE-<unset>}"
  echo "HF_HOME=${HF_HOME-<unset>}"
  echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE-<unset>} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE-<unset>}"
} > "$out/ENV_CHECK.txt" 2>&1

grep -q "^pip install exit 0$" "$out/ENV_CHECK.txt"
