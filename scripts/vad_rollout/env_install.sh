#!/bin/bash
# VLM-VAD-ROLLOUT-001 environments (HANDOFF-VLM-VAD-ROLLOUT-001-R1, Part A item 5). No GPU work.
# Usage: ENV_ROOT=<environment root> MAX_JOBS=<build jobs> [CUDA_HOME=<CUDA toolkit>] \
#   scripts/vad_rollout/env_install.sh <output_root>
# Installs Miniforge under $ENV_ROOT/miniforge and two environments:
#   $ENV_ROOT/vad    the README "Training environment" steps in order (python 3.12, requirements.txt with
#                    --no-deps, pip install -e . --no-deps, flash-attn and causal-conv1d==1.6.1 from source).
#                    The source builds use the docker/Dockerfile variables of the vision-opd workspace at
#                    ed471085 with TORCH_CUDA_ARCH_LIST=12.0 (sm_120, RTX PRO 6000).
#   $ENV_ROOT/judge  python 3.11 with vllm==0.28.0 and transformers==5.16.1 (the toolchain of
#                    TASK-VLM-VAD-REPRO-001-016; vllm 0.28.0 fails on PixtralRotaryEmbedding with
#                    transformers 5.21.0, TASK-VLM-VAD-ROLLOUT-001-006, DIRECTIVE-VLM-VAD-ROLLOUT-001-002).
# ENV_INSTALL_ONLY=judge runs only the judge steps, leaves the VAD environment unchanged, and writes
# pip-freeze-judge-before.txt and the diff of the judge environment.
# Records under <output_root>: ENV_INSTALL.txt, env_install.log, pip-freeze-vad.txt, pip-freeze-judge.txt.
# ENV_INSTALL.txt holds the step exit codes, the nvcc and driver versions, and the SASS architectures of
# the flash-attn and causal-conv1d binaries (cuobjdump --list-elf). A step that exists is skipped.
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
jobs="${MAX_JOBS:-8}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
miniforge_version=24.11.3-0
conda="$env_root/miniforge/bin/conda"
vad="$env_root/vad"
judge="$env_root/judge"
mkdir -p "$out" "$env_root"
cd "$root" || exit 1
log="$out/ENV_INSTALL.txt"
: > "$log"
export PYTHONDONTWRITEBYTECODE=1

# step <name> <command...>: run, record exit code and wall time, stop at the first failure.
step() {
  local name="$1" start code
  shift
  start=$(date +%s)
  echo "== step $name: $*" >> "$out/env_install.log"
  "$@" >> "$out/env_install.log" 2>&1
  code=$?
  echo "step $name exit $code wall_seconds $(( $(date +%s) - start ))" >> "$log"
  if [[ $code -ne 0 ]]; then
    { echo "== env_install.log tail"; tail -n 60 "$out/env_install.log"; } >> "$log"
    exit 1
  fi
}

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== ENV_ROOT $env_root MAX_JOBS $jobs CUDA_HOME $cuda_home"
  echo "== nvcc"; "$cuda_home/bin/nvcc" --version
  echo "== driver"; nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv
} >> "$log" 2>&1

only_judge=0
[[ "${ENV_INSTALL_ONLY:-}" == judge ]] && only_judge=1
echo "== ENV_INSTALL_ONLY ${ENV_INSTALL_ONLY:-<unset>}" >> "$log"
if [[ $only_judge -eq 0 ]]; then
if [[ ! -x "$conda" ]]; then
  url="https://github.com/conda-forge/miniforge/releases/download/${miniforge_version}/Miniforge3-${miniforge_version}-Linux-x86_64.sh"
  if command -v wget > /dev/null; then fetch=(wget -q -O "$env_root/miniforge.sh" "$url"); else fetch=(curl -fsSL -o "$env_root/miniforge.sh" "$url"); fi
  step miniforge-download "${fetch[@]}"
  step miniforge-install bash "$env_root/miniforge.sh" -b -p "$env_root/miniforge"
fi

# README: conda create -n vad python=3.12 -y (a prefix environment under ENV_ROOT)
[[ -x "$vad/bin/python" ]] || step vad-create "$conda" create -p "$vad" python=3.12 -y
step vad-pip-upgrade "$vad/bin/pip" install --upgrade pip
step vad-requirements "$vad/bin/pip" install --no-deps -r requirements.txt
step vad-editable "$vad/bin/pip" install -e . --no-deps
step vad-flash-attn env CUDA_HOME="$cuda_home" PATH="$vad/bin:$cuda_home/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0 \
  FLASH_ATTN_CUDA_ARCHS=120 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="$jobs" \
  "$vad/bin/pip" install flash-attn --no-build-isolation
step vad-causal-conv1d env CUDA_HOME="$cuda_home" PATH="$vad/bin:$cuda_home/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0 \
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="$jobs" \
  "$vad/bin/pip" install causal-conv1d==1.6.1 --no-build-isolation
fi

[[ -x "$judge/bin/python" ]] && "$judge/bin/pip" freeze > "$out/pip-freeze-judge-before.txt" 2>> "$log"
[[ -x "$judge/bin/python" ]] || step judge-create "$conda" create -p "$judge" python=3.11 -y
step judge-pip-upgrade "$judge/bin/pip" install --upgrade pip
step judge-vllm "$judge/bin/pip" install vllm==0.28.0 transformers==5.16.1

"$vad/bin/pip" freeze > "$out/pip-freeze-vad.txt" 2>> "$log"
"$judge/bin/pip" freeze > "$out/pip-freeze-judge.txt" 2>> "$log"
if [[ -f "$out/pip-freeze-judge-before.txt" ]]; then
  { echo "== judge environment diff (before -> after)"; diff "$out/pip-freeze-judge-before.txt" "$out/pip-freeze-judge.txt"; } >> "$log"
fi
{
  echo "== pip freeze lines: vad $(wc -l < "$out/pip-freeze-vad.txt") judge $(wc -l < "$out/pip-freeze-judge.txt")"
  grep -i -E '^(torch|vllm|flash[-_]attn|causal[-_]conv1d|transformers|verl|vad[-_]opd)[= @]' "$out/pip-freeze-vad.txt"
  grep -i -E '^(torch|vllm|transformers)[= @]' "$out/pip-freeze-judge.txt"
  echo "== SASS architectures (cuobjdump --list-elf)"
  find "$vad/lib" \( -name 'flash_attn_2_cuda*.so' -o -name 'causal_conv1d_cuda*.so' \) -print | sort | while read -r so; do
    echo "$so $("$cuda_home/bin/cuobjdump" --list-elf "$so" | grep -o 'sm_[0-9]*' | sort | uniq -c | tr '\n' ' ')"
  done
  echo "== du -sh"; du -sh "$vad" "$judge"
  echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$log" 2>&1
