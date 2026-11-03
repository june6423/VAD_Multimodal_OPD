#!/bin/bash
# VLM-VAD-GRAD-001 environments on the b200 server (HANDOFF-VLM-VAD-GRAD-001-R1, Part A item 1;
# DIRECTIVE-VLM-VAD-GRAD-001-001 Decisions 2 to 4). No GPU work.
# A copy of scripts/vad_rollout/env_install.sh with these changes:
#   1. CUDA toolkit 12.8 from the nvidia conda channel (label cuda-12.8.1) in $ENV_ROOT/cuda-12.8. The container
#      (NGC PyTorch, CUDA 13.1) has no 12.8 toolkit. The system Python and torch are not used.
#   2. flash-attn and causal-conv1d==1.6.1 are built with TORCH_CUDA_ARCH_LIST=10.0 and FLASH_ATTN_CUDA_ARCHS=100
#      (sm_100, B200).
#   3. TMPDIR of the source builds is a new directory under /tmp (tmpfs). The conda package cache and the pip
#      cache are under $ENV_ROOT (conda-pkgs, pip-cache).
#   4. ENV_INSTALL.txt also records the OS release, the glibc and gcc versions, the system nvcc path and version,
#      and the total wall time.
# The judge environment is unchanged: python 3.11, vllm==0.28.0, transformers==5.16.1.
# Usage: ENV_ROOT=<environment root> MAX_JOBS=<build jobs> scripts/vad_grad/env_install_b200.sh <output_root>
# Records under <output_root>: ENV_INSTALL.txt, env_install.log, pip-freeze-vad.txt, pip-freeze-judge.txt.
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
jobs="${MAX_JOBS:-16}"
miniforge_version=24.11.3-0
conda="$env_root/miniforge/bin/conda"
cuda="$env_root/cuda-12.8"
vad="$env_root/vad"
judge="$env_root/judge"
mkdir -p "$out" "$env_root" "$env_root/conda-pkgs" "$env_root/pip-cache"
cd "$root" || exit 1
log="$out/ENV_INSTALL.txt"
: > "$log"
export PYTHONDONTWRITEBYTECODE=1 CONDA_PKGS_DIRS="$env_root/conda-pkgs" PIP_CACHE_DIR="$env_root/pip-cache"
build_tmp="/tmp/vad_grad_build_$$"
mkdir -p "$build_tmp"
t0=$(date +%s)

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
    rm -rf "$build_tmp"
    exit 1
  fi
}

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== ENV_ROOT $env_root MAX_JOBS $jobs build TMPDIR $build_tmp"
  echo "== os"; head -n 4 /etc/os-release
  echo "== glibc"; ldd --version | head -n 1
  echo "== gcc"; gcc --version | head -n 1
  echo "== system nvcc"; ls -l /usr/local/cuda; /usr/local/cuda/bin/nvcc --version | tail -n 2
  echo "== driver"; nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv
} >> "$log" 2>&1

if [[ ! -x "$conda" ]]; then
  url="https://github.com/conda-forge/miniforge/releases/download/${miniforge_version}/Miniforge3-${miniforge_version}-Linux-x86_64.sh"
  step miniforge-download curl -fsSL -o "$env_root/miniforge.sh" "$url"
  step miniforge-install bash "$env_root/miniforge.sh" -b -p "$env_root/miniforge"
fi

[[ -x "$cuda/bin/nvcc" ]] || step cuda-toolkit "$conda" create -p "$cuda" -c nvidia/label/cuda-12.8.1 cuda-toolkit=12.8.1 -y
# Conda CUDA packages keep headers and libraries under targets/x86_64-linux. torch builds look in include/ and lib64/.
cuda_inc="$cuda/targets/x86_64-linux/include"
cuda_lib="$cuda/targets/x86_64-linux/lib"
{
  echo "== env nvcc"; "$cuda/bin/nvcc" --version | tail -n 2
  echo "== cuda headers: include/cuda_runtime.h $([[ -e $cuda/include/cuda_runtime.h ]] && echo yes || echo no)," \
    "targets include/cuda_runtime.h $([[ -e $cuda_inc/cuda_runtime.h ]] && echo yes || echo no)"
} >> "$log" 2>&1
build_env=(env CUDA_HOME="$cuda" PATH="$vad/bin:$cuda/bin:$PATH" CPATH="$cuda_inc:${CPATH:-}"
  LIBRARY_PATH="$cuda_lib:$cuda/lib:${LIBRARY_PATH:-}" TMPDIR="$build_tmp" TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS="$jobs")

# README: conda create -n vad python=3.12 -y (a prefix environment under ENV_ROOT)
[[ -x "$vad/bin/python" ]] || step vad-create "$conda" create -p "$vad" python=3.12 -y
step vad-pip-upgrade "$vad/bin/pip" install --upgrade pip
step vad-requirements "$vad/bin/pip" install --no-deps -r requirements.txt
step vad-editable "$vad/bin/pip" install -e . --no-deps
step vad-flash-attn "${build_env[@]}" FLASH_ATTN_CUDA_ARCHS=100 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  "$vad/bin/pip" install flash-attn --no-build-isolation
step vad-causal-conv1d "${build_env[@]}" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  "$vad/bin/pip" install causal-conv1d==1.6.1 --no-build-isolation
rm -rf "$build_tmp"

[[ -x "$judge/bin/python" ]] || step judge-create "$conda" create -p "$judge" python=3.11 -y
step judge-pip-upgrade "$judge/bin/pip" install --upgrade pip
step judge-vllm "$judge/bin/pip" install vllm==0.28.0 transformers==5.16.1

"$vad/bin/pip" freeze > "$out/pip-freeze-vad.txt" 2>> "$log"
"$judge/bin/pip" freeze > "$out/pip-freeze-judge.txt" 2>> "$log"
{
  echo "== pip freeze lines: vad $(wc -l < "$out/pip-freeze-vad.txt") judge $(wc -l < "$out/pip-freeze-judge.txt")"
  grep -i -E '^(torch|vllm|flash[-_]attn|causal[-_]conv1d|transformers|verl|vad[-_]opd)[= @]' "$out/pip-freeze-vad.txt"
  grep -i -E '^(torch|vllm|transformers)[= @]' "$out/pip-freeze-judge.txt"
  echo "== SASS architectures (cuobjdump --list-elf)"
  find "$vad/lib" \( -name 'flash_attn_2_cuda*.so' -o -name 'causal_conv1d_cuda*.so' \) -print | sort | while read -r so; do
    echo "$so $("$cuda/bin/cuobjdump" --list-elf "$so" | grep -o 'sm_[0-9]*' | sort | uniq -c | tr '\n' ' ')"
  done
  echo "== du -sh"; du -sh "$cuda" "$vad" "$judge"
  echo "== total wall_seconds $(( $(date +%s) - t0 ))"
  echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$log" 2>&1
