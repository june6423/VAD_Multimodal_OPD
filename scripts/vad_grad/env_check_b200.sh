#!/bin/bash
# VLM-VAD-GRAD-001 environment check on b200 (HANDOFF-VLM-VAD-GRAD-001-R1, Part A item 1, gate G1;
# DIRECTIVE-VLM-VAD-GRAD-001-001 Decisions 3 and 5). Uses the one GPU.
# A copy of scripts/vad_rollout/env_check.sh with these additions:
#   1. OS release, glibc version, the nvcc version of $ENV_ROOT/cuda-12.8, and the system nvcc path.
#   2. Import times of torch and vllm in the VAD environment and of vllm in the judge environment.
#   3. torch.cuda.get_arch_list(), and the SASS architectures of the flash-attn and causal-conv1d binaries.
#   4. scripts/vad_grad/layer_check.py: one Qwen3.5-4B GatedDeltaNet layer and one full-attention layer, forward and
#      backward on the GPU with flash_attention_2 and the GatedDeltaNet fast path.
# The check fails (exit 1) when the device count is not 1, the capability is not (10, 0), a binary has no sm_100,
# a result is not finite, or an import fails.
# Usage: ENV_ROOT=<environment root> MODEL_DIR=<directory with the Qwen3.5-4B config.json> \
#   scripts/vad_grad/env_check_b200.sh <output_root>
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
model_dir="${MODEL_DIR:?MODEL_DIR must name a directory with the Qwen3.5-4B config.json}"
cuda="$env_root/cuda-12.8"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/ENV_CHECK.txt"
export PYTHONDONTWRITEBYTECODE=1
status=0

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== os"; head -n 4 /etc/os-release
  echo "== glibc"; ldd --version | head -n 1
  echo "== system nvcc path"; ls -l /usr/local/cuda; ls /usr/local/cuda/bin/nvcc
  echo "== environment nvcc ($cuda)"; "$cuda/bin/nvcc" --version | tail -n 2
  echo "== nvidia-smi"; nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,memory.used,memory.total --format=csv
  echo "== VAD environment ($env_root/vad)"
} > "$log" 2>&1

"$env_root/vad/bin/python" - >> "$log" 2>&1 <<'EOF'
import importlib
import sys
import time

start = time.time()
import torch  # noqa: E402

print("import torch seconds", round(time.time() - start, 1))
start = time.time()
import vllm  # noqa: E402,F401

print("import vllm seconds", round(time.time() - start, 1))
for name in ("torch", "vllm", "flash_attn", "causal_conv1d", "fla", "verl", "transformers"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "?"), getattr(module, "__file__", "?"))
count = torch.cuda.device_count()
caps = [torch.cuda.get_device_capability(i) for i in range(count)]
print("cuda", torch.version.cuda, "device_count", count)
for i in range(count):
    print("device", i, torch.cuda.get_device_name(i), "capability", caps[i])
print("torch arch list", torch.cuda.get_arch_list())

from flash_attn import flash_attn_func  # noqa: E402

q, k, v = (torch.randn(2, 128, 8, 64, device="cuda:0", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
o = flash_attn_func(q, k, v, causal=True)
o.float().sum().backward()
torch.cuda.synchronize()
fwd = bool(torch.isfinite(o).all())
bwd = all(bool(torch.isfinite(t.grad).all()) for t in (q, k, v))
print("flash_attn forward", tuple(o.shape), "finite", fwd, "backward grad finite", bwd)
ok = count == 1 and caps[0] == (10, 0) and fwd and bwd
print("device and flash_attn check", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
code=$?
echo "vad check exit $code" >> "$log"
[[ $code -eq 0 ]] || status=1

echo "== SASS architectures (cuobjdump --list-elf)" >> "$log"
sos=$(find "$env_root/vad/lib" \( -name 'flash_attn_2_cuda*.so' -o -name 'causal_conv1d_cuda*.so' \) -print | sort)
[[ $(echo "$sos" | grep -c .) -eq 2 ]] || { echo "expected 2 binaries, found: $sos" >> "$log"; status=1; }
for so in $sos; do
  archs=$("$cuda/bin/cuobjdump" --list-elf "$so" | grep -o 'sm_[0-9]*' | sort | uniq -c | tr '\n' ' ')
  echo "$so $archs" >> "$log"
  echo "$archs" | grep -q 'sm_100' || { echo "no sm_100 in $so" >> "$log"; status=1; }
done

echo "== layer check (scripts/vad_grad/layer_check.py, model config $model_dir)" >> "$log"
"$env_root/vad/bin/python" scripts/vad_grad/layer_check.py "$model_dir" cuda flash_attention_2 512 >> "$log" 2>&1
code=$?
echo "layer check exit $code" >> "$log"
[[ $code -eq 0 ]] || status=1

echo "== judge environment ($env_root/judge)" >> "$log"
"$env_root/judge/bin/python" - >> "$log" 2>&1 <<'EOF'
import time

start = time.time()
import torch  # noqa: E402
import vllm  # noqa: E402

print("import torch and vllm seconds", round(time.time() - start, 1))
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
EOF
code=$?
echo "judge check exit $code" >> "$log"
[[ $code -eq 0 ]] || status=1
echo "== status $status" >> "$log"
exit "$status"
