#!/bin/bash
# VLM-VAD-ROLLOUT-001 environment check (HANDOFF-VLM-VAD-ROLLOUT-001-R1, Part A item 5). Uses GPU 0 only.
# Usage: ENV_ROOT=<environment root> scripts/vad_rollout/env_check.sh <output_root>
# In the VAD environment: imports torch, vllm, flash_attn, causal_conv1d and verl, prints the versions and
# the device capability of every GPU, and runs one flash-attn forward and backward on GPU 0. In the judge
# environment: imports vllm. The record is <output_root>/ENV_CHECK.txt.
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/ENV_CHECK.txt"
export PYTHONDONTWRITEBYTECODE=1
status=0

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== nvidia-smi"; nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,memory.used,memory.total --format=csv
  echo "== tools"; for tool in curl git nvcc; do echo "$tool $(command -v "$tool" || echo missing)"; done
  echo "== VAD environment ($env_root/vad)"
} > "$log" 2>&1

"$env_root/vad/bin/python" - >> "$log" 2>&1 <<'EOF'
import importlib

import torch

for name in ("torch", "vllm", "flash_attn", "causal_conv1d", "verl", "transformers"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "?"), getattr(module, "__file__", "?"))
print("cuda", torch.version.cuda, "device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("device", i, torch.cuda.get_device_name(i), "capability", torch.cuda.get_device_capability(i))
print("torch arch list", torch.cuda.get_arch_list())

from flash_attn import flash_attn_func

q, k, v = (torch.randn(2, 128, 8, 64, device="cuda:0", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
o = flash_attn_func(q, k, v, causal=True)
o.float().sum().backward()
torch.cuda.synchronize()
print("flash_attn forward", tuple(o.shape), "finite", bool(torch.isfinite(o).all()),
      "backward grad finite", all(bool(torch.isfinite(t.grad).all()) for t in (q, k, v)))
EOF
code=$?
echo "vad check exit $code" >> "$log"
[[ $code -eq 0 ]] || status=1

echo "== judge environment ($env_root/judge)" >> "$log"
"$env_root/judge/bin/python" - >> "$log" 2>&1 <<'EOF'
import torch
import vllm

print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
EOF
code=$?
echo "judge check exit $code" >> "$log"
[[ $code -eq 0 ]] || status=1
exit "$status"
