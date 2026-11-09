#!/bin/bash
# Part B item 1 of HANDOFF-VLM-VAD-GRAD-003-R2: env check, input hashes, folds and row files, disk timing, gate G2.
# 1. CHECK_G2.txt: git HEAD, the changed paths from 2627743a (G1 scope), the fix-file blob hashes, GPU, memory, disk,
#    package versions and paths (verl from this checkout), VAD_PACKFIX unset.
# 2. inputs.py: INPUTS.sha256, INPUTS_CHECK.json (manifest lines of the used rollout001 files, other inputs hashed).
# 3. batches.py for M0, A1_80 and A1_130: <ck>/rows/ROWS_*.jsonl and BATCHES.json (fold sizes, dummies, micro-batches).
# 4. g2.py at M0 (fold 1, n=8): G2.json. 5. io_check.py: IO_CHECK.json (4 vectors of the real size).
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> \
#   ADAM_ROOT=<TASK-GRAD-001-009 output> scripts/vad_grad3/check_g2.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}" adam_root="${ADAM_ROOT:?}"
data="$b200/harness/data"
r1="$data/rollout001"
r053="$r1/TASK-VLM-VAD-ROLLOUT-001-053"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR VAD_PACKFIX
log="$out/CHECK_G2.txt"
status=0
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== paths changed from 2627743a"; git -c safe.directory="$root" diff --name-status 2627743a HEAD
  echo "== fix-file blobs (expected 4e08c2bd..., 8522e81b...)"
  git -c safe.directory="$root" hash-object verl/models/transformers/qwen3_5_packfix.py verl/models/transformers/monkey_patch.py
  echo "== VAD_PACKFIX=${VAD_PACKFIX:-unset}"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total --format=csv
  grep -E "MemTotal|MemAvailable" /proc/meminfo
  echo "cgroup memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
  df -B1 "$out"
} > "$log" 2>&1
"$py" - >> "$log" 2>&1 <<'EOF'
import importlib
import torch
for name in ("torch", "flash_attn", "fla", "verl", "transformers", "safetensors"):
    m = importlib.import_module(name)
    print(name, getattr(m, "__version__", "?"), getattr(m, "__file__", "?"))
print("cuda", torch.version.cuda, "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))
EOF
"$py" scripts/vad_grad3/map_command.py "$r1/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1

"$py" scripts/vad_grad3/inputs.py "$b200" "$labels_root" "$adam_root" "$out" >> "$log" 2>&1
code=$?; echo "inputs.py exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1

declare -A step=([M0]=step0 [A1_80]=step80 [A1_130]=step130)
for ck in M0 A1_80 A1_130; do
  mkdir -p "$out/$ck/rows"
  "$py" scripts/vad_grad3/batches.py "$r053/${step[$ck]}/rollouts.jsonl" "$labels_root/${step[$ck]}/LABELS.jsonl" \
    "$out/$ck/rows" >> "$log" 2>&1
  code=$?; echo "batches.py $ck exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
  cp "$out/$ck/rows/BATCHES.json" "$out/BATCHES_$ck.json" 2>/dev/null
done
[[ $status -eq 0 ]] || { echo "== RESULT FAILED before G2" >> "$log"; exit 1; }

"$py" scripts/vad_grad3/g2.py "$root" "$out/COMMAND_B200.txt" "$data/models/Qwen3.5-4B" "$r053/step0/rollouts.jsonl" \
  "$out/M0/rows" "$r1/TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS.txt" "$out/g2" >> "$out/g2.log" 2>&1
code=$?; echo "g2.py exit $code (3 = gate failed)" >> "$log"; [[ $code -eq 0 ]] || status=1
cp "$out/g2/G2.json" "$out/G2.json" 2>/dev/null

"$py" scripts/vad_grad3/io_check.py "$out" 3569090560 4 >> "$log" 2>&1
code=$?; echo "io_check.py exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
echo "cgroup memory.peak $(cat /sys/fs/cgroup/memory.peak 2>/dev/null) events $(tr '\n' ' ' < /sys/fs/cgroup/memory.events 2>/dev/null)" >> "$log"
echo "== RESULT $([[ $status -eq 0 ]] && echo OK || echo FAILED) $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit "$status"
