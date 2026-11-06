#!/bin/bash
# Part B item 1 of HANDOFF-VLM-VAD-PACKFIX-001-R1 (gate G2, and the env check and input hashes).
# Runs on branch exp/VLM-VAD-PACKFIX-001-g2: the merge of exp/VLM-VAD-PACK-001 (scripts/vad_pack/, scripts/vad_grad/)
# and exp/VLM-VAD-PACKFIX-001 (the fix). Steps:
#   1. scripts/vad_grad/env_check_b200.sh (ENV_CHECK.txt).
#   2. scripts/vad_packfix/inputs.sh (INPUTS.sha256).
#   3. scripts/vad_pack/run_pack.sh with PILOT=1 and VAD_PACKFIX=1 into <out>/pack: the 6 pilot micro-batches of
#      TASK-VLM-VAD-PACK-001-002 (pilot.py select gives the same selection), arm P with the fix and arm U.
#   4. scripts/vad_packfix/g2_compare.py: |L_P - L_U| / max(1, |L_U|) of each row, against 1e-3, and the reference
#      rows of TASK-VLM-VAD-PACK-001-002 (REF_ROOT).
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> REF_ROOT=<TASK-PACK-002 output> \
#   scripts/vad_packfix/g2_check.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="${ENV_ROOT:?}" ref_root="${REF_ROOT:?}" b200="${B200_ROOT:?}"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/G2_RUN.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD)" >> "$log"

MODEL_DIR="$b200/harness/data/models/Qwen3.5-4B" bash scripts/vad_grad/env_check_b200.sh "$out" >> "$log" 2>&1
code=$?
echo "env_check_b200.sh exit $code" >> "$log"
[[ $code -eq 0 ]] || exit 1

bash scripts/vad_packfix/inputs.sh "$out/INPUTS.sha256" >> "$log" 2>&1
code=$?
echo "inputs.sh exit $code" >> "$log"
[[ $code -eq 0 ]] || exit 1

VAD_PACKFIX=1 PILOT=1 bash scripts/vad_pack/run_pack.sh "$out/pack" >> "$log" 2>&1
code=$?
echo "run_pack.sh (PILOT=1, VAD_PACKFIX=1) exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
[[ $code -eq 0 ]] || exit 1
echo "fix lines in the pack logs: $(cat "$out"/pack/pack_*.log | grep -c 'Packed linear-attention fix applied')" >> "$log"

"$py" scripts/vad_packfix/g2_compare.py "$out/pack" "$ref_root" "$out/G2_CHECK.json" >> "$log" 2>&1
code=$?
echo "g2_compare.py exit $code" >> "$log"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit "$code"
