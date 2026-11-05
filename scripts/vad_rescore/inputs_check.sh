#!/bin/bash
# Part B item 1 (HANDOFF-VLM-VAD-GRAD-002-R1, gate G1): scripts/vad_grad/env_check_b200.sh (uses the GPU), then the
# sha256 of every input of the rescoring, and a check of the TASK-VLM-VAD-ROLLOUT-001-053 rollout files against
# scripts/vad_rescore/MANIFEST_053.sha256 (their lines of vast_backup/VLM-VAD-ROLLOUT-001/TRANSFER/MANIFEST_A.all).
# Inputs: the TASK-053 rollouts of step0 and step80, the TASK-GRAD-001-005 labels, TRAJ.jsonl and ANSWER_POS.jsonl of
# TASK-GRAD-001-013, the A1 COMMAND.txt, the training parquet, and every file of the 5 model directories.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> \
#   TRAJ_ROOT=<TASK-GRAD-001-013 output> scripts/vad_rescore/inputs_check.sh <output_root>
# Records: ENV_CHECK.txt (env_check_b200.sh), INPUTS.sha256 (sha256sum lines), INPUTS_CHECK.txt (counts, results).
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${B200_ROOT:?}/harness/data"
labels_root="${LABELS_ROOT:?}" traj_root="${TRAJ_ROOT:?}"
r1="$data/rollout001"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/INPUTS_CHECK.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD)" > "$log"
MODEL_DIR="$data/models/Qwen3.5-4B" scripts/vad_grad/env_check_b200.sh "$out"
env_code=$?
echo "env_check_b200.sh exit $env_code" >> "$log"

files=("$r1/TASK-VLM-VAD-ROLLOUT-001-053/step0/rollouts.jsonl" "$r1/TASK-VLM-VAD-ROLLOUT-001-053/step80/rollouts.jsonl"
       "$labels_root/step0/LABELS.jsonl" "$labels_root/step80/LABELS.jsonl"
       "$traj_root/step0/TRAJ.jsonl" "$traj_root/step80/TRAJ.jsonl"
       "$traj_root/step0/ANSWER_POS.jsonl" "$traj_root/step80/ANSWER_POS.jsonl"
       "$r1/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$data/vad/train_vad.parquet")
dirs=("$data/models/Qwen3.5-4B" "$r1/TASK-VLM-VAD-ROLLOUT-001-022/merged/step_80" "$r1/TASK-VLM-VAD-ROLLOUT-001-022/merged/step_130"
      "$r1/TASK-VLM-VAD-ROLLOUT-001-036/merged/step_80" "$r1/TASK-VLM-VAD-ROLLOUT-001-036/merged/step_130")
missing=0
for f in "${files[@]}"; do [[ -f "$f" ]] || { echo "MISSING $f" >> "$log"; missing=$((missing + 1)); }; done
for d in "${dirs[@]}"; do [[ -d "$d" ]] || { echo "MISSING $d" >> "$log"; missing=$((missing + 1)); }; done
{ printf '%s\n' "${files[@]}"; find "${dirs[@]}" -type f | sort; } | xargs -d '\n' -P 8 -n 4 sha256sum 2>> "$log" \
  | sort -k2 > "$out/INPUTS.sha256"
echo "== inputs hashed: $(wc -l < "$out/INPUTS.sha256") files, missing $missing" >> "$log"
for d in "${dirs[@]}"; do echo "model dir $d files $(find "$d" -type f | wc -l)" >> "$log"; done

sed "s#  harness/outputs/VLM-VAD-ROLLOUT-001/#  $r1/#" scripts/vad_rescore/MANIFEST_053.sha256 > "$out/manifest_053.local"
sha256sum -c "$out/manifest_053.local" >> "$log" 2>&1
m_code=$?
echo "== TASK-053 manifest check exit $m_code ($(wc -l < scripts/vad_rescore/MANIFEST_053.sha256) lines)" >> "$log"
rm -f "$out/manifest_053.local"
if [[ $env_code -eq 0 && $m_code -eq 0 && $missing -eq 0 ]]; then echo "== RESULT OK" >> "$log"; exit 0; fi
echo "== RESULT FAILED" >> "$log"
exit 1
