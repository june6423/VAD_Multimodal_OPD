#!/bin/bash
# sha256 list of the inputs of the PACKFIX-001 training (HANDOFF-VLM-VAD-PACKFIX-001-R1 Part B item 1):
# the student and teacher model Qwen3.5-4B, the judge model Mistral-Small-3.2-24B-Instruct-2506, the training and
# fast-validation data, and the chat template of this checkout.
# Usage: B200_ROOT=<root> scripts/vad_packfix/inputs.sh <out file>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${B200_ROOT:?}/harness/data"
status=0
{
  find "$data/models/Qwen3.5-4B" "$data/models/Mistral-Small-3.2-24B-Instruct-2506" -type f | sort
  ls "$data/vad/train_vad.parquet" "$data/vad/benchmarks/opd_fast_benchmarks.parquet" "$data/vad/vad_data_manifest.json"
  ls "$root/chat_templates/perception_chat_template_qwen35.jinja"
} > "$out.list" || status=1
xargs -d '\n' sha256sum < "$out.list" > "$out" || status=1
echo "inputs hashed: $(wc -l < "$out") of $(wc -l < "$out.list") files"
rm -f "$out.list"
exit "$status"
