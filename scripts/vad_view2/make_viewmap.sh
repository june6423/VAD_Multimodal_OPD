#!/bin/bash
# VIEWMAP.json of VLM-VAD-VIEW-002 (HANDOFF-VLM-VAD-VIEW-002-R1 Part A item 1) into <out dir>.
# viewmap.py checks the VIEW-001 views of TASK-VLM-VAD-VIEW-001-006 (manifest sha256 EXPECTED and a pixel check of 30
# rows). When the check fails (exit 3), views.py (the VIEW-001 copy, unchanged) makes P1_FIT.json and the views again
# in <out dir>/views, and viewmap.py maps them. An existing VIEWMAP.json is kept.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> scripts/vad_view2/make_viewmap.sh <out dir>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${B200_ROOT:?}/harness/data"
py="${ENV_ROOT:?}/vad/bin/python"
view1="$B200_ROOT/harness/outputs/VLM-VAD-VIEW-001/TASK-VLM-VAD-VIEW-001-006/views"
expected=64908c06acfad9cf3a96621d9614947b7393d4111dc329e84307c229084c7856
parquet="$data/vad/train_vad.parquet"
mkdir -p "$out"
[[ -f "$out/VIEWMAP.json" ]] && { echo "VIEWMAP.json exists"; exit 0; }
"$py" "$root/scripts/vad_view2/viewmap.py" "$parquet" "$view1" "$expected" "$out"
code=$?
echo "viewmap.py (VIEW-001 views) exit $code"
[[ $code -eq 3 ]] || exit "$code"
echo "the VIEW-001 views do not match: views.py makes them again in $out/views"
v="$out/views"
mkdir -p "$v"
[[ -f "$v/P1_FIT.json" ]] || "$py" "$root/scripts/vad_view2/views.py" fit "$parquet" \
  "$data/hub/Vision-OPD-6K/train.jsonl" "$v/P1_FIT.json" || exit 1
[[ -f "$v/REMOVED.json" ]] || OMP_NUM_THREADS=1 "$py" "$root/scripts/vad_view2/views.py" build "$parquet" \
  "$data/hub/Vision-OPD-6K/train.jsonl" "$v/P1_FIT.json" "$data/models/Qwen3.5-4B" "$v" || exit 1
"$py" "$root/scripts/vad_view2/viewmap.py" "$parquet" "$v" - "$out"
