#!/bin/bash
# G2 (HANDOFF-VLM-VAD-GRAD-001-R1, Part B item 1; DIRECTIVE-VLM-VAD-GRAD-001-002 D5): sha256 check of the files that
# ET copied from the class-A backup to $DATA_ROOT/rollout001/. No GPU.
# $DATA_ROOT/rollout001/ITEM1_MANIFEST.sha256 holds the lines of vast_backup/VLM-VAD-ROLLOUT-001/TRANSFER/MANIFEST_A.all
# for these files (paths start with harness/outputs/VLM-VAD-ROLLOUT-001/). The script checks the SHA-256 of that list
# against EXPECTED_MANIFEST_SHA256, maps each path to rollout001/, and runs sha256sum -c with 8 processes.
# Usage: DATA_ROOT=<data root> EXPECTED_MANIFEST_SHA256=<hex> scripts/vad_grad/item1_check.sh <output_root>
# Records: <output_root>/ITEM1_CHECK.txt (counts, every non-OK line) and ITEM1_CHECK.log (full sha256sum -c output).
set -uo pipefail
out="$1"
data="${DATA_ROOT:?DATA_ROOT must name the data root}"
expected="${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 must give the manifest hash}"
dir="$data/rollout001"
mkdir -p "$out/check"
log="$out/ITEM1_CHECK.txt"
manifest="$dir/ITEM1_MANIFEST.sha256"
got=$(sha256sum "$manifest" | cut -d' ' -f1)
lines=$(wc -l < "$manifest")
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== manifest $manifest lines $lines sha256 $got expected $expected"
} > "$log"
if [[ "$got" != "$expected" ]]; then echo "STOP: manifest sha256 differs" >> "$log"; exit 3; fi
sed 's#  harness/outputs/VLM-VAD-ROLLOUT-001/#  #' "$manifest" > "$out/check/manifest.local"
split -e -n r/8 -d -a 1 "$out/check/manifest.local" "$out/check/part"
start=$(date +%s)
(cd "$dir" && find "$out/check" -name 'part?' | sort | xargs -d '\n' -n 1 -P 8 sh -c 'sha256sum -c "$1" > "$1.out" 2>&1' _)
cat "$out/check"/part?.out > "$out/ITEM1_CHECK.log"
ok=$(grep -c ': OK$' "$out/ITEM1_CHECK.log")
bad=$(grep -c -v ': OK$' "$out/ITEM1_CHECK.log")
bytes=$(cut -c67- "$out/check/manifest.local" | (cd "$dir" && xargs -d '\n' stat -c %s 2>/dev/null) | awk '{s += $1} END {printf "%.0f", s}')
{
  echo "== check: OK $ok, not OK $bad of $lines lines, $(( $(date +%s) - start )) s, bytes of the listed files $bytes"
  grep -v ': OK$' "$out/ITEM1_CHECK.log"
  echo "== df"; df -B1 "$dir"
} >> "$log" 2>&1
[[ $ok -eq $lines && $bad -eq 0 ]] && { echo "== RESULT OK" >> "$log"; exit 0; }
echo "== RESULT FAILED" >> "$log"
exit 1
