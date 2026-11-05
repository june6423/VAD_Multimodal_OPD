#!/bin/bash
# Part B item 11 (HANDOFF-VLM-VAD-GRAD-001-R3/R4, gate G11): stats.py of this checkout on the three GRAM.json files of
# an earlier FULL output. CPU only.
# Usage: ENV_ROOT=<environment root> GRAM_ROOT=<FULL task output root> scripts/vad_grad/stats_r3.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="${ENV_ROOT:?}/vad/bin/python"
gram_root="${GRAM_ROOT:?}"
mkdir -p "$out"
cd "$root" || exit 1
log="$out/RUN_STATS.txt"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD) GRAM_ROOT $gram_root"
  sha256sum scripts/vad_grad/stats.py "$gram_root"/step0/GRAM.json "$gram_root"/step80/GRAM.json "$gram_root"/step130/GRAM.json
} >> "$log" 2>&1
for ck in step0 step80 step130; do
  "$py" scripts/vad_grad/stats.py "$gram_root/$ck/GRAM.json" "$out/STATS_$ck.json" >> "$log" 2>&1
  code=$?
  echo "$ck stats exit $code" >> "$log"
  [[ $code -eq 0 ]] || exit 1
done
sha256sum "$out"/STATS_step0.json "$out"/STATS_step80.json "$out"/STATS_step130.json >> "$log" 2>&1
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
