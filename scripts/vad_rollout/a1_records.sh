#!/bin/bash
# VLM-VAD-ROLLOUT-001 post-run records of a training arm (HANDOFF-VLM-VAD-ROLLOUT-001-R4 Part B item 9, gate G7;
# DT routing of 2026-10-29 for the fetch limit). Usage: ENV_ROOT=<environment root> \
#   scripts/vad_rollout/a1_records.sh <arm output_root> <output_root>
# The rollouts and tokstats folders stay on the server. This script writes into <output_root>:
#   LISTING.txt      byte size and SHA-256 of every file under <arm output_root>/rollouts and /tokstats
#   CONSOLE_TAIL.txt the last 400 lines of <arm output_root>/console.log
#   ARCHIVE.txt      size and SHA-256 of <arm output_root>/rollouts_tokstats.tar (both folders)
#   STEP_METRICS_COMPACT.json  per step the keys timing_s/*, response_length/*, prompt_length/*,
#                    self_distillation/*, actor/grad_norm and val-core/*, val-aux/* of STEP_METRICS.json
#   tables.json, tables.md  analyze_rollouts.py over the server paths, run with the VAD environment python
# No GPU.
set -uo pipefail

arm="$1"
out="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${ENV_ROOT:?ENV_ROOT must name the environment root}/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1

(cd "$arm" && find rollouts tokstats -type f | sort | while read -r f; do
  printf '%s %s %s\n' "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)" "$f"
done) > "$out/LISTING.txt"
echo "files $(wc -l < "$out/LISTING.txt")" >> "$out/LISTING.txt"
tail -n 400 "$arm/console.log" > "$out/CONSOLE_TAIL.txt"
"$python" - "$arm/STEP_METRICS.json" "$out/STEP_METRICS_COMPACT.json" <<'EOF'
import json
import sys

prefixes = ("timing_s/", "response_length/", "prompt_length/", "self_distillation/", "val-core/", "val-aux/")
steps = json.load(open(sys.argv[1]))
compact = {step: {k: v for k, v in metrics.items() if k.startswith(prefixes) or k == "actor/grad_norm"}
           for step, metrics in steps.items()}
with open(sys.argv[2], "w") as f:
    json.dump(compact, f, sort_keys=True, separators=(",", ":"))
    f.write("\n")
EOF

tar -C "$arm" -cf "$arm/rollouts_tokstats.tar" rollouts tokstats
code=$?
{
  echo "tar exit $code"
  echo "path $arm/rollouts_tokstats.tar"
  echo "bytes $(stat -c %s "$arm/rollouts_tokstats.tar")"
  echo "sha256 $(sha256sum "$arm/rollouts_tokstats.tar" | cut -d' ' -f1)"
} > "$out/ARCHIVE.txt"
[[ $code -eq 0 ]] || exit "$code"

PYTHONDONTWRITEBYTECODE=1 "$python" scripts/vad_rollout/analyze_rollouts.py "$arm/rollouts" "$arm/tokstats" "$out" \
  > "$out/analyze.log" 2>&1
code=$?
echo "analyze_rollouts exit $code" >> "$out/analyze.log"
exit "$code"
