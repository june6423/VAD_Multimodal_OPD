#!/bin/bash
# VLM-VAD-ROLLOUT-001 R8 gate G3 (HANDOFF-VLM-VAD-ROLLOUT-001-R8, Part B item 1). No GPU, no judge call.
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> \
#   scripts/vad_rollout/r8_g3.sh <TASK-025 output root> <TASK-026 output root> <output_root>
# Writes the SHA-256 of every file under both task output roots, runs rescore_run.sh for <TASK-025>/run (tag
# released_step80) into <output_root>/released_step80 and for <TASK-026>/run (tag base_qwen35_4b_step0) into
# <output_root>/base_qwen35_4b_step0, writes the SHA-256 again, and writes <output_root>/G3.txt: the rescore
# exit codes, the hash comparison, and per arm the correct count and accuracy per benchmark and Avg6 from
# summary.json, compared with the values that R8 G3 names.
set -uo pipefail

t025="$1"
t026="$2"
out="$3"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${ENV_ROOT:-$HOME/vad-rollout/envs}/vad/bin:$PATH"
mkdir -p "$out"
tree_sha() { (cd "$1" && find . -type f | sort | xargs -d '\n' sha256sum); }

tree_sha "$t025" > "$out/T025_SHA256_BEFORE.txt"
tree_sha "$t026" > "$out/T026_SHA256_BEFORE.txt"
bash "$here/rescore_run.sh" "$t025/run" released_step80 "$out/released_step80"
c025=$?
bash "$here/rescore_run.sh" "$t026/run" base_qwen35_4b_step0 "$out/base_qwen35_4b_step0"
c026=$?
tree_sha "$t025" > "$out/T025_SHA256_AFTER.txt"
tree_sha "$t026" > "$out/T026_SHA256_AFTER.txt"

{
  echo "rescore released_step80 exit $c025"
  echo "rescore base_qwen35_4b_step0 exit $c026"
  for t in T025 T026; do
    if cmp -s "$out/${t}_SHA256_BEFORE.txt" "$out/${t}_SHA256_AFTER.txt"; then
      echo "$t output files $(wc -l < "$out/${t}_SHA256_AFTER.txt") sha256 unchanged"
    else
      echo "$t output files sha256 CHANGED"
    fi
  done
  python3 - "$out" <<'EOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
names = ["vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn"]
# R8 G3: released step 80 counts of TASK-025 (and Avg6 78.73), Base counts of TASK-026 for the first five.
expected = {
    "released_step80": dict(zip(names, [178, 520, 696, 667, 17856, 4239])),
    "base_qwen35_4b_step0": dict(zip(names[:5], [157, 415, 691, 653, 15150])),
}
for arm, want in expected.items():
    path = out / arm / "run" / "summary.json"
    if not path.exists():
        print(arm, "summary.json missing")
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    correct = {item["benchmark"]: item["correct"] for item in summary["benchmarks"]}
    for name in names:
        check = "" if name not in want else f" expected {want[name]} {'match' if correct[name] == want[name] else 'DIFFERS'}"
        print(arm, name, "correct", correct[name], "accuracy", f"{summary['scores'][name]:.4f}{check}")
    avg6 = summary["avg6"]
    check = f" expected 78.73 {'match' if round(avg6, 2) == 78.73 else 'DIFFERS'}" if arm == "released_step80" else ""
    print(arm, "Avg6", f"{avg6:.6f}{check}")
EOF
} > "$out/G3.txt" 2>&1
cat "$out/G3.txt"
[[ $c025 -eq 0 && $c026 -eq 0 ]]
