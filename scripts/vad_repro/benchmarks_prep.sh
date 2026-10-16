#!/bin/bash
# VLM-VAD-REPRO-001 Part B task 3 (HANDOFF-VLM-VAD-REPRO-001-R1, gate G4). No GPU work.
# Usage: scripts/vad_repro/benchmarks_prep.sh <output_root>
# Runs scripts/prepare_benchmarks.sh --fast-only (vstar, zoombench, hrbench-4k, hrbench-8k).
# MME-RealWorld is not downloaded. The Hugging Face home stays inside <output_root>.
# The record is <output_root>/BENCHMARKS_PREP.txt.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
bench=/workspace/harness/data/vad/benchmarks
mkdir -p "$out"
cd "$root" || exit 1
log="$out/BENCHMARKS_PREP.txt"
echo "container offline variables: $(env | grep -o '^[A-Z_]*OFFLINE=[^ ]*' | tr '\n' ' ')" > "$log"
# The container sets HF_HUB_OFFLINE=1. The downloads need the Hub.
export HF_HUB_OFFLINE=0 BENCHMARK_DATA_DIR="$bench"
export HF_HOME="$out/hf-home" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PYTHONDONTWRITEBYTECODE=1

start=$(date +%s)
bash scripts/prepare_benchmarks.sh --fast-only > "$out/prepare_benchmarks.log" 2>&1
code=$?
{
  echo "prepare_benchmarks exit $code wall_seconds $(( $(date +%s) - start ))"
  echo "== log tail"; tail -n 40 "$out/prepare_benchmarks.log"
  echo "== counts and revisions"
  python - "$bench" <<'EOF'
import json
import sys
from pathlib import Path

import pandas as pd

bench = Path(sys.argv[1])
total = 0
for name in ("vstar.json", "zoombench.json", "hr_bench_4k.json", "hr_bench_8k.json"):
    path = bench / name
    count = len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None
    total += count or 0
    print("json", name, "records", count)
print("json total", total)
parquet = bench / "opd_fast_benchmarks.parquet"
if parquet.exists():
    frame = pd.read_parquet(parquet)
    print("parquet rows", len(frame), "equal_to_json_total", len(frame) == total)
    print("parquet data_source counts", frame["data_source"].value_counts().to_dict())
else:
    print("parquet missing")
for folder in sorted(bench.glob("*_data")):
    commits = sorted({p.read_text().splitlines()[0].strip()
                      for p in (folder / ".cache/huggingface/download").rglob("*.metadata")})
    print("revision", folder.name, commits)
EOF
  echo "== listing (bytes path), image folders summarized"
  (cd "$bench" && find . -maxdepth 1 -type f -printf '%s %p\n' | sort -k2)
  for d in "$bench"/*/; do echo "dir $(basename "$d") files $(find "$d" -type f -not -path '*/.cache/*' | wc -l) $(du -sh "$d" | cut -f1)"; done
  echo "== df -h /workspace after"; df -h /workspace
} >> "$log" 2>&1

exit "$code"
