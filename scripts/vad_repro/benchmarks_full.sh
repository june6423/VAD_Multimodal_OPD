#!/bin/bash
# VLM-VAD-REPRO-001 R2 Part B task 2 (HANDOFF-VLM-VAD-REPRO-001-R2, gate G2). No GPU work.
# Usage: scripts/vad_repro/benchmarks_full.sh <output_root>
# Runs scripts/prepare_benchmarks.sh without --fast-only (all six benchmarks). The four fast JSON
# files of R1 task 003 exist, so prepare_data.py skips them. The script rebuilds
# opd_fast_benchmarks.parquet from the same JSON files and records its SHA-256 before and after.
# The Hugging Face home stays inside <output_root>. The record is <output_root>/BENCHMARKS_FULL.txt.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
bench=/workspace/harness/data/vad/benchmarks
mkdir -p "$out"
cd "$root" || exit 1
log="$out/BENCHMARKS_FULL.txt"
echo "container offline variables: $(env | grep -o '^[A-Z_]*OFFLINE=[^ ]*' | tr '\n' ' ')" > "$log"
export HF_HUB_OFFLINE=0 BENCHMARK_DATA_DIR="$bench"
export HF_HOME="$out/hf-home" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PYTHONDONTWRITEBYTECODE=1

{
  echo "== df -h /workspace before"; df -h /workspace
  echo "== opd_fast_benchmarks.parquet before"; sha256sum "$bench/opd_fast_benchmarks.parquet"
} >> "$log" 2>&1

start=$(date +%s)
bash scripts/prepare_benchmarks.sh > "$out/prepare_benchmarks.log" 2>&1
code=$?
{
  echo "prepare_benchmarks exit $code wall_seconds $(( $(date +%s) - start ))"
  echo "== log tail"; grep -v 'Processing images' "$out/prepare_benchmarks.log" | tail -n 40
  echo "== opd_fast_benchmarks.parquet after"; sha256sum "$bench/opd_fast_benchmarks.parquet"
  echo "== counts and revisions"
  python - "$bench" <<'EOF'
import json
import sys
from pathlib import Path

bench = Path(sys.argv[1])
for name in ("vstar.json", "zoombench.json", "hr_bench_4k.json", "hr_bench_8k.json",
             "MME_RealWorld.json", "MME_RealWorld_CN.json"):
    path = bench / name
    count = len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None
    print("json", name, "records", count)
for folder in sorted(p for p in bench.iterdir() if (p / ".cache/huggingface/download").is_dir()):
    commits = sorted({p.read_text().splitlines()[0].strip()
                      for p in (folder / ".cache/huggingface/download").rglob("*.metadata")})
    print("revision", folder.name, commits)
EOF
  echo "== listing (bytes path), folders summarized"
  (cd "$bench" && find . -maxdepth 1 -type f -printf '%s %p\n' | sort -k2)
  for d in "$bench"/*/; do echo "dir $(basename "$d") files $(find "$d" -type f -not -path '*/.cache/*' | wc -l) $(du -sh "$d" | cut -f1)"; done
  echo "== df -h /workspace after"; df -h /workspace
} >> "$log" 2>&1

exit "$code"
