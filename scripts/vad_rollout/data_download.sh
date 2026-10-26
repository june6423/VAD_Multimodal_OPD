#!/bin/bash
# VLM-VAD-ROLLOUT-001 data staging (HANDOFF-VLM-VAD-ROLLOUT-001-R1, Part A item 6). No GPU work.
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> scripts/vad_rollout/data_download.sh <output_root>
# Downloads from the Hub at fixed revisions (hf download of the VAD environment):
#   $DATA_ROOT/hub/Vision-OPD-6K                               yuanqianhao/Vision-OPD-6K
#   $DATA_ROOT/models/Qwen3.5-4B                               Qwen/Qwen3.5-4B
#   $DATA_ROOT/models/VAD_for_Qwen3.5-4b                       zhangkangning/VAD_for_Qwen3.5-4b
#   $DATA_ROOT/models/Mistral-Small-3.2-24B-Instruct-2506      mistralai/Mistral-Small-3.2-24B-Instruct-2506
#   $DATA_ROOT/vad/benchmarks/<folder of prepare_data.py>      the five benchmark repositories
# The benchmark revisions are the ones of VLM-VAD-REPRO-001 R1 and R2. The script downloads each benchmark
# repository at its revision into the folder that evaluation/vision_opd_official/prepare_data.py uses, then
# runs scripts/prepare_benchmarks.sh with HF_HUB_OFFLINE=1, so snapshot_download uses the local copy.
# Then it extracts the Vision-OPD-6K archives and runs scripts/prepare_data.py --skip-download and
# --verify-only as scripts/vad_repro/prepare_check.sh (R1). The record is <output_root>/DATA_DOWNLOAD.txt.
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${DATA_ROOT:-$HOME/vad-rollout/data}"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
src="$data/hub/Vision-OPD-6K"
vad="$data/vad"
bench="$vad/benchmarks"
mkdir -p "$out" "$src" "$vad" "$bench" "$data/models"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH"
export HF_HOME="$out/hf-home" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PYTHONDONTWRITEBYTECODE=1
log="$out/DATA_DOWNLOAD.txt"
: > "$log"
status=0

# repo type, repository, revision, destination
downloads=(
  "dataset yuanqianhao/Vision-OPD-6K eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4 $src"
  "model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a $data/models/Qwen3.5-4B"
  "model zhangkangning/VAD_for_Qwen3.5-4b 1836a75ccda0ae3fde58803df57fd6c4fb4cecca $data/models/VAD_for_Qwen3.5-4b"
  "model mistralai/Mistral-Small-3.2-24B-Instruct-2506 95a6d26c4bfb886c58daf9d3f7332c857cb27b43 $data/models/Mistral-Small-3.2-24B-Instruct-2506"
  "dataset lmms-lab/vstar-bench b44023b4dca749ed8a76b85eb576627d05a1c174 $bench/vstar_data"
  "dataset inclusionAI/ZoomBench b788097e57d30510c6877824833234a73bf80d25 $bench/ZoomBench_data"
  "dataset DreamMr/HR-Bench 83b9013d6293b85dc507e87199ca52517536939c $bench/HR-Bench_data"
  "dataset yifanzhang114/MME-RealWorld-Lmms-eval f157390129e8d2b60757eb68d340aade9446b52b $bench/yifanzhang114_MME-RealWorld-Lmms-eval"
  "dataset yifanzhang114/MME-RealWorld-CN-Lmms-eval 37ac6f4e57ef211b37d1ee04a8f1268f1411b3e0 $bench/yifanzhang114_MME-RealWorld-CN-Lmms-eval"
)

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== DATA_ROOT $data"
  echo "== hf $(command -v hf)"
  echo "== df -h DATA_ROOT before"; df -h "$data"
} >> "$log" 2>&1

export HF_HUB_OFFLINE=0
for entry in "${downloads[@]}"; do
  read -r kind repo rev dest <<< "$entry"
  mkdir -p "$dest"
  start=$(date +%s)
  hf download "$repo" --repo-type "$kind" --revision "$rev" --local-dir "$dest" > "$out/download_${repo//\//_}.log" 2>&1
  code=$?
  echo "download $repo exit $code wall_seconds $(( $(date +%s) - start ))" >> "$log"
  [[ $code -eq 0 ]] || { status=1; tail -n 20 "$out/download_${repo//\//_}.log" >> "$log"; }
  # Manifest check: the Hub listing at the revision against the local files.
  python3 - "$dest" "$repo" "$kind" "$rev" >> "$log" 2>&1 <<'EOF'
import sys
from pathlib import Path

from huggingface_hub import HfApi

dest, repo, kind, rev = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
info = HfApi().repo_info(repo, repo_type=kind, revision=rev, files_metadata=True)
local = {p.relative_to(dest).as_posix(): p.stat().st_size for p in dest.rglob("*")
         if p.is_file() and ".cache" not in p.relative_to(dest).parts}
listed = {s.rfilename: s.size for s in info.siblings}
mismatch = sorted(name for name, size in listed.items() if local.get(name) != size)
print("manifest", repo, "requested", rev, "api_sha", info.sha, "listed_files", len(listed),
      "mismatch", len(mismatch), mismatch[:10], "bytes", sum(listed.values()))
sys.exit(0 if info.sha == rev and not mismatch else 1)
EOF
  [[ $? -eq 0 ]] || status=1
done

# Vision-OPD-6K archives, as scripts/vad_repro/prepare_check.sh (R1). GNU tar does not detect gzip on a
# pipe, so the first two bytes select -z (gzip magic 1f8b).
extract() {
  local dir="$1" pattern="$2" start flags=-xf magic
  if [[ -n "$(find "$dir" -type f ! -name "$pattern" -print -quit)" ]]; then
    echo "skip extraction in $dir: extracted files already exist" >> "$log"
    return 0
  fi
  magic=$(cd "$dir" && ls $pattern | sort | head -n 1 | xargs head -c 2 | od -An -tx1 | tr -d ' ')
  [[ "$magic" == 1f8b ]] && flags=-xzf
  echo "extract $dir first-part magic $magic tar $flags" >> "$log"
  start=$(date +%s)
  (cd "$dir" && ls $pattern | sort | xargs cat | tar $flags - -C .)
  local code=$?
  echo "extract $dir exit $code wall_seconds $(( $(date +%s) - start ))" >> "$log"
  return "$code"
}
extract "$src/images" 'images.tar.gz*' || status=1
extract "$src/teacher_images" 'teacher_images.tar.gz*' || status=1

export HF_HUB_OFFLINE=1
start=$(date +%s)
python3 scripts/prepare_data.py --data-dir "$src" --skip-download \
  --output "$vad/train_vad.parquet" --degraded-dir "$vad/evidence_degraded" \
  --manifest "$vad/vad_data_manifest.json" --workers 16 > "$out/prepare_data.log" 2>&1
code=$?
{ echo "prepare_data exit $code wall_seconds $(( $(date +%s) - start ))"; tail -n 40 "$out/prepare_data.log"; } >> "$log"
[[ $code -eq 0 ]] || status=1
python3 scripts/prepare_data.py --data-dir "$src" --output "$vad/train_vad.parquet" --verify-only > "$out/verify.log" 2>&1
code=$?
{ echo "verify_only exit $code"; tail -n 20 "$out/verify.log"; } >> "$log"
[[ $code -eq 0 ]] || status=1

start=$(date +%s)
BENCHMARK_DATA_DIR="$bench" bash scripts/prepare_benchmarks.sh > "$out/prepare_benchmarks.log" 2>&1
code=$?
{ echo "prepare_benchmarks exit $code wall_seconds $(( $(date +%s) - start ))"; grep -v 'Processing images' "$out/prepare_benchmarks.log" | tail -n 40; } >> "$log"
[[ $code -eq 0 ]] || status=1

{
  echo "== manifest"; cat "$vad/vad_data_manifest.json"
  echo "== file counts (archive parts excluded)"
  echo "images $(find "$src/images" -type f ! -name 'images.tar.gz*' | wc -l)"
  echo "teacher_images $(find "$src/teacher_images" -type f ! -name 'teacher_images.tar.gz*' | wc -l)"
  echo "evidence_degraded $(find "$vad/evidence_degraded" -type f | wc -l)"
  echo "== benchmark counts and revisions"
  python3 - "$bench" <<'EOF'
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
  echo "== SHA-256"
  sha256sum "$vad/train_vad.parquet" "$bench/opd_fast_benchmarks.parquet"
  echo "== sizes"
  du -sh "$src" "$vad" "$bench" "$data/models"/*
  echo "== df -h DATA_ROOT after"; df -h "$data"
} >> "$log" 2>&1

echo "== status $status" >> "$log"
exit "$status"
