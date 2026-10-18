#!/bin/bash
# VLM-VAD-REPRO-001 R3 judge download (DT STATUS after TASK-VLM-VAD-REPRO-001-014), container VAD_A100_001.
# No GPU work. Usage: scripts/vad_repro/judge_download.sh <output_root>
# /workspace/hf_cache no longer exists, so this script downloads mistralai/Mistral-Small-3.2-24B-Instruct-2506
# at revision 95a6d26c4bfb886c58daf9d3f7332c857cb27b43 to
# /workspace/harness/data/vad/judge/Mistral-Small-3.2-24B-Instruct-2506 and compares the local files with
# the Hub listing at that revision. The record is <output_root>/JUDGE_DOWNLOAD.txt.
set -uo pipefail

out="$1"
repo=mistralai/Mistral-Small-3.2-24B-Instruct-2506
rev=95a6d26c4bfb886c58daf9d3f7332c857cb27b43
dest=/workspace/harness/data/vad/judge/Mistral-Small-3.2-24B-Instruct-2506
mkdir -p "$out" "$dest"
log="$out/JUDGE_DOWNLOAD.txt"
export HF_HUB_OFFLINE=0 HF_HOME="$out/hf-home" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PYTHONDONTWRITEBYTECODE=1

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== df -h /workspace before"; df -h /workspace
} > "$log" 2>&1

start=$(date +%s)
hf download "$repo" --revision "$rev" --local-dir "$dest" > "$out/download.log" 2>&1
code=$?
{
  echo "hf download exit $code wall_seconds $(( $(date +%s) - start ))"
  echo "== download log tail"; tail -n 20 "$out/download.log"
  echo "== listing (bytes path), .cache excluded"
  (cd "$dest" && find . -path ./.cache -prune -o -type f -printf '%s %p\n' | sort -k2)
  echo "== manifest check"
  python - "$dest" "$repo" "$rev" <<'EOF'
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

dest, repo, rev = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
info = HfApi().repo_info(repo, revision=rev, files_metadata=True)
local = {p.relative_to(dest).as_posix(): p.stat().st_size for p in dest.rglob("*")
         if p.is_file() and ".cache" not in p.relative_to(dest).parts}
metadata = sorted({m.read_text().splitlines()[0].strip()
                   for m in (dest / ".cache/huggingface/download").rglob("*.metadata")})
print("requested_revision", rev, "api_sha", info.sha, "metadata_commits", metadata)
bad = 0
for s in info.siblings:
    ok = local.get(s.rfilename) == s.size
    bad += not ok
    print(f"{'OK' if ok else 'DIFF'} {s.rfilename} listed {s.size} local {local.get(s.rfilename)}")
print("listed_files", len(info.siblings), "local_files", len(local), "mismatch", bad,
      "extra_local", sorted(set(local) - {s.rfilename for s in info.siblings}))
print("local_total_bytes", sum(local.values()))
config = dest / "config.json"
if config.exists():
    print("model_type", json.loads(config.read_text()).get("model_type"))
EOF
  echo "manifest check exit $?"
  echo "== df -h /workspace after"; df -h /workspace
} >> "$log" 2>&1

exit "$code"
