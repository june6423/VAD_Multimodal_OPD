#!/bin/bash
# VLM-VAD-REPRO-001 R2 Part B task 3 (HANDOFF-VLM-VAD-REPRO-001-R2, gate G3). No GPU work.
# Usage: scripts/vad_repro/released_ckpt.sh <output_root>
# Downloads zhangkangning/VAD_for_Qwen3.5-4b at revision 1836a75ccda0ae3fde58803df57fd6c4fb4cecca to
# /workspace/harness/data/vad/VAD_for_Qwen3.5-4b. It uses huggingface-cli when the container has it,
# otherwise hf (huggingface_hub 1.x), and records which one. Then it compares the local files
# with the Hub listing at the revision. The record is <output_root>/RELEASED_CKPT.txt.
set -uo pipefail

out="$1"
repo=zhangkangning/VAD_for_Qwen3.5-4b
rev=1836a75ccda0ae3fde58803df57fd6c4fb4cecca
dest=/workspace/harness/data/vad/VAD_for_Qwen3.5-4b
mkdir -p "$out" "$dest"
log="$out/RELEASED_CKPT.txt"
export HF_HUB_OFFLINE=0 HF_HOME="$out/hf-home" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PYTHONDONTWRITEBYTECODE=1

if command -v huggingface-cli > /dev/null && huggingface-cli --help > /dev/null 2>&1; then cli=huggingface-cli; else cli=hf; fi
{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== df -h /workspace before"; df -h /workspace
  echo "== cli $cli ($(command -v "$cli"))"
} > "$log" 2>&1

start=$(date +%s)
"$cli" download "$repo" --revision "$rev" --local-dir "$dest" > "$out/download.log" 2>&1
code=$?
{
  echo "download exit $code wall_seconds $(( $(date +%s) - start ))"
  echo "== download log tail"; tail -n 20 "$out/download.log"
  echo "== listing (bytes path), .cache excluded"
  (cd "$dest" && find . -path ./.cache -prune -o -type f -printf '%s %p\n' | sort -k2)
  echo "== config.json"; cat "$dest/config.json"
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
print("model_type", json.loads((dest / "config.json").read_text()).get("model_type"))
print("model.safetensors bytes", local.get("model.safetensors"))
EOF
  echo "manifest check exit $?"
  echo "== df -h /workspace after"; df -h /workspace
} >> "$log" 2>&1

exit "$code"
