#!/bin/bash
# VLM-VAD-REPRO-001 Part B task 2 (HANDOFF-VLM-VAD-REPRO-001-R1, gate G3). No GPU work.
# Usage: scripts/vad_repro/prepare_check.sh <output_root>
# Extracts the Vision-OPD-6K image archives in place (the archive parts stay), then runs
# scripts/prepare_data.py --skip-download and --verify-only. original_images.tar.gz stays packed.
# Writes under the dataset folder, /workspace/harness/data/vad and <output_root>.
# The record is <output_root>/PREPARE_CHECK.txt.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
src=/workspace/harness/data/vision-opd/Vision-OPD-6K
vad=/workspace/harness/data/vad
mkdir -p "$out" "$vad"
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1
log="$out/PREPARE_CHECK.txt"
: > "$log"
status=0

# Same order as extract_split_archive: the parts in sorted order, one tar stream.
# GNU tar does not detect gzip on a pipe, so the first two bytes select -z (gzip magic 1f8b).
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

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== df -h /workspace before"; df -h /workspace
  echo "== archive parts"; ls -l "$src/images" | grep 'images.tar.gz'; ls -l "$src/teacher_images" | grep 'teacher_images.tar.gz'
} >> "$log" 2>&1

extract "$src/images" 'images.tar.gz*' || status=1
extract "$src/teacher_images" 'teacher_images.tar.gz*' || status=1

if [[ $status -eq 0 ]]; then
  start=$(date +%s)
  echo "== prepare_data.py --skip-download" >> "$log"
  python scripts/prepare_data.py --data-dir "$src" --skip-download \
    --output "$vad/train_vad.parquet" --degraded-dir "$vad/evidence_degraded" \
    --manifest "$vad/vad_data_manifest.json" --workers 16 > "$out/prepare_data.log" 2>&1
  code=$?
  echo "prepare_data exit $code wall_seconds $(( $(date +%s) - start ))" >> "$log"
  tail -n 40 "$out/prepare_data.log" >> "$log"
  [[ $code -eq 0 ]] || status=1
fi

echo "== prepare_data.py --verify-only" >> "$log"
python scripts/prepare_data.py --data-dir "$src" --output "$vad/train_vad.parquet" --verify-only > "$out/verify.log" 2>&1
code=$?
echo "verify_only exit $code" >> "$log"
tail -n 20 "$out/verify.log" >> "$log"
[[ $code -eq 0 ]] || status=1

{
  echo "== manifest"; cat "$vad/vad_data_manifest.json"
  echo "== file counts (archive parts excluded)"
  echo "images $(find "$src/images" -type f ! -name 'images.tar.gz*' | wc -l)"
  echo "teacher_images $(find "$src/teacher_images" -type f ! -name 'teacher_images.tar.gz*' | wc -l)"
  echo "evidence_degraded $(find "$vad/evidence_degraded" -type f | wc -l)"
  echo "== archive parts after (kept)"; ls -l "$src/images" | grep 'images.tar.gz'; ls -l "$src/teacher_images" | grep 'teacher_images.tar.gz'
  echo "== parquet"; ls -l "$vad/train_vad.parquet"; sha256sum "$vad/train_vad.parquet"
  echo "== du -sh"; du -sh "$src/images" "$src/teacher_images" "$vad"
  echo "== df -h /workspace after"; df -h /workspace
} >> "$log" 2>&1

exit "$status"
