#!/bin/bash
# VLM-VAD-ROLLOUT-001 archive of the steps that a resume repeats (HANDOFF-VLM-VAD-ROLLOUT-001-R5 Part B item 7,
# DT routing of 2026-10-30). Usage: scripts/vad_rollout/archive_steps.sh <arm output_root> <first step> <output_root>
# A resume from the last checkpoint repeats the later steps. The rollout dump rewrites <step>.jsonl
# (ray_trainer.py line 515), but the tokstats writer appends to step_<step>_rank_<rank>.jsonl (dp_actor.py
# lines 380-381). This script moves rollouts/<step>.jsonl and tokstats/step_<step>_rank_*.jsonl of every
# step >= <first step> into <arm output_root>/archive_steps_<first step>/rollouts and /tokstats and writes
# ARCHIVE_STEPS.txt into <output_root>: per file the line count, the SHA-256 before and after the move, and
# the path. It deletes nothing and stops if the archive folder exists. No GPU.
set -uo pipefail

arm="$1"
first="$2"
mkdir -p "$3"
out="$(cd "$3" && pwd)"
report="$out/ARCHIVE_STEPS.txt"
archive="archive_steps_$first"
cd "$arm" || exit 1
if [[ -e "$archive" ]]; then echo "archive $arm/$archive exists; nothing moved" | tee "$report"; exit 2; fi
mkdir -p "$archive/rollouts" "$archive/tokstats"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "latest_checkpointed_iteration $(cat checkpoints/latest_checkpointed_iteration.txt 2>/dev/null || echo none)"
  echo "first_step $first archive $arm/$archive"
  echo "== lines sha256_before sha256_after file (archive path = $archive/<file>)"
} > "$report"
status=0
moved=0
for f in rollouts/*.jsonl tokstats/step_*_rank_*.jsonl; do
  [[ -f "$f" ]] || continue
  step="${f##*/}"
  step="${step%.jsonl}"
  step="${step#step_}"
  step="${step%%_rank_*}"
  [[ "$step" =~ ^[0-9]+$ && "$step" -ge "$first" ]] || continue
  lines=$(wc -l < "$f")
  before=$(sha256sum "$f" | cut -d' ' -f1)
  mv -n "$f" "$archive/$f"
  after=$(sha256sum "$archive/$f" 2>/dev/null | cut -d' ' -f1)
  [[ "$after" == "$before" && ! -e "$f" ]] || status=1
  echo "$lines $before ${after:-missing} $f" >> "$report"
  moved=$((moved + 1))
done
{
  echo "moved $moved status $status"
  echo "remaining rollouts $(find rollouts -maxdepth 1 -name '*.jsonl' | wc -l) tokstats $(find tokstats -maxdepth 1 -name 'step_*_rank_*.jsonl' | wc -l)"
} >> "$report"
cat "$report"
exit "$status"
