#!/bin/bash
# VLM-VAD-ROLLOUT-001 arm A2 PRE_RUN checks (HANDOFF-VLM-VAD-ROLLOUT-001-R6, Part B items 1 and 2, gates G1 and G2).
# Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> WANDB_ENV_FILE=<file> JUDGE_API_BASE=<endpoint> \
#   scripts/vad_rollout/r6_prerun.sh <A1 output root> <env_install output root> <output_root>
# G1: extracts the tree of commit 2627743a (git archive) into <output_root>/src and runs its
#   scripts/vad_rollout/train_arm.sh A2 --dry-run there. Then it diffs the main_ppo argument list against the
#   A1 list in <A1 output root>/COMMAND.txt. Records: DRYRUN_A2.txt, ARGS_A1.txt, ARGS_A2.txt, ARGS_DIFF.txt.
# G2: pip freeze of the VAD and the judge environment against pip-freeze-vad.txt and pip-freeze-judge.txt of
#   <env_install output root>, then judge_check.sh. Records: FREEZE_DIFF.txt, JUDGE_CHECK.txt (judge_check.sh).
# No GPU. Summary: R6_PRERUN.txt.
set -uo pipefail

a1="$1"
install="$2"
out="$3"
commit=2627743a0cfc14350b8a04c377be251dc4ff169f
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
mkdir -p "$out"
log="$out/R6_PRERUN.txt"
cd "$root" || exit 1
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD)" > "$log"

# G1
rm -rf "$out/src"
mkdir -p "$out/src"
git -c safe.directory="$root" archive "$commit" | tar -x -C "$out/src"
echo "== src tree of $commit (git archive exit ${PIPESTATUS[0]})" >> "$log"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash "$out/src/scripts/vad_rollout/train_arm.sh" A2 "$out/A2_dry_run_output" --dry-run \
  > "$out/DRYRUN_A2.txt" 2>&1
echo "dry run exit $?" >> "$log"
grep -E '^ROLLOUT_N ' "$out/DRYRUN_A2.txt" >> "$log"
args() { sed -n '/^== main_ppo arguments/,$p' "$1" | tail -n +2; }
args "$a1/COMMAND.txt" > "$out/ARGS_A1.txt"
# The project root of the extracted tree differs from the A1 clone; A2 trains in the A1 clone (slot vad_train).
a1_root="${A1_PROJECT_ROOT:-/workspace/harness/vad_train}"
args "$out/DRYRUN_A2.txt" | sed "s#$out/src#$a1_root#g" > "$out/ARGS_A2.txt"
diff "$out/ARGS_A1.txt" "$out/ARGS_A2.txt" > "$out/ARGS_DIFF.txt"
{
  echo "A2 arguments: project root $out/src replaced by $a1_root before the diff"
  echo "argument counts: A1 $(wc -l < "$out/ARGS_A1.txt") A2 $(wc -l < "$out/ARGS_A2.txt")"
  echo "== diff A1 -> A2"
  cat "$out/ARGS_DIFF.txt"
} >> "$log"

# G2
{
  echo "== VAD freeze against $install/pip-freeze-vad.txt (editable lines removed)"
  diff <(grep -v '^-e ' "$install/pip-freeze-vad.txt") <("$env_root/vad/bin/pip" freeze 2>/dev/null | grep -v '^-e ')
  echo "vad diff exit $?"
  echo "== editable lines now"
  "$env_root/vad/bin/pip" freeze 2>/dev/null | grep '^-e '
  echo "== judge freeze against $install/pip-freeze-judge.txt"
  diff "$install/pip-freeze-judge.txt" <("$env_root/judge/bin/pip" freeze 2>/dev/null)
  echo "judge diff exit $?"
} > "$out/FREEZE_DIFF.txt" 2>&1
grep -E 'diff exit' "$out/FREEZE_DIFF.txt" >> "$log"
bash scripts/vad_rollout/judge_check.sh "$out"
echo "judge_check exit $?" >> "$log"
grep -E '^(models|reply|RESULT)' "$out/JUDGE_CHECK.txt" >> "$log"
rm -rf "$out/src"
