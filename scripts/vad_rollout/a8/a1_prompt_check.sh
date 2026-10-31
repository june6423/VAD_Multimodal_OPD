#!/bin/bash
# R9 G4 read check (DT answer after TASK-047): the A1 prompts of the given dataset rows against the A8 generate.py
# prompt, the prompt token counts per step, and the A1 step-1 (Base) statistics. It reads only and uses no GPU.
# Usage: ENV_ROOT=<environment root> a1_prompt_check.sh <A1 COMMAND.txt> <A1 output root> <output root> <row> [...]
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
env_root="${ENV_ROOT:?ENV_ROOT must name the environment root}"
command_txt="$1"
a1_root="$2"
out="$3"
shift 3
mkdir -p "$out"
export PATH="$env_root/vad/bin:$PATH"
cd "$root" || exit 1
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git rev-parse HEAD 2>/dev/null)" > "$out/a1_prompt_check.log"
python3 scripts/vad_rollout/a8/a1_prompt_check.py "$root" "$command_txt" "$a1_root" "$out" "$@" >> "$out/a1_prompt_check.log" 2>&1
status=$?
cat "$out/a1_prompt_check.log"
exit "$status"
