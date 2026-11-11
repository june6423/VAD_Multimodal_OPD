#!/bin/bash
# DIRECTIVE-VLM-VAD-VIEW-001-003 checks P and V after the FULL task (VALIDATION; reads the FULL outputs, writes only
# under <output_root>).
# 1. CHECK_PV.txt: git HEAD, GPU, disk. COMMAND_B200.txt (map_command.py, as run_view.sh).
# 2. Check P: check_pv.py p -> CHECK_P.json.
# 3. Check V: check_pv.py gen -> views_gen/ (generated v(p1, 2) and the released v0 for each kept row);
#    stage1.py fresh (unchanged, M0 teacher) -> GEN_FRESH.jsonl; check_pv.py v -> CHECK_V.json, CHECK_V_SIZES.tsv.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> FULL_ROOT=<FULL task output root> \
#   scripts/vad_view/check_pv.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" full="${FULL_ROOT:?}"
r1="$b200/harness/data/rollout001"
py="$env_root/vad/bin/python"
vv=scripts/vad_view
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR VAD_PACKFIX
log="$out/CHECK_PV.txt"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) FULL_ROOT $full"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  df -B1 "$out"
} >> "$log" 2>&1
"$py" scripts/vad_grad3/map_command.py "$r1/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1

run() {  # run <name> <command...>: log the exit code and the seconds; stop on a failure
  local name="$1" t0 code
  shift
  t0=$(date +%s)
  "$@"
  code=$?
  echo "$name exit $code seconds $(( $(date +%s) - t0 )) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  [[ $code -eq 0 ]] || exit 1
}
run check_p "$py" $vv/check_pv.py p "$full" "$out/CHECK_P.json" >> "$log" 2>&1
run gen "$py" $vv/check_pv.py gen "$full/views" "$full/views/P1_FIT.json" "$out/views_gen" >> "$log" 2>&1
run fresh "$py" $vv/stage1.py fresh "$root" "$out/COMMAND_B200.txt" "$out/views_gen" "$out/GEN_FRESH.jsonl" \
  >> "$out/stage1.log" 2>&1
run check_v "$py" $vv/check_pv.py v "$full/stage1/FRESH.jsonl" "$out/GEN_FRESH.jsonl" "$out/views_gen" \
  "$out/CHECK_V.json" "$out/CHECK_V_SIZES.tsv" >> "$log" 2>&1
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
