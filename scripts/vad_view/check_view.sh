#!/bin/bash
# HANDOFF-VLM-VAD-VIEW-001-R2 Part B item 1: env check, input hashes, view check (G3), gradient check (G2).
# 1. CHECK_VIEW.txt: git HEAD, the changed paths from 50801aa6 (G1 scope), GPU, memory, disk, package versions,
#    VAD_PACKFIX unset.
# 2. inputs.py (vad_grad3): INPUTS.sha256 and INPUTS_CHECK.json; INPUTS_VIEW.sha256: the new inputs of VIEW-001.
# 3. views.py fit, build (all 6241 rows), g3: P1_FIT.json, views/REMOVED.json, G3_VIEWS.json, IMAGE_TOKENS.tsv.
# 4. G2 at M0, fold 1, v0: batches.py -> M0/rows; grads.py run O B8-f1@g2 (GRAD-003 code) -> g2/old; vgrads.py
#    B8-f1@p1-z2.0 -> g2/new; vg2.py -> G2.json (exit 3 = gate failed). Then the two vector files are removed.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> \
#   ADAM_ROOT=<TASK-GRAD-001-009 output> scripts/vad_view/check_view.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}" adam_root="${ADAM_ROOT:?}"
data="$b200/harness/data"
r1="$data/rollout001"
r053="$r1/TASK-VLM-VAD-ROLLOUT-001-053"
g3="$r1/TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS.txt"
parquet="$data/vad/train_vad.parquet" src="$data/hub/Vision-OPD-6K/train.jsonl"
mme="$data/vad/benchmarks/MME_RealWorld.json"
base="$data/models/Qwen3.5-4B"
py="$env_root/vad/bin/python"
vv=scripts/vad_view
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR VAD_PACKFIX
log="$out/CHECK_VIEW.txt"
status=0
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== paths changed from 50801aa6"; git -c safe.directory="$root" diff --name-status 50801aa6 HEAD
  echo "== VAD_PACKFIX=${VAD_PACKFIX:-unset}"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total --format=csv
  grep -E "MemTotal|MemAvailable" /proc/meminfo
  echo "cgroup memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
  df -B1 "$out"
} > "$log" 2>&1
"$py" $vv/versions.py >> "$log" 2>&1
"$py" scripts/vad_grad3/map_command.py "$r1/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1

"$py" scripts/vad_grad3/inputs.py "$b200" "$labels_root" "$adam_root" "$out" >> "$log" 2>&1
code=$?; echo "inputs.py exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
sha256sum "$parquet" "$src" "$mme" "$r053/step130/rollouts.jsonl" "$adam_root/V_A1_80.safetensors" \
  > "$out/INPUTS_VIEW.sha256" 2>> "$log"
code=$?; echo "sha256sum exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1

v="$out/views"
mkdir -p "$v"
"$py" $vv/views.py fit "$parquet" "$src" "$out/P1_FIT.json" >> "$log" 2>&1
code=$?; echo "views.py fit exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
"$py" $vv/views.py build "$parquet" "$src" "$out/P1_FIT.json" "$base" "$v" >> "$log" 2>&1
code=$?; echo "views.py build exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
cp "$v/REMOVED.json" "$out/VIEWS_REMOVED.json" 2>/dev/null
"$py" $vv/views.py g3 "$v" "$out/P1_FIT.json" "$out/G3_VIEWS.json" "$out/IMAGE_TOKENS.tsv" >> "$log" 2>&1
code=$?; echo "views.py g3 exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
[[ $status -eq 0 ]] || { echo "== RESULT FAILED before G2" >> "$log"; exit 1; }

mkdir -p "$out/M0/rows"
"$py" scripts/vad_grad3/batches.py "$r053/step0/rollouts.jsonl" "$labels_root/step0/LABELS.jsonl" "$out/M0/rows" \
  >> "$log" 2>&1 || { echo "== RESULT FAILED batches.py" >> "$log"; exit 1; }
args=("$root" "$out/COMMAND_B200.txt" "$base" "$r053/step0/rollouts.jsonl" "$out/M0/rows" "$g3")
"$py" scripts/vad_grad3/grads.py run "${args[@]}" "$out/g2/old" O B8-f1@g2 >> "$out/g2_old.log" 2>&1
code=$?; echo "grads.py run O B8-f1@g2 exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
"$py" $vv/vgrads.py "${args[@]}" "$v" "$out/g2/new" B8-f1@p1-z2.0 >> "$out/g2_new.log" 2>&1
code=$?; echo "vgrads.py B8-f1@p1-z2.0 exit $code" >> "$log"; [[ $code -eq 0 ]] || status=1
if [[ $status -eq 0 ]]; then
  "$py" $vv/vg2.py "$out/g2/old/vectors/O__B8-f1@g2__total.pt" "$out/g2/new/vectors/O__B8-f1@p1-z2.0__total.pt" \
    "$out/G2.json" >> "$log" 2>&1
  code=$?; echo "vg2.py exit $code (3 = gate failed)" >> "$log"; [[ $code -eq 0 ]] || status=1
fi
cp "$out/g2/old/progress_O__B8-f1@g2.json" "$out/G2_PROGRESS_OLD.json" 2>/dev/null
cp "$out/g2/new/progress_O__B8-f1@p1-z2.0.json" "$out/G2_PROGRESS_NEW.json" 2>/dev/null
rm -f "$out/g2/old/vectors/"*.pt "$out/g2/new/vectors/"*.pt
echo "removed the G2 vector files; left: $(find "$out/g2" -name '*.pt' | wc -l)" >> "$log"
echo "cgroup memory.peak $(cat /sys/fs/cgroup/memory.peak 2>/dev/null) events $(tr '\n' ' ' < /sys/fs/cgroup/memory.events 2>/dev/null)" >> "$log"
echo "== RESULT $([[ $status -eq 0 ]] && echo OK || echo FAILED) $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
exit "$status"
