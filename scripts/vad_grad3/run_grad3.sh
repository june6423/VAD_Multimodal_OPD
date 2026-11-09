#!/bin/bash
# Part B items 2 and 4 of HANDOFF-VLM-VAD-GRAD-003-R2: the vectors of Definition 9, the Gram files and stats.py for
# M0, A1_80 and A1_130. One GPU. A restart resumes: the displacements and the Gram files are skipped when they exist,
# grads.py skips finished keys and finished micro-batches (progress files, partial vectors).
# Per checkpoint: batches.py -> rows/; grads.py run O (keys B8-f1..4, B1-f*-j0..7, R8-f1-s1..3, B8-f1@rep); grads.py run
#   Fx with VAD_PACKFIX=1 (keys B8-f1, R8-f1-s1..3); grads.py gram (then the vector files are removed).
# PILOT=1 (Part A item 9): pilot.py select (2 folds x 6 prompts per checkpoint), random partition s1 only, a forced
#   stop after one micro-batch and a resume (first checkpoint), the full row files in rows_full/ for pilot.py project
#   (PROJECTION.json). The B1-f1-j0 key uses its full row file (rows_full/), so that B1 micro-batches hold >= 2 rows
#   of both labels (pilot.py mixed_b1 checks it).
# At the end: stats.py -> STATS_GRAD3.json, then the displacement vectors are removed (DISPLACEMENT_REMOVED.txt).
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> \
#   ADAM_ROOT=<TASK-GRAD-001-009 output> [PILOT=1] scripts/vad_grad3/run_grad3.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}" adam_root="${ADAM_ROOT:?}"
pilot="${PILOT:-0}"
data="$b200/harness/data"
r1="$data/rollout001"
r053="$r1/TASK-VLM-VAD-ROLLOUT-001-053"
g3="$r1/TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS.txt"
py="$env_root/vad/bin/python"
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR VAD_PACKFIX
log="$out/RUN_GRAD3.txt"
{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD) PILOT $pilot"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  df -B1 "$out"
} >> "$log" 2>&1
"$py" scripts/vad_grad3/map_command.py "$r1/TASK-VLM-VAD-ROLLOUT-001-022/COMMAND.txt" "$out/COMMAND_B200.txt" \
  "$root" "$b200/harness" >> "$log" 2>&1 || exit 1

# Memory of the run cgroup and of the nearest ancestor cgroup with a limit (anon + shmem is the threshold value).
memlog() {
  local cg up
  cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
  up="$cg"
  while [[ "$up" != /sys/fs/cgroup && "$(cat "$up/memory.max" 2>/dev/null)" == max ]]; do up="$(dirname "$up")"; done
  echo "$1 memory run peak $(cat "$cg/memory.peak") $(grep -E '^(anon|shmem|file) ' "$cg/memory.stat" | tr '\n' ' ')" \
    "| limit $up max $(cat "$up/memory.max") peak $(cat "$up/memory.peak") $(grep -E '^(max|oom_kill) ' "$up/memory.events" | tr '\n' ' ')" \
    "| disk used $(du -sb "$out" | cut -f1) | $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
stage() {  # stage <name> <command...>: run, log the exit code and the seconds to TIMES.jsonl
  local name="$1" t0 code
  shift
  t0=$(date +%s)
  "$@"
  code=$?
  echo "{\"stage\": \"$name\", \"seconds\": $(( $(date +%s) - t0 )), \"exit\": $code}" >> "$out/TIMES.jsonl"
  echo "$name exit $code at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  return $code
}

disp="$out/displacement"
a1="$r1/TASK-VLM-VAD-ROLLOUT-001-022" a2="$r1/TASK-VLM-VAD-ROLLOUT-001-036"
if [[ ! -f "$disp/DISPLACEMENT_VECTORS.json" || $(ls "$disp"/D__*.pt 2>/dev/null | wc -l) -ne 4 ]]; then
  stage displacement "$py" scripts/vad_grad3/grads.py displacement "$data/models/Qwen3.5-4B" "$g3" "$disp" \
    "$a1/keep/global_step_80/actor" "$a1/checkpoints/global_step_130/actor" \
    "$a2/keep/global_step_80/actor" "$a2/checkpoints/global_step_130/actor" >> "$out/displacement.log" 2>&1 || exit 1
  memlog displacement >> "$log" 2>&1
fi

declare -A step=([M0]=step0 [A1_80]=step80 [A1_130]=step130)
declare -A model=([M0]="$data/models/Qwen3.5-4B" [A1_80]="$a1/merged/step_80" [A1_130]="$a1/merged/step_130")
declare -A adam=([M0]="$adam_root/V_A1_80.safetensors" [A1_80]="$adam_root/V_A1_80.safetensors"
                 [A1_130]="$adam_root/V_A1_130.safetensors")
seeds=$([[ "$pilot" == 1 ]] && echo "1" || echo "1 2 3")
first=1
for ck in M0 A1_80 A1_130; do
  dir="$out/$ck"
  [[ -f "$dir/GRAM_$ck.json" ]] && { echo "$ck GRAM exists, skipped" >> "$log"; first=0; continue; }
  mkdir -p "$dir/rows"
  ro="$r053/${step[$ck]}/rollouts.jsonl" lab="$labels_root/${step[$ck]}/LABELS.jsonl"
  if [[ ! -f "$dir/rows/BATCHES.json" ]]; then
    extra=()
    if [[ "$pilot" == 1 ]]; then
      "$py" scripts/vad_grad3/pilot.py select "$ro" "$lab" "$dir/PILOT_PROMPTS.json" >> "$log" 2>&1 || exit 1
      mkdir -p "$dir/rows_full"
      "$py" scripts/vad_grad3/batches.py "$ro" "$lab" "$dir/rows_full" >> "$log" 2>&1 || exit 1
      extra=("$dir/PILOT_PROMPTS.json")
    fi
    "$py" scripts/vad_grad3/batches.py "$ro" "$lab" "$dir/rows" "${extra[@]}" >> "$log" 2>&1 || exit 1
    if [[ "$pilot" == 1 ]]; then
      # DIRECTIVE-VLM-VAD-GRAD-003-005: the pilot B1 micro-batches of 6 prompts hold 1 row each. The full B1-f1-j0
      # key gives B1 micro-batches with >= 2 rows of both labels.
      cp "$dir/rows_full/ROWS_B1-f1-j0.jsonl" "$dir/rows/ROWS_B1-f1-j0.jsonl" || exit 1
      "$py" scripts/vad_grad3/pilot.py mixed_b1 "$dir/rows/ROWS_B1-f1-j0.jsonl" >> "$log" 2>&1 || exit 1
    fi
  fi
  keys_o=() keys_fx=(B8-f1)
  for f in 1 2 3 4; do [[ -f "$dir/rows/ROWS_B8-f$f.jsonl" ]] && keys_o+=("B8-f$f"); done
  for f in 1 2 3 4; do for j in 0 1 2 3 4 5 6 7; do
    [[ -f "$dir/rows/ROWS_B1-f$f-j$j.jsonl" ]] && keys_o+=("B1-f$f-j$j")
  done; done
  for s in $seeds; do keys_o+=("R8-f1-s$s"); keys_fx+=("R8-f1-s$s"); done
  keys_o+=("B8-f1@rep")
  args=("$root" "$out/COMMAND_B200.txt" "${model[$ck]}" "$ro" "$dir/rows" "$g3" "$dir")
  echo "$ck keys O ${keys_o[*]} | Fx ${keys_fx[*]}" >> "$log"
  if [[ "$pilot" == 1 && $first -eq 1 && ! -f "$dir/STOPPED_ONCE" ]]; then
    STOP_AFTER_MICRO=1 stage "stop_$ck" "$py" scripts/vad_grad3/grads.py run "${args[@]}" O "${keys_o[@]}" \
      >> "$dir/grads_O.log" 2>&1
    code=$?
    [[ "$code" == 75 ]] || { echo "$ck forced stop gave exit $code, not 75" >> "$log"; exit 1; }
    touch "$dir/STOPPED_ONCE"
  fi
  stage "O_$ck" "$py" scripts/vad_grad3/grads.py run "${args[@]}" O "${keys_o[@]}" >> "$dir/grads_O.log" 2>&1 || exit 1
  memlog "O_$ck" >> "$log" 2>&1
  VAD_PACKFIX=1 stage "Fx_$ck" "$py" scripts/vad_grad3/grads.py run "${args[@]}" Fx "${keys_fx[@]}" \
    >> "$dir/grads_Fx.log" 2>&1 || exit 1
  memlog "Fx_$ck" >> "$log" 2>&1
  stage "gram_$ck" "$py" scripts/vad_grad3/grads.py gram "$dir" "$disp" "${adam[$ck]}" "$g3" "$ck" \
    >> "$dir/gram.log" 2>&1 || exit 1
  echo "$ck vector files left: $(find "$dir/vectors" -name '*.pt' | wc -l)" >> "$log"
  first=0
done

stage stats "$py" scripts/vad_grad3/stats.py "$out" "$out/STATS_GRAD3.json" M0 A1_80 A1_130 >> "$log" 2>&1 || exit 1
if [[ "$pilot" == 1 ]]; then
  "$py" scripts/vad_grad3/pilot.py project "$out" "$out/PROJECTION.json" M0 A1_80 A1_130 >> "$log" 2>&1 || exit 1
fi
rm -f "$disp"/D__*.pt
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) removed the displacement vectors; left: $(ls "$disp"/*.pt 2>/dev/null | wc -l)" \
  > "$out/DISPLACEMENT_REMOVED.txt"
memlog end >> "$log" 2>&1
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
