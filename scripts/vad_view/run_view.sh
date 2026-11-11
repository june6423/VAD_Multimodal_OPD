#!/bin/bash
# HANDOFF-VLM-VAD-VIEW-001-R3 Part B items 2 and 4 (PILOT=1: Part A item 10) with DIRECTIVE-VLM-VAD-VIEW-001-001.
# One GPU. A restart resumes: each stage skips its finished work (done files, output lines, progress files).
# Stage 1: views.py fit and build (views/), stage1.py fresh and prefix (stage1/), stats1.py -> STATS_VIEW1.json, then
#   STAGE1_DONE.txt (ET fetches the stage 1 outputs and sends STATUS to DT; the script continues).
# Stage 2: heldout.py select -> HELDOUT.json; grads.py displacement (vad_grad3, unchanged); for M0 and A1_80:
#   batches.py -> <ck>/rows/, vgrads.py (keys B8-f<f>@<view> and B8-f<f>@geo, the geometric pool of the views of
#   GEO_VIEWS = the stage 2 views), heldout.py grad, vgram.py gram, linesearch.py (fp32),
#   vgram.py remove; then stats2.py -> STATS_VIEW2.json; then the displacement vectors are removed.
# PILOT=1: pilot.py select (vad_grad3: 2 folds x 6 prompts with the longest-response prompt) for each checkpoint,
#   vpilot.py select (40 fresh rows, 20 trajectories per step, the view rows), views v0 and p0.30-z3.0 and the
#   geometric pool of these 2 views, held-out
#   subset of 2 parts x 10 items and 10 set-B items, a forced stop after one line (stage 1 fresh) and after one
#   micro-batch (first checkpoint vgrads) with a resume, the line search on MEAN__V, alpha 1, 10 + 10 items, and
#   vpilot.py project -> PROJECTION.json.
# Usage: B200_ROOT=<root> ENV_ROOT=<environment root> LABELS_ROOT=<TASK-GRAD-001-005 output> \
#   ADAM_ROOT=<TASK-GRAD-001-009 output> [PILOT=1] scripts/vad_view/run_view.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}" labels_root="${LABELS_ROOT:?}" adam_root="${ADAM_ROOT:?}"
pilot="${PILOT:-0}"
data="$b200/harness/data"
r1="$data/rollout001"
r053="$r1/TASK-VLM-VAD-ROLLOUT-001-053"
g3="$r1/TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS.txt"
parquet="$data/vad/train_vad.parquet" src="$data/hub/Vision-OPD-6K/train.jsonl"
mme="$data/vad/benchmarks/MME_RealWorld.json" bench="$data/vad/benchmarks"
base="$data/models/Qwen3.5-4B"
a1="$r1/TASK-VLM-VAD-ROLLOUT-001-022" a2="$r1/TASK-VLM-VAD-ROLLOUT-001-036"
adam="$adam_root/V_A1_80.safetensors"
py="$env_root/vad/bin/python"
vv=scripts/vad_view
mkdir -p "$out"
cd "$root" || exit 1
export PATH="$env_root/vad/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 \
  PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset VAD_TOKSTATS_DIR VAD_PACKFIX
log="$out/RUN_VIEW.txt"
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
forced_stop() {  # forced_stop <name> <env var> <command...>: once, run with <env var>=1 and expect exit 75
  local name="$1" var="$2" code
  shift 2
  [[ -f "$out/STOPPED_$name" ]] && return 0
  stage "stop_$name" env "$var=1" "$@"
  code=$?
  [[ "$code" == 75 ]] || { echo "$name forced stop gave exit $code, not 75" >> "$log"; exit 1; }
  touch "$out/STOPPED_$name"
}

declare -A step=([M0]=step0 [A1_80]=step80)
declare -A model=([M0]="$base" [A1_80]="$a1/merged/step_80")
v="$out/views" s1="$out/stage1"
mkdir -p "$v" "$s1"
build_rows=() fresh_rows=() held="$out/HELDOUT.json" ls_env=()
views2=(p0.00-z1.5 p0.00-z2.0 p0.00-z3.0 p1-z1.5 p1-z2.0 p1-z3.0 p0.30-z2.0 p0.30-z3.0)
if [[ "$pilot" == 1 ]]; then
  for ck in M0 A1_80; do
    mkdir -p "$out/$ck"
    [[ -f "$out/$ck/PILOT_PROMPTS.json" ]] || "$py" scripts/vad_grad3/pilot.py select "$r053/${step[$ck]}/rollouts.jsonl" \
      "$labels_root/${step[$ck]}/LABELS.jsonl" "$out/$ck/PILOT_PROMPTS.json" >> "$log" 2>&1 || exit 1
  done
  [[ -f "$out/PILOT_BUILD.json" ]] || "$py" $vv/vpilot.py select "$parquet" "$src" "$base" "$r053" \
    "$out/M0/PILOT_PROMPTS.json" "$out/A1_80/PILOT_PROMPTS.json" "$out" >> "$log" 2>&1 || exit 1
  build_rows=("$out/PILOT_BUILD.json") fresh_rows=("$out/PILOT_FRESH.json") held="$out/PILOT_HELDOUT.json"
  ls_env=(DIRECTIONS=MEAN__V ALPHAS=1 LS_PER_SET=10 LS_FLOOR_PASS=1)
  views2=(p1-z2.0 p0.30-z3.0)
fi
GEO_VIEWS="$(IFS=,; echo "${views2[*]}")"
export GEO_VIEWS
echo "GEO_VIEWS $GEO_VIEWS" >> "$log"

# Stage 1
[[ -f "$v/P1_FIT.json" ]] || stage views_fit "$py" $vv/views.py fit "$parquet" "$src" "$v/P1_FIT.json" \
  >> "$v/views.log" 2>&1 || exit 1
[[ -f "$v/REMOVED.json" ]] || stage views_build "$py" $vv/views.py build "$parquet" "$src" "$v/P1_FIT.json" "$base" \
  "$v" "${build_rows[@]}" >> "$v/views.log" 2>&1 || exit 1
memlog views >> "$log" 2>&1
if [[ ! -f "$s1/FRESH.done" ]]; then
  [[ "$pilot" == 1 ]] && forced_stop fresh STOP_AFTER "$py" $vv/stage1.py fresh "$root" "$out/COMMAND_B200.txt" "$v" \
    "$s1/FRESH.jsonl" "${fresh_rows[@]}" >> "$s1/stage1.log" 2>&1
  stage fresh "$py" $vv/stage1.py fresh "$root" "$out/COMMAND_B200.txt" "$v" "$s1/FRESH.jsonl" "${fresh_rows[@]}" \
    >> "$s1/stage1.log" 2>&1 || exit 1
  touch "$s1/FRESH.done"
fi
for s in 0 80 130; do
  [[ -f "$s1/PREFIX_step$s.done" ]] && continue
  trajs=()
  [[ "$pilot" == 1 ]] && trajs=("$out/PILOT_PREFIX_step$s.json")
  stage "prefix_step$s" "$py" $vv/stage1.py prefix "$root" "$out/COMMAND_B200.txt" "$v" "$r053/step$s/rollouts.jsonl" \
    "$s1/PREFIX_step$s.jsonl" "${trajs[@]}" >> "$s1/stage1.log" 2>&1 || exit 1
  touch "$s1/PREFIX_step$s.done"
done
memlog stage1 >> "$log" 2>&1
stage stats1 "$py" $vv/stats1.py "$s1" "$out/STATS_VIEW1.json" 0 80 130 >> "$log" 2>&1 || exit 1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) stage 1 done" > "$out/STAGE1_DONE.txt"

# Stage 2
[[ -f "$out/HELDOUT.json" ]] || stage heldout_select "$py" $vv/heldout.py select "$mme" "$out/HELDOUT.json" \
  >> "$log" 2>&1 || exit 1
if [[ "$pilot" == 1 && ! -f "$held" ]]; then
  "$py" $vv/heldout.py subset "$out/HELDOUT.json" 2 10 10 "$held" >> "$log" 2>&1 || exit 1
fi
disp="$out/displacement"
if [[ ! -f "$disp/DISPLACEMENT_VECTORS.json" || $(ls "$disp"/D__*.pt 2>/dev/null | wc -l) -ne 4 ]]; then
  stage displacement "$py" scripts/vad_grad3/grads.py displacement "$base" "$g3" "$disp" \
    "$a1/keep/global_step_80/actor" "$a1/checkpoints/global_step_130/actor" \
    "$a2/keep/global_step_80/actor" "$a2/checkpoints/global_step_130/actor" >> "$out/displacement.log" 2>&1 || exit 1
  memlog displacement >> "$log" 2>&1
fi
first=1
for ck in M0 A1_80; do
  dir="$out/$ck"
  [[ -f "$dir/VECTORS_REMOVED.json" ]] && { echo "$ck vectors removed, skipped" >> "$log"; first=0; continue; }
  mkdir -p "$dir/rows"
  ro="$r053/${step[$ck]}/rollouts.jsonl" lab="$labels_root/${step[$ck]}/LABELS.jsonl"
  if [[ ! -f "$dir/rows/BATCHES.json" ]]; then
    extra=()
    if [[ "$pilot" == 1 ]]; then
      mkdir -p "$dir/rows_full"
      "$py" scripts/vad_grad3/batches.py "$ro" "$lab" "$dir/rows_full" >> "$log" 2>&1 || exit 1
      extra=("$dir/PILOT_PROMPTS.json")
    fi
    stage "rows_$ck" "$py" scripts/vad_grad3/batches.py "$ro" "$lab" "$dir/rows" "${extra[@]}" >> "$log" 2>&1 || exit 1
  fi
  keys=()
  for f in 1 2 3 4; do
    [[ -f "$dir/rows/ROWS_B8-f$f.jsonl" ]] || continue
    for w in "${views2[@]}"; do keys+=("B8-f$f@$w"); done
    keys+=("B8-f$f@geo")
  done
  echo "$ck keys ${keys[*]}" >> "$log"
  args=("$root" "$out/COMMAND_B200.txt" "${model[$ck]}" "$ro" "$dir/rows" "$g3" "$v" "$dir")
  if [[ "$pilot" == 1 && $first -eq 1 ]]; then
    forced_stop "vgrads_$ck" STOP_AFTER_MICRO "$py" $vv/vgrads.py "${args[@]}" "${keys[@]}" >> "$dir/vgrads.log" 2>&1
  fi
  stage "vgrads_$ck" "$py" $vv/vgrads.py "${args[@]}" "${keys[@]}" >> "$dir/vgrads.log" 2>&1 || exit 1
  memlog "vgrads_$ck" >> "$log" 2>&1
  if [[ ! -f "$dir/HELDOUT.done" ]]; then
    stage "heldout_$ck" "$py" $vv/heldout.py grad "$root" "$out/COMMAND_B200.txt" "${model[$ck]}" "$g3" "$held" \
      "$bench" "$dir" >> "$dir/heldout.log" 2>&1 || exit 1
    touch "$dir/HELDOUT.done"
    memlog "heldout_$ck" >> "$log" 2>&1
  fi
  [[ -f "$dir/GRAM_$ck.json" ]] || stage "gram_$ck" "$py" $vv/vgram.py gram "$dir" "$disp" "$adam" "$g3" "$ck" \
    >> "$dir/gram.log" 2>&1 || exit 1
  if [[ ! -f "$dir/linesearch/LINESEARCH_$ck.json" ]]; then
    stage "linesearch_$ck" env "${ls_env[@]}" "$py" $vv/linesearch.py "$root" "${model[$ck]}" "$g3" "$adam" "$dir" \
      "$held" "$bench" "$dir/HELDOUT_VALUES.jsonl" "$ck" "$dir/linesearch" >> "$dir/linesearch.log" 2>&1 || exit 1
    memlog "linesearch_$ck" >> "$log" 2>&1
  fi
  stage "remove_$ck" "$py" $vv/vgram.py remove "$dir" >> "$log" 2>&1 || exit 1
  first=0
done

stage stats2 "$py" $vv/stats2.py "$out" "$out/STATS_VIEW2.json" M0 A1_80 >> "$log" 2>&1 || exit 1
if [[ "$pilot" == 1 ]]; then
  "$py" $vv/vpilot.py project "$out" "$r053" "$out/PROJECTION.json" >> "$log" 2>&1 || exit 1
fi
rm -f "$disp"/D__*.pt
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) removed the displacement vectors; left: $(ls "$disp"/*.pt 2>/dev/null | wc -l)" \
  > "$out/DISPLACEMENT_REMOVED.txt"
memlog end >> "$log" 2>&1
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
