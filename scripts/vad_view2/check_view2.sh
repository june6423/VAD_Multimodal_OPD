#!/bin/bash
# Part B item 1 of HANDOFF-VLM-VAD-VIEW-002-R1: the environment check, the input hashes, the image check (G2) and the
# loss check (G3). Steps:
#   1. ENV_CHECK.txt: git HEAD, versions of python, torch, transformers and vllm, nvidia-smi, container memory.max,
#      free disk. INPUTS.sha256 (scripts/vad_packfix/inputs.sh) and INPUTS_VIEW2.sha256 (the VIEW-001 views
#      manifest and P1_FIT.json, HELDOUT.json, the step-0 validation dump of the PACKFIX FULL task, train.jsonl).
#   2. make_viewmap.sh -> viewmap/VIEWMAP.json (VIEW-001 views reused only when they match).
#   3. G3: two check runs of run_view2.sh (CHECK=1: step 1 from M0, training only) with the Ray hook g3hook.py:
#      g3/view2 = this code with ARM=B; g3/packfix = the code of exp/VLM-VAD-PACKFIX-001 (git archive of
#      PACKFIX_COMMIT into packfix_src/, VAD_PACKFIX=0, the same settings). The first run generates the 768 rollout
#      rows of step 1 and saves them (replay.pkl); the second run replays them. g3compare.py -> G3_CHECK.json.
#   4. G2: check runs of T8, N1V8 and N1R (step 1, replay of the same rollouts; N1V8 and N1R take every 8th row),
#      g2/B = g3/view2. g2check.py -> G2_CHECK.json (also the N1R slot counts of steps 1 to 40).
#   The checkpoints of the check runs and the gradient files are removed at the end (the JSON summaries stay).
# Exit 0 when G2 and G3 pass, else 1.
# Usage: B200_ROOT=<root> ENV_ROOT=<env root> CUDA_VISIBLE_DEVICES=0 [ROLLOUT_GPU_UTIL] \
#   scripts/vad_view2/check_view2.sh <output_root>
set -uo pipefail
out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
b200="${B200_ROOT:?}" env_root="${ENV_ROOT:?}"
py="$env_root/vad/bin/python"
data="$b200/harness/data"
view1="$b200/harness/outputs/VLM-VAD-VIEW-001/TASK-VLM-VAD-VIEW-001-006"
packfix_b="$b200/harness/outputs/VLM-VAD-PACKFIX-001/TASK-VLM-VAD-PACKFIX-001-026/B"
packfix_commit=eb933d4771d52fce283741d2352bb568a9fdf453
mkdir -p "$out"
cd "$root" || exit 1
log="$out/CHECK_VIEW2.txt"
echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ) git HEAD $(git -c safe.directory="$root" rev-parse HEAD)" >> "$log"

cg="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
while [[ "$cg" != /sys/fs/cgroup && ( ! -r "$cg/memory.max" || "$(cat "$cg/memory.max")" == max ) ]]; do
  cg="$(dirname "$cg")"
done
export VAD_CONTAINER_CGROUP="$cg"
{
  echo "git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  "$py" --version
  "$py" -c 'import torch, transformers, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "vllm", vllm.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv
  echo "container cgroup $cg memory.max $(cat "$cg/memory.max") oom_kill $(awk '$1=="oom_kill"{print $2}' "$cg/memory.events")"
  echo "orphan ray and vllm processes: $(ps -eo pid,ppid,args | awk '$2==1 && $3 ~ /^(ray|VLLM)::/' | wc -l)"
  df -B1 "$out"
} > "$out/ENV_CHECK.txt" 2>&1
code=$?
echo "env check exit $code" >> "$log"
B200_ROOT="$b200" bash scripts/vad_packfix/inputs.sh "$out/INPUTS.sha256" >> "$log" 2>&1 || exit 1
sha256sum "$view1/views/MANIFEST.jsonl" "$view1/views/P1_FIT.json" "$view1/HELDOUT.json" \
  "$packfix_b/validation/0.jsonl" "$data/hub/Vision-OPD-6K/train.jsonl" > "$out/INPUTS_VIEW2.sha256" || exit 1
cat "$out/INPUTS_VIEW2.sha256" >> "$log"

bash scripts/vad_view2/make_viewmap.sh "$out/viewmap" >> "$log" 2>&1 || { echo "make_viewmap.sh failed" >> "$log"; exit 1; }
export VAD_VIEW_MAP="$out/viewmap/VIEWMAP.json"
echo "VIEWMAP.json sha256 $(sha256sum "$VAD_VIEW_MAP" | cut -d' ' -f1)" >> "$log"

g3py="$out/g3py"
mkdir -p "$g3py"
cp -f scripts/vad_view2/g3hook.py "$g3py/vad_view2_g3.py"
src="$out/packfix_src"
if [[ ! -f "$src/ARCHIVE_DONE" ]]; then
  rm -rf "$src"
  mkdir -p "$src"
  git -c safe.directory="$root" archive "$packfix_commit" | tar -x -C "$src" || exit 1
  echo "$packfix_commit" > "$src/ARCHIVE_DONE"
fi
echo "PACKFIX code $packfix_commit in $src: ray_trainer.py sha256 $(sha256sum "$src/verl/trainer/ppo/ray_trainer.py" | cut -d' ' -f1)" >> "$log"
export CHECK=1 WORKER_HOOK=vad_view2_g3.setup G3_PY="$g3py" VAD_G3_REPLAY="$out/replay.pkl" HELDOUT_JSON=-

# run <arm> <dir> [code root] [gradient file]
run() {
  local arm="$1" dir="$2" code_root="${3:-}" grad="${4:-}" start code
  start=$(date +%s)
  ARM="$arm" CODE_ROOT="$code_root" VAD_G3_GRAD="$grad" bash scripts/vad_view2/run_view2.sh "$dir"
  code=$?
  echo "check run $arm ${code_root:-view2} exit $code wall_seconds $(( $(date +%s) - start )) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
  return "$code"
}
[[ -f "$out/g3/view2/grad.bin.json" ]] || run B "$out/g3/view2" "" "$out/g3/view2/grad.bin" || exit 1
echo "replay.pkl sha256 $(sha256sum "$out/replay.pkl" | cut -d' ' -f1) bytes $(stat -c %s "$out/replay.pkl")" >> "$log"
[[ -f "$out/g3/packfix/grad.bin.json" ]] || run B "$out/g3/packfix" "$src" "$out/g3/packfix/grad.bin" || exit 1
"$py" scripts/vad_view2/g3compare.py "$out/g3/view2" "$out/g3/packfix" "$out/G3_CHECK.json" >> "$log" 2>&1
g3=$?
echo "g3compare.py exit $g3" >> "$log"

mkdir -p "$out/g2"
ln -sfn ../g3/view2 "$out/g2/B"
for arm in T8 N1V8 N1R; do
  [[ -f "$out/g2/$arm/views_log/teacher_1.jsonl" ]] || run "$arm" "$out/g2/$arm" || exit 1
done
"$py" scripts/vad_view2/g2check.py "$out/g2" "$VAD_VIEW_MAP" "$out/G2_CHECK.json" >> "$log" 2>&1
g2=$?
echo "g2check.py exit $g2" >> "$log"

for d in "$out/g3/view2" "$out/g3/packfix" "$out/g2/T8" "$out/g2/N1V8" "$out/g2/N1R"; do
  echo "removed: $(du -sh "$d/checkpoints" 2>/dev/null | cut -f1) checkpoints of $d" >> "$log"
  rm -rf "$d/checkpoints"
  rm -f "$d/grad.bin"
done
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ) G3 exit $g3 G2 exit $g2" >> "$log"
[[ $g3 -eq 0 && $g2 -eq 0 ]]
