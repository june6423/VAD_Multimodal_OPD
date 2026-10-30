#!/bin/bash
# VLM-VAD-ROLLOUT-001 rescore of one official evaluation run under the R8 judge rule
# (HANDOFF-VLM-VAD-ROLLOUT-001-R8, Part A item 3, gate G3). Usage: DATA_ROOT=<data root> ENV_ROOT=<environment root> \
#   scripts/vad_rollout/rescore_run.sh <source run dir> <arm tag> <output_root>
# <source run dir> is the RUN_DIR of scripts/evaluate_official.sh and <arm tag> its MODEL_TAG (for example
# released_step80). The script copies the answer file and the judge file of each of the six benchmarks into
# <output_root>/run, then runs there repair_judges.py, audit_results.py and cal_acc.py per benchmark and then
# summarize_results.py, with the arguments of scripts/evaluate_official.sh. It makes no judge call and no
# inference, and it stops at the first non-zero exit. <output_root> gets:
#   SOURCE_SHA256_BEFORE.txt, SOURCE_SHA256_AFTER.txt  SHA-256 of every file under <source run dir>
#   RESCORE.txt  stage exit codes, the repair counts, the score lines, per benchmark the judge_source counts and
#                the sample_uid and raw label of every row that the rule counts as incorrect, summary.md
#   run/summary.json, run/summary.md, run/audits/<benchmark>.json, run/logs/<benchmark>.{repair,audit,score}.log
set -uo pipefail

src="$1"
tag="$2"
out="$3"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data="${DATA_ROOT:-$HOME/vad-rollout/data}"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
code_dir="$root/evaluation/vision_opd_official"
benchmarks=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
declare -A json_name=([vstar]=vstar.json [zoombench]=zoombench.json [hrbench-4k]=hr_bench_4k.json
  [hrbench-8k]=hr_bench_8k.json [mme-realworld]=MME_RealWorld.json [mme-realworld-cn]=MME_RealWorld_CN.json)
export PATH="$env_root/vad/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run="$out/run"
log="$out/RESCORE.txt"
name="${tag}_seed42_answer.jsonl"
mkdir -p "$run/audits" "$run/logs"

source_sha() { (cd "$src" && find . -type f | sort | xargs -d '\n' sha256sum) > "$1"; }
finish() {
  source_sha "$out/SOURCE_SHA256_AFTER.txt"
  if cmp -s "$out/SOURCE_SHA256_BEFORE.txt" "$out/SOURCE_SHA256_AFTER.txt"; then
    echo "source files $(wc -l < "$out/SOURCE_SHA256_AFTER.txt") sha256 unchanged" >> "$log"
  else
    echo "source files sha256 CHANGED" >> "$log"
  fi
  echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $1" >> "$log"
  exit "$1"
}
# stage <label> <log file> <command...>: run the command with its output in <log file>, then copy the log and
# the exit code into RESCORE.txt. Stop at a non-zero exit.
stage() {
  local label="$1" file="$2" code
  shift 2
  "$@" > "$file" 2>&1
  code=$?
  { echo "-- $label exit $code"; cat "$file"; } >> "$log"
  [[ $code -eq 0 ]] || finish "$code"
}

{
  echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== git HEAD $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "== source $src tag $tag output $out"
} >> "$log"
source_sha "$out/SOURCE_SHA256_BEFORE.txt"

for b in "${benchmarks[@]}"; do
  bench_json="$data/vad/benchmarks/${json_name[$b]}"
  answer="$run/model_answer/$b/$name"
  judge="$run/judge/$b/$name"
  echo "== $b" >> "$log"
  mkdir -p "$(dirname "$answer")" "$(dirname "$judge")"
  stage "copy answer" "$run/logs/$b.copy.log" cp "$src/model_answer/$b/$name" "$answer"
  stage "copy judge" "$run/logs/$b.copy.log" cp "$src/judge/$b/$name" "$judge"
  stage repair "$run/logs/$b.repair.log" python3 "$code_dir/repair_judges.py" --judge-json "$judge"
  stage audit "$run/logs/$b.audit.log" python3 "$code_dir/audit_results.py" --benchmark "$b" \
    --benchmark-json "$bench_json" --answer-json "$answer" --judge-json "$judge" --answer-source reused \
    --answer-source-path "$src/model_answer/$b/$name" --output "$run/audits/$b.json"
  stage score "$run/logs/$b.score.log" python3 "$code_dir/cal_acc.py" --benchmark "$b" --judge_json "$judge" \
    --benchmark_json "$bench_json"
  python3 - "$judge" >> "$log" 2>&1 <<'EOF'
import collections
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
sources = collections.Counter(str(row.get("judge_source", "<none>")) for row in rows)
print("rows", len(rows), "yes", sum(row["judge"] == "Yes" for row in rows), "judge_source", dict(sorted(sources.items())))
for row in rows:
    if row.get("judge_source") == "noncanonical_as_incorrect":
        print("as_incorrect", row.get("sample_uid"), json.dumps(row.get("judge_raw_before_repair"), ensure_ascii=False))
EOF
done

echo "== summary" >> "$log"
stage summarize "$run/logs/summarize.log" python3 "$code_dir/summarize_results.py" --run-dir "$run" \
  --benchmarks "$(IFS=,; echo "${benchmarks[*]}")" --model-tag "$tag" --model-id "$tag" \
  --judge-api-base "${JUDGE_API_BASE:-http://127.0.0.1:8900/v1}" \
  --judge-model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --judge-max-tokens 2048 --seed 42 \
  --max-tokens 8192 --enable-thinking False --evaluator-commit c2e345f
finish 0
