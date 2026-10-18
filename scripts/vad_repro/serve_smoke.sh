#!/bin/bash
# VLM-VAD-REPRO-001 R3 Part B task 4 (HANDOFF-VLM-VAD-REPRO-001-R3, gates G2 and G3), container VAD_A100_001.
# Usage: GPU_UUID=<uuid> JUDGE_API_BASE=<G1 endpoint> scripts/vad_repro/serve_smoke.sh <output_root>
# 1. Prints the per-tensor dtype counts of the safetensors header of the released model (header only).
# 2. Starts serve_model.sh for the base model as base_qwen35_4b and waits for /v1/models.
# 3. Runs infer.py, judge_qwenlm.py and cal_acc.py on the first 20 vstar records (seed 42, max tokens
#    8192, thinking off, the G1 judge), as scripts/evaluate_official.sh calls them. No repair step.
# 4. Stops the server. The record is <output_root>/SERVE_SMOKE.txt.
set -uo pipefail

out="$1"
root=/workspace/harness/VAD_Multimodal_OPD
eval_code="$root/evaluation/vision_opd_official"
base=/workspace/harness/data/vision-opd/Qwen3.5-4B
released=/workspace/harness/data/vad/VAD_for_Qwen3.5-4b/model.safetensors
judge_model=mistralai/Mistral-Small-3.2-24B-Instruct-2506
tag=base_qwen35_4b
smoke="$out/smoke"
mkdir -p "$smoke"
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1
log="$out/SERVE_SMOKE.txt"
status=0

{
  echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== JUDGE_API_BASE ${JUDGE_API_BASE:-<unset>}"
  echo "== safetensors header dtype counts: $released"
  python - "$released" <<'EOF'
import sys
from collections import Counter

from safetensors import safe_open

counts, numel = Counter(), Counter()
with safe_open(sys.argv[1], framework="pt") as handle:
    for key in handle.keys():
        tensor_slice = handle.get_slice(key)
        dtype = tensor_slice.get_dtype()
        n = 1
        for size in tensor_slice.get_shape():
            n *= size
        counts[dtype] += 1
        numel[dtype] += n
for dtype in sorted(counts):
    print(f"dtype {dtype} tensors {counts[dtype]} elements {numel[dtype]}")
print("tensors total", sum(counts.values()), "elements total", sum(numel.values()))
EOF
  echo "safetensors header exit $?"
} > "$log" 2>&1

# models <name>: print the /v1/models ids of port 8000; exit 0 when <name> is among them.
models() {
  python - "$1" 2>/dev/null <<'EOF'
import json
import sys
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
    ids = [m["id"] for m in json.load(response)["data"]]
print("models", ids)
sys.exit(0 if sys.argv[1] in ids else 1)
EOF
}
start_server() {
  setsid bash "$root/scripts/vad_repro/serve_model.sh" "$1" "$2" > "$3" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 240); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    models "$2" > /dev/null && return 0
    sleep 5
  done
  return 1
}
stop_server() {
  kill -TERM -- -"$server_pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || return 0; sleep 2; done
  kill -KILL -- -"$server_pid" 2>/dev/null
}

if [[ -n "$(models "")" ]]; then
  echo "port 8000 already answers; stop" >> "$log"
  exit 3
fi
t0=$(date +%s)
if start_server "$base" "$tag" "$out/model_server.log"; then
  {
    echo "== model server ready after $(( $(date +%s) - t0 )) s"
    grep -m 1 '^COMMAND' "$out/model_server.log"
    grep -m 1 '^GPU_UUID' "$out/model_server.log"
    models "$tag"
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
    python -c "import vllm; print('vllm', vllm.__version__)"
  } >> "$log" 2>&1

  python - "/workspace/harness/data/vad/benchmarks/vstar.json" "$smoke/vstar_20.json" <<'EOF'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))[:20]
json.dump(rows, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
EOF
  t1=$(date +%s)
  python "$eval_code/infer.py" --benchmark vstar --benchmark_json "$smoke/vstar_20.json" \
    --out_dir "$smoke/model_answer" --model_name "${tag}_seed42" --seed 42 \
    --api_base http://127.0.0.1:8000/v1 --api_key EMPTY --model_id "$tag" --max_tokens 8192 \
    --max_retries 3 --parallel_workers 64 --enable_thinking False > "$smoke/infer.log" 2>&1
  s_infer=$?
  t2=$(date +%s)
  (cd "$smoke" && python "$eval_code/judge_qwenlm.py" --benchmark vstar --model "${tag}_seed42" \
    --api_base "${JUDGE_API_BASE:-}" --api_key EMPTY --judge_model "$judge_model" --judge_max_tokens 2048) \
    > "$smoke/judge.log" 2>&1
  s_judge=$?
  t3=$(date +%s)
  python "$eval_code/cal_acc.py" --benchmark vstar --judge_json "$smoke/judge/vstar/${tag}_seed42_answer.jsonl" \
    --benchmark_json "$smoke/vstar_20.json" > "$smoke/score.log" 2>&1
  s_score=$?
  {
    echo "infer exit $s_infer wall_seconds $((t2 - t1))"
    echo "judge exit $s_judge wall_seconds $((t3 - t2))"
    echo "cal_acc exit $s_score"
    echo "== judge log (progress lines removed)"; grep -v 'it/s\]' "$smoke/judge.log" | tail -n 20
    echo "== score"; cat "$smoke/score.log"
    echo "== answers and labels"
    python - "$smoke" "$tag" "$eval_code" <<'EOF'
import json
import sys
from collections import Counter
from pathlib import Path

smoke, tag, eval_code = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, eval_code)
from repair_judges import canonical_label  # noqa: E402

answers = [json.loads(line) for line in open(smoke / "model_answer/vstar" / f"{tag}_seed42_answer.jsonl")]
judged = json.load(open(smoke / "judge/vstar" / f"{tag}_seed42_answer.jsonl"))
for row in judged:
    print(json.dumps({"uid": row.get("sample_uid"), "gt": row.get("response"),
                      "answer": str(row.get("model_answer"))[:300], "judge": row.get("judge"),
                      "source": row.get("judge_source")}, ensure_ascii=False))
labels = Counter(str(r.get("judge", "")) for r in judged)
print("answers", len(answers), "judged", len(judged))
print("labels raw", dict(labels))
print("labels in {Yes, No}", sum(v for k, v in labels.items() if k in ("Yes", "No")))
print("labels canonical (repair_judges.canonical_label, not applied)",
      dict(Counter(canonical_label(r.get("judge")) for r in judged)))
print("judge sources", dict(Counter(str(r.get("judge_source")) for r in judged)))
EOF
  } >> "$log" 2>&1
  [[ $s_infer -eq 0 && $s_judge -eq 0 && $s_score -eq 0 ]] || status=1
else
  { echo "model server not ready; log tail:"; tail -n 80 "$out/model_server.log"; } >> "$log"
  status=2
fi
stop_server
{ echo "== server stopped"; nvidia-smi --query-gpu=index,uuid,memory.used --format=csv; } >> "$log" 2>&1
exit "$status"
