#!/bin/bash
# VLM-VAD-ROLLOUT-001 judge check (HANDOFF-VLM-VAD-ROLLOUT-001-R4, Part A item 2, gate G3). No GPU work.
# Usage: JUDGE_API_BASE=<judge endpoint> ENV_ROOT=<environment root> scripts/vad_rollout/judge_check.sh <output_root>
# The judge_check.sh of 1f12b81d (scripts/vad_repro/) with the endpoint from JUDGE_API_BASE and the python of
# the VAD environment. Waits up to 3600 s until /v1/models lists the served name, then sends one chat request
# with the judge prompt of evaluation/vision_opd_official/judge_qwenlm.py (temperature 0, max tokens 2048)
# and prints the reply. The record is <output_root>/JUDGE_CHECK.txt.
set -uo pipefail

out="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="${ENV_ROOT:-$HOME/vad-rollout/envs}"
endpoint="${JUDGE_API_BASE:-http://127.0.0.1:8900/v1}"
mkdir -p "$out"
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1

# scripts/evaluate_official.sh and train_vad.sh call curl for their endpoint checks.
echo "curl: $(command -v curl || echo missing)" > "$out/TOOLS.txt"
"$env_root/vad/bin/python" - "$endpoint" "$root/evaluation/vision_opd_official" > "$out/JUDGE_CHECK.txt" 2>&1 <<'EOF'
import json
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen

endpoint, eval_dir = sys.argv[1].rstrip("/"), sys.argv[2]
sys.path.insert(0, eval_dir)
from judge_qwenlm import PROMPT_TEMPLATE  # noqa: E402
from openai import OpenAI  # noqa: E402

name = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
print("date", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
print("endpoint", endpoint)
ids = []
deadline = time.time() + 3600
while time.time() < deadline:
    try:
        with urlopen(f"{endpoint}/models", timeout=10) as response:
            ids = [m["id"] for m in json.load(response)["data"]]
        if name in ids:
            break
    except Exception as exc:  # the judge may still be loading
        ids = [f"unreachable {type(exc).__name__}"]
    time.sleep(10)
print("models", ids, "at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
if name not in ids:
    print("RESULT endpoint does not list", name)
    sys.exit(1)

prompt = PROMPT_TEMPLATE.format(
    question="What is the color of the bus? (A) red (B) blue (C) green (D) white",
    gt="A",
    response="The bus in the image is red, so the answer is (A).",
)
print("prompt", json.dumps(prompt))
reply = OpenAI(api_key="EMPTY", base_url=endpoint, timeout=600).chat.completions.create(
    model=name, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=2048)
text = (reply.choices[0].message.content or "").strip()
print("reply", json.dumps(text))
print("usage", reply.usage)
ok = text.startswith("Yes") or text.startswith("No")
print("RESULT reply starts with Yes or No:", ok)
sys.exit(0 if ok else 1)
EOF
