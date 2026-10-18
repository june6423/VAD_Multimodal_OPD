#!/bin/bash
# VLM-VAD-REPRO-001 R3 Part B task 3 (HANDOFF-VLM-VAD-REPRO-001-R3, gate G1), container VAD_A100_001.
# Usage: scripts/vad_repro/judge_check.sh <output_root> <judge READY.json>
# Tries each judge address in READY.json and 127.0.0.1. For the first address whose /v1/models lists
# the served name, it sends one chat request with the judge prompt of
# evaluation/vision_opd_official/judge_qwenlm.py (temperature 0, max tokens 2048) and prints the reply.
# Records: <output_root>/JUDGE_CHECK.txt and <output_root>/ENDPOINT.txt (the endpoint URL).
set -uo pipefail

out="$1"
ready="$2"
root=/workspace/harness/VAD_Multimodal_OPD
mkdir -p "$out"
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1

# scripts/evaluate_official.sh calls curl for its endpoint checks.
echo "curl: $(command -v curl || echo missing)" > "$out/TOOLS.txt"
# The judge task writes READY.json when its server answers. Wait up to 3600 s. Stop early when the
# judge record in the same folder reports a failed start.
for _ in $(seq 1 360); do
  [[ -f "$ready" ]] && break
  grep -q "judge not ready" "$(dirname "$ready")/JUDGE_SERVE.txt" 2>/dev/null && break
  sleep 10
done
if [[ ! -f "$ready" ]]; then
  { echo "READY.json missing: $ready"; tail -n 40 "$(dirname "$ready")/JUDGE_SERVE.txt" 2>&1; } > "$out/JUDGE_CHECK.txt"
  exit 1
fi
python - "$ready" "$out" "$root/evaluation/vision_opd_official" > "$out/JUDGE_CHECK.txt" 2>&1 <<'EOF'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ready_path, out, eval_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, eval_dir)
from judge_qwenlm import PROMPT_TEMPLATE  # noqa: E402
from openai import OpenAI  # noqa: E402

ready = json.loads(ready_path.read_text())
print("date", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
print("READY.json", json.dumps(ready))
name, port = ready["served_name"], ready["port"]
endpoint = None
for host in [*ready["ips"], "127.0.0.1"]:
    base = f"http://{host}:{port}/v1"
    try:
        with urlopen(f"{base}/models", timeout=10) as response:
            models = json.load(response)
        ids = [m["id"] for m in models["data"]]
        print("models", base, ids)
        if name in ids:
            endpoint = base
            break
    except Exception as exc:  # report each unreachable address
        print("unreachable", base, type(exc).__name__, exc)
if endpoint is None:
    print("RESULT no endpoint lists", name)
    sys.exit(1)
(out / "ENDPOINT.txt").write_text(endpoint + "\n")
print("endpoint", endpoint)

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
