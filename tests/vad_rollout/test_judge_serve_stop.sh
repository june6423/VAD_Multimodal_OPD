#!/bin/bash
# VLM-VAD-ROLLOUT-001 gate G3: scripts/vad_rollout/judge_serve.sh exits 0 within 60 s after <output_root>/STOP
# appears, and the server process is gone. A stand-in "vllm" serves /v1/models on a local port with the
# system python3. No GPU. Usage: bash tests/vad_rollout/test_judge_serve_stop.sh [work dir]
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${1:-$(mktemp -d)}"
fake_env="$work/fake-judge-env"
out="$work/judge"
port="${JUDGE_TEST_PORT:-18900}"
mkdir -p "$fake_env/bin" "$work/model"
ln -sf "$(command -v python3)" "$fake_env/bin/python"

# Stand-in for "vllm serve <model> --served-model-name <name> --port <port> ...": answers /v1/models.
cat > "$fake_env/bin/vllm" <<'EOF'
#!/bin/bash
here="$(dirname "$0")"
exec "$here/python" - "$@" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
name = args[args.index("--served-model-name") + 1]
port = int(args[args.index("--port") + 1])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": name}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


print("stand-in vllm serving", name, "on", port, flush=True)
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
EOF
chmod +x "$fake_env/bin/vllm"

CUDA_VISIBLE_DEVICES=7 JUDGE_MODEL_DIR="$work/model" JUDGE_PYTHON="$fake_env/bin/python" JUDGE_PORT="$port" \
  bash "$root/scripts/vad_rollout/judge_serve.sh" "$out" &
script_pid=$!

for _ in $(seq 1 60); do [[ -f "$out/READY.json" ]] && break; sleep 1; done
if [[ ! -f "$out/READY.json" ]]; then
  echo "FAIL: READY.json missing"; cat "$out/JUDGE_SERVE.txt"; kill "$script_pid"; exit 1
fi
server_pid=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["pid"])' "$out/READY.json")
echo "READY.json written; stand-in server pid $server_pid alive: $(kill -0 "$server_pid" 2>/dev/null && echo yes || echo no)"

touch "$out/STOP"
stop_time=$(date +%s)
echo "STOP created at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait "$script_pid"
code=$?
elapsed=$(( $(date +%s) - stop_time ))
if kill -0 "$server_pid" 2>/dev/null; then gone=no; else gone=yes; fi
echo "judge_serve.sh exit $code after ${elapsed} s; stand-in server gone: $gone"
echo "== JUDGE_SERVE.txt tail"; tail -n 4 "$out/JUDGE_SERVE.txt"
if [[ $code -eq 0 && $elapsed -le 60 && $gone == yes ]]; then echo "G3 PASS"; exit 0; fi
echo "G3 FAIL"
exit 1
