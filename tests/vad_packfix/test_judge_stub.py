"""Tests of the training judge stub (DIRECTIVE-VLM-VAD-PACKFIX-001-007, TASK-VLM-VAD-PACKFIX-001-024 failure).

train_vad.sh runs check_judge_endpoint before the training. A closed port makes curl exit 7, so the training cannot
start without a judge. The stub answers the model list and refuses each other request (HTTP 503) with a log line.
"""

import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"


def check_endpoint(api_base):
    """Run the check_judge_endpoint function of scripts/train_vad.sh unchanged."""
    text = (ROOT / "scripts/train_vad.sh").read_text()
    func = re.search(r"^check_judge_endpoint\(\) \{\n.*?^\}\n", text, re.S | re.M).group(0)
    script = f"{func}\nBENCHMARK_EVAL_JUDGE_API_KEY=EMPTY\nBENCHMARK_EVAL_JUDGE_API_BASE={api_base}\n" \
             f"BENCHMARK_EVAL_JUDGE_MODEL={MODEL}\ncheck_judge_endpoint\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_stub(tmp_path):
    port = free_port()
    log = tmp_path / "stub.log"
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/vad_packfix/judge_stub.py"), str(port), MODEL,
                             str(log)])
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    return proc, port, log


def test_closed_port_fails_the_endpoint_check():
    assert check_endpoint("http://127.0.0.1:9/v1") != 0


def test_stub_passes_the_endpoint_check_and_refuses_other_requests(tmp_path):
    proc, port, log = start_stub(tmp_path)
    try:
        assert check_endpoint(f"http://127.0.0.1:{port}/v1") == 0
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                                     data=json.dumps({"model": MODEL}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("the stub answered a judge call")
        except urllib.error.HTTPError as err:
            assert err.code == 503
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    lines = log.read_text().splitlines()
    assert [line.split()[:2] for line in lines] == [["GET", "/v1/models"], ["POST", "/v1/chat/completions"]]
    assert lines[1].split()[2] == "refused"
