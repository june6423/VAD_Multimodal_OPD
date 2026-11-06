"""Judge stub for the training processes of run_packfix.sh (DIRECTIVE-VLM-VAD-PACKFIX-001-007).

train_vad.sh checks that the judge endpoint lists the judge model before the training starts. The training calls no
judge (ray_trainer.py 654-667 and 2792-2793). This CPU stub answers GET <base>/models with the model name and answers
each other request with HTTP 503, so an unexpected judge call fails. Each request is one line in the log:
"<method> <path> served|refused".
Usage: python judge_stub.py <port> <model> <log file>
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, model, log_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]


class Handler(BaseHTTPRequestHandler):
    def _log(self, result):
        with open(log_path, "a") as f:
            f.write(f"{self.command} {self.path} {result}\n")

    def _reply(self):
        if self.command == "GET" and self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list", "data": [{"id": model, "object": "model"}]}).encode()
            self.send_response(200)
            self._log("served")
        else:
            body = b'{"error": "judge stub: the training must not call the judge"}'
            self.send_response(503)
            self._log("refused")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
