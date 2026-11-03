"""DIRECTIVE-VLM-VAD-GRAD-001-003 Decisions 4 to 6: with PYTHONPATH=<repo root> (as in g4.sh, label.sh, run_grad.sh),
verl imports from the repo root, and provenance.verl_origin stops when verl is not under the given root."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "vad_grad" / "provenance.py"


def run(root: Path, cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, str(SCRIPT), str(root)], cwd=cwd, env=env, capture_output=True, text=True)


def test_verl_imports_from_repo_root(tmp_path):
    pytest.importorskip("torch")
    res = run(ROOT, tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    rec = json.loads(res.stdout.strip().splitlines()[-1])
    assert Path(rec["verl_file"]).is_relative_to(ROOT)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert rec["verl_git_head"] == head


def test_stops_when_verl_is_not_under_root(tmp_path):
    pytest.importorskip("torch")
    res = run(tmp_path, tmp_path)
    assert res.returncode != 0
    assert "STOP" in res.stderr
