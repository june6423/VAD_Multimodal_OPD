"""Tests of scripts/vad_view2/eqcompare.py (DIRECTIVE-VLM-VAD-VIEW-002-002 Decision 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vad_view2 import eqcompare  # noqa: E402


def _run(d: Path, pg: float, grad: np.ndarray, paths: list[str]) -> None:
    (d / "views_log").mkdir(parents=True)
    (d / "metrics_check_1.jsonl").write_text(json.dumps({"step": 1, "data": {"actor/pg_loss": pg, "actor/vopd_loss": pg}}) + "\n")
    grad.astype(np.float32).tofile(d / "grad.bin")
    (d / "grad.bin.json").write_text(json.dumps({"elements": int(grad.size), "params": [{"name": "w", "numel": int(grad.size)}]}))
    rows = [{"position": j, "uid": f"r{np.random.rand()}", "slot": j % 8 + 1, "path": p, "sha256": p[::-1]}
            for j, p in enumerate(paths)]
    (d / "views_log" / "teacher_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_eqcompare_pass_and_fail(tmp_path):
    g = np.linspace(-1.0, 1.0, 1000)
    paths = [f"/v/{j}.png" for j in range(16)]
    _run(tmp_path / "new" / "T8", 0.5, g, paths)
    _run(tmp_path / "old" / "T8", 0.5002, g * 1.01, paths)  # loss_rel 4e-4, cosine 1
    _run(tmp_path / "new" / "N1R", 0.5, g, paths)
    _run(tmp_path / "old" / "N1R", 0.5, g, paths[:-1] + ["/v/other.png"])  # one teacher image differs
    _run(tmp_path / "new" / "N1V8", 0.5, g, paths)
    _run(tmp_path / "old" / "N1V8", 0.502, g, paths)  # loss_rel 4e-3
    sys.argv = ["eqcompare.py", str(tmp_path / "new"), str(tmp_path / "old"), str(tmp_path / "EQ.json"), "T8", "N1R", "N1V8"]
    assert eqcompare.main() == 1
    res = json.loads((tmp_path / "EQ.json").read_text())
    assert res["arms"]["T8"]["pass"] is True
    assert abs(res["arms"]["T8"]["gradient"]["cosine"] - 1.0) < 1e-9
    assert res["arms"]["N1R"]["pass"] is False and res["arms"]["N1R"]["teacher_rows_equal"] is False
    assert res["arms"]["N1V8"]["pass"] is False
    sys.argv = sys.argv[:4] + ["T8"]
    assert eqcompare.main() == 0
