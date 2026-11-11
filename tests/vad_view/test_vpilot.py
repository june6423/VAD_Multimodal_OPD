"""vpilot.py selection and projection helpers; stage1.py resume helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import stage1  # noqa: E402
import vpilot  # noqa: E402


def test_pick_fresh():
    rows = [{"row": i, "bbox": [0, 0, 10 + i % 50, 10 + i % 30]} for i in range(300)]
    rows[17]["bbox"] = [0, 0, 500, 500]
    rows[230]["bbox"] = [5, 5, 6, 6]
    tokens = {i: 100 + (i * 7) % 90 for i in range(300)}
    tokens[42] = 5000
    got = vpilot.pick_fresh(rows, tokens)
    assert got[:3] == [17, 230, 42]
    assert len(got) == 40 and len(set(got)) == 40
    assert got == vpilot.pick_fresh(rows, tokens)


def test_pick_fresh_overlap():
    rows = [{"row": i, "bbox": [0, 0, 10, 10]} for i in range(100)]
    rows[3]["bbox"] = [0, 0, 90, 90]
    rows[4]["bbox"] = [0, 0, 1, 1]
    tokens = {i: 10 for i in range(100)}
    tokens[3] = 99  # the largest box is also the longest prompt
    got = vpilot.pick_fresh(rows, tokens)
    assert got[:2] == [3, 4] and len(got) == 2 + vpilot.RANDOM_ROWS and len(set(got)) == len(got)


def test_pick_prefix():
    a = vpilot.pick_prefix(list(range(3000)), 0)
    b = vpilot.pick_prefix(list(range(3000)), 80)
    assert len(a) == 20 and len(set(a)) == 20 and a == sorted(a) and a != b


def test_stage_seconds_and_scaled(tmp_path):
    (tmp_path / "TIMES.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (
        {"stage": "stop_fresh", "seconds": 30, "exit": 75}, {"stage": "fresh", "seconds": 50, "exit": 0},
        {"stage": "gram_M0", "seconds": 10, "exit": 0})))
    sec = vpilot.stage_seconds(tmp_path)
    assert sec == {"fresh": 80, "gram_M0": 10}
    s = vpilot.scaled(80, 20, 1000)
    assert s["fixed"] == 60 and s["full"] == 1060
    assert vpilot.scaled(10, 15, 5)["fixed"] == 0.0


def test_stage1_resume_helpers(tmp_path, monkeypatch):
    path = tmp_path / "FRESH.jsonl"
    assert stage1.done_keys(path, "row") == set()
    path.write_text(json.dumps({"row": 4}) + "\n" + json.dumps({"row": 9}) + "\n")
    assert stage1.done_keys(path, "row") == {4, 9}
    monkeypatch.setenv("STOP_AFTER", "2")
    tick = stage1.stopper()
    tick()
    with pytest.raises(SystemExit) as e:
        tick()
    assert e.value.code == stage1.STOP_EXIT
    monkeypatch.setenv("STOP_AFTER", "0")
    tick = stage1.stopper()
    for _ in range(5):
        tick()
