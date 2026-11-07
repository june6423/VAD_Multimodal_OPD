"""Tests of the split validation of DIRECTIVE-VLM-VAD-PACKFIX-001-008: val_split.py, val_concat.py and valtrim.py."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.vad_packfix import val_split, valtrim

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "vad_packfix"


def run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True)


def test_ranges_cover_all_rows_in_order():
    spans = val_split.ranges(2636, 4)
    assert spans == [(0, 659), (659, 1318), (1318, 1977), (1977, 2636)]
    spans = val_split.ranges(10, 4)
    assert [b - a for a, b in spans] == [3, 3, 2, 2] and spans[-1][1] == 10


def test_split_then_concat_gives_the_original_row_order(tmp_path):
    src = tmp_path / "val.parquet"
    pd.DataFrame({"q": [f"q{i}" for i in range(10)], "data_source": ["a"] * 10}).to_parquet(src)
    assert run("val_split.py", src, tmp_path / "split", 4).returncode == 0
    split = json.loads((tmp_path / "split" / "VAL_SPLIT.json").read_text())
    assert split["rows"] == 10 and len(split["parts"]) == 4
    dumps = []
    for k in range(4):
        part = pd.read_parquet(tmp_path / "split" / f"part_{k}.parquet")
        dump = tmp_path / f"d{k}.jsonl"  # a val_only dump: one row for each input row, in input order
        dump.write_text("".join(json.dumps({"input": q, "gts": "A", "score": 1.0}) + "\n" for q in part["q"]))
        dumps.append(dump)
    r = run("val_concat.py", tmp_path / "split" / "VAL_SPLIT.json", tmp_path / "0.jsonl", *dumps)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [json.loads(line) for line in (tmp_path / "0.jsonl").read_text().splitlines()]
    assert [x["input"] for x in rows] == [f"q{i}" for i in range(10)]


def test_concat_refuses_a_short_part_dump(tmp_path):
    split = tmp_path / "VAL_SPLIT.json"
    split.write_text(json.dumps({"rows": 3, "parts": [{"part": 0, "start": 0, "end": 2}, {"part": 1, "start": 2, "end": 3}]}))
    (tmp_path / "a.jsonl").write_text('{"input": "q0"}\n')
    (tmp_path / "b.jsonl").write_text('{"input": "q2"}\n')
    r = run("val_concat.py", split, tmp_path / "0.jsonl", tmp_path / "a.jsonl", tmp_path / "b.jsonl")
    assert r.returncode == 1 and not (tmp_path / "0.jsonl").exists()


def test_valtrim_trims_after_each_call_only_with_the_flag(monkeypatch):
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    async def orig(self, batch):
        return ("out", len(batch))

    calls = []
    monkeypatch.setattr(AgentLoopWorker, "generate_sequences", orig)
    monkeypatch.setattr(valtrim.hostmem, "setup", lambda: calls.append("hostmem"))
    monkeypatch.setattr(valtrim, "trim", lambda: calls.append("trim"))
    monkeypatch.delenv("VAD_VALTRIM", raising=False)
    valtrim.setup()
    assert AgentLoopWorker.generate_sequences is orig and calls == ["hostmem"]
    monkeypatch.setenv("VAD_VALTRIM", "1")
    valtrim.setup()
    assert asyncio.run(AgentLoopWorker.generate_sequences(None, [1, 2, 3])) == ("out", 3)
    assert calls == ["hostmem", "hostmem", "trim"]
