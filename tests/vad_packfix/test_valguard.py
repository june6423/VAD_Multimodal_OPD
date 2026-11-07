"""Tests of DIRECTIVE-VLM-VAD-PACKFIX-001-009: valguard.sh (actions 3 and 4), val_batches.py and the batch log of
valtrim.py (action 6), and the memwatch interval during validation (action 5)."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.vad_packfix import valtrim

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "vad_packfix"


def guard(tmp_path, limit, *cmd, oom_after=None):
    cg = tmp_path / "cg"
    cg.mkdir(exist_ok=True)
    (cg / "memory.events").write_text("oom 0\noom_kill 3\n")
    console = tmp_path / "console.log"
    if oom_after is not None:
        def bump():
            time.sleep(oom_after)
            (cg / "memory.events").write_text("oom 1\noom_kill 4\n")
        threading.Thread(target=bump, daemon=True).start()
    start = time.time()
    with console.open("w") as f:
        r = subprocess.run(["bash", str(SCRIPTS / "valguard.sh"), str(limit), str(console), str(cg), *cmd],
                           stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - start, console.read_text()


def test_guard_passes_the_exit_code(tmp_path):
    code, _, _ = guard(tmp_path, 60, "bash", "-c", "exit 3")
    assert code == 3
    code, secs, _ = guard(tmp_path, 60, "true")
    assert code == 0 and secs < 10


def test_guard_stops_at_the_wall_time_limit(tmp_path):
    code, secs, _ = guard(tmp_path, 2, "sleep", "60")
    assert code == 124 and secs < 15


def test_guard_stops_when_a_worker_dies(tmp_path):
    code, secs, out = guard(tmp_path, 120, "bash", "-c",
                            "sleep 1; echo '(raylet) A worker died or was killed while executing a task'; sleep 60")
    assert code == 125 and secs < 20
    assert "VALGUARD stop: worker death line" in out


def test_guard_stops_on_a_container_oom_kill(tmp_path):
    code, secs, out = guard(tmp_path, 120, "sleep", "60", oom_after=1)
    assert code == 125 and secs < 20
    assert "VALGUARD stop: oom_kill 3 -> 4" in out


def test_val_batches_gives_row_range_and_pixels_of_each_batch(tmp_path):
    paths = []
    for i in range(10):
        p = tmp_path / f"im{i}.png"
        Image.new("RGB", (10 + i, 20)).save(p)
        paths.append(p)
    split = tmp_path / "split"
    split.mkdir()
    # Two parts: global rows 0-5 and 6-9. Row 3 has two images.
    cells = [[{"path": str(paths[i])}] for i in range(10)]
    cells[3] = [{"path": str(paths[3])}, {"path": str(paths[0])}]
    pd.DataFrame({"images": cells[:6]}).to_parquet(split / "part_0.parquet")
    pd.DataFrame({"images": cells[6:]}).to_parquet(split / "part_1.parquet")
    (split / "VAL_SPLIT.json").write_text(json.dumps(
        {"rows": 10, "parts": [{"part": 0, "start": 0, "end": 6}, {"part": 1, "start": 6, "end": 10}]}))
    r = subprocess.run([sys.executable, str(SCRIPTS / "val_batches.py"), str(split), "4", str(tmp_path / "B.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    b = json.loads((tmp_path / "B.json").read_text())
    assert b["val_batch_size"] == 4
    got = [(x["part"], x["batch"], x["row_start"], x["row_end"], x["images"]) for x in b["batches"]]
    assert got == [(0, 0, 0, 4, 5), (0, 1, 4, 6, 2), (1, 0, 6, 10, 4)]
    px = [sum((10 + i) * 20 for i in range(4)) + 10 * 20, sum((10 + i) * 20 for i in (4, 5)),
          sum((10 + i) * 20 for i in range(6, 10))]
    assert [x["pixels"] for x in b["batches"]] == px
    assert "VAL_BATCH part 1 batch 0 rows 6-10 images 4" in r.stdout


def test_valtrim_logs_each_manager_batch_with_the_flag(monkeypatch, capsys):
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker

    def orig(self, prompts):
        return ("out", len(prompts))

    async def worker_orig(self, batch):
        return batch

    monkeypatch.setattr(AgentLoopManager, "generate_sequences", orig)
    monkeypatch.setattr(AgentLoopWorker, "generate_sequences", worker_orig)
    monkeypatch.setattr(valtrim.hostmem, "setup", lambda: None)
    monkeypatch.delenv("VAD_VALTRIM", raising=False)
    valtrim.setup()
    assert AgentLoopManager.generate_sequences is orig
    monkeypatch.setenv("VAD_VALTRIM", "1")
    valtrim.setup()
    assert AgentLoopManager.generate_sequences(None, [1] * 32) == ("out", 32)
    assert AgentLoopManager.generate_sequences(None, [1] * 24) == ("out", 24)
    out = capsys.readouterr().out
    assert "VAD_VALBATCH" in out and "batch=0 rows=32 start_utc" in out and "batch=1 rows=24 start_utc" in out
    assert "batch=1 rows=24 end_utc" in out


def test_memwatch_samples_each_second_while_the_flag_file_exists(tmp_path):
    flag = tmp_path / "FAST"
    flag.touch()
    log = tmp_path / "m.log"
    subprocess.run(["bash", str(SCRIPTS / "memwatch.sh"), str(log), "4", str(flag)], capture_output=True, timeout=30)
    fast = sum(1 for line in log.read_text().splitlines() if line[:2].isdigit() and line[2] == ":")
    flag.unlink()
    log2 = tmp_path / "m2.log"
    subprocess.run(["bash", str(SCRIPTS / "memwatch.sh"), str(log2), "4", str(flag)], capture_output=True, timeout=30)
    slow = sum(1 for line in log2.read_text().splitlines() if line[:2].isdigit() and line[2] == ":")
    assert fast >= 3 and slow <= 1
