"""HANDOFF-VLM-VAD-ROLLOUT-001-R9 Part B: a8/schedule.py phases, failure stop, and MemAvailable stop."""

import subprocess
import sys
from pathlib import Path

SCHEDULE = Path(__file__).resolve().parents[2] / "scripts" / "vad_rollout" / "a8" / "schedule.py"


def run(tmp_path, lines, limit):
    jobs = tmp_path / "jobs.tsv"
    jobs.write_text("".join(f"{p}\t{n}\t{c}\n" for p, n, c in lines))
    proc = subprocess.run([sys.executable, str(SCHEDULE), str(jobs), "0,1", str(tmp_path / "logs"), str(limit)])
    return proc.returncode, (tmp_path / "logs" / "SCHEDULE.txt").read_text()


def test_phases_run_in_order_and_a_failure_stops(tmp_path):
    marker = tmp_path / "marker"
    code, record = run(tmp_path, [(1, "a", f"touch {marker}"), (2, "b", f"test -f {marker}"), (3, "c", "false"),
                                  (4, "d", "true")], limit=0)
    assert code == 1
    assert "\ta\t" in record and "\tb\t" in record and "exit 0" in record
    assert "STOP job c exit 1" in record and "\td\t" not in record


def test_low_mem_available_stops_with_exit_3(tmp_path):
    code, record = run(tmp_path, [(1, "slow", "sleep 60")], limit=1e9)
    assert code == 3
    assert "R9 stop condition 2" in record
