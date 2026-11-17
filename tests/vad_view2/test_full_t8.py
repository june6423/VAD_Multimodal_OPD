"""Tests of full_t8() of scripts/vad_view2/run_view2.sh (DIRECTIVE-VLM-VAD-VIEW-002-008).

The function text is taken from the script and run in bash with a stub train(). The stub records its arguments and
writes the checkpoint pointer: a normal call reaches the stop step, a failing call reaches only <fail_latest>.
"""

import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "vad_view2" / "run_view2.sh"


def _full_t8_text():
    m = re.search(r"^full_t8\(\) \{\n.*?^\}\n", SCRIPT.read_text(), re.S | re.M)
    assert m, "full_t8 not found"
    return m.group(0)


def _run(tmp_path, latest, failures):
    """failures: {tag: latest written by the failed call}. Returns (exit code, train calls, log lines)."""
    out = tmp_path / "o"
    (out / "checkpoints").mkdir(parents=True)
    (out / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(str(latest))
    fail_cases = "\n".join(f"    {tag}) echo {lat} > \"$out/checkpoints/latest_checkpointed_iteration.txt\"; return 1 ;;"
                           for tag, lat in failures.items())
    stub = f"""
out="{out}"; log="{tmp_path / 'log.txt'}"; : > "$log"
train() {{
  echo "train $*" >> "{tmp_path / 'calls.txt'}"
  case "$3" in
{fail_cases}
    *) echo "$4" > "$out/checkpoints/latest_checkpointed_iteration.txt" ;;
  esac
}}
{_full_t8_text()}
full_t8
"""
    code = subprocess.run(["bash", "-c", stub]).returncode
    calls = (tmp_path / "calls.txt").read_text().splitlines() if (tmp_path / "calls.txt").exists() else []
    return code, calls, (tmp_path / "log.txt").read_text().splitlines()


def test_four_processes_of_ten_steps_with_total_40_and_save_5(tmp_path):
    code, calls, _ = _run(tmp_path, 0, {})
    assert code == 0
    assert calls == ["train 40 5 full10_1 10", "train 40 5 full20_1 20", "train 40 5 full30_1 30", "train 40 5 full40_1 40"]


def test_a_restarted_task_starts_at_the_first_unfinished_range(tmp_path):
    code, calls, _ = _run(tmp_path, 20, {})
    assert code == 0 and calls == ["train 40 5 full30_1 30", "train 40 5 full40_1 40"]


def test_one_failure_gives_a_second_process_in_the_same_range(tmp_path):
    code, calls, log = _run(tmp_path, 0, {"full20_1": 15})
    assert code == 0
    assert calls == ["train 40 5 full10_1 10", "train 40 5 full20_1 20", "train 40 5 full20_2 20",
                     "train 40 5 full30_1 30", "train 40 5 full40_1 40"]
    assert any("range to step 20: failure at latest 15" in line and "second process" in line for line in log)


def test_two_failures_in_one_range_stop_the_arm(tmp_path):
    code, calls, log = _run(tmp_path, 0, {"full30_1": 25, "full30_2": 25})
    assert code == 1
    assert calls[-2:] == ["train 40 5 full30_1 30", "train 40 5 full30_2 30"] and len(calls) == 4
    assert log[-1] == "two failures in the range to step 30: stop"
