"""label.py rules (G5) with a fake judge: rule rows, R8 rule, transport retries, and the TASK-053 consistency check."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "vision_opd_official"))
from repair_judges import canonical_label  # noqa: E402

spec = importlib.util.spec_from_file_location("grad_label", ROOT / "scripts" / "vad_grad" / "label.py")
label = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label)


def rows_and_inputs():
    rows = [{"traj": 0, "correct": True, "judge_source": "first letter"},
            {"traj": 1, "correct": False, "judge_source": "llm"},
            {"traj": 2, "correct": False, "judge_source": "llm"},
            {"traj": 3, "correct": False, "judge_source": "llm"},
            {"traj": 4, "correct": False, "judge_source": "llm"}]
    inputs = [("src", f"response {r['traj']}", "A", {}) for r in rows]
    return rows, inputs


def fake_judge(outputs: dict[str, list[str]]):
    """Rule result for response 0; the judge text for the others comes from a queue per response."""
    calls = []

    def judge(sources, responses, gts, extras):
        calls.append(list(responses))
        out = []
        for resp in responses:
            if resp == "response 0":
                out.append({"judge": "Yes", "judge_source": "first letter"})
            else:
                out.append({"judge": outputs[resp].pop(0), "judge_source": "llm"})
        return out

    return judge, calls


def test_rules_retry_and_r8():
    rows, inputs = rows_and_inputs()
    judge, calls = fake_judge({"response 1": ["Yes"], "response 2": ["No"], "response 3": ["Maybe."],
                               "response 4": ["[JUDGE_API_ERROR] timeout", "[JUDGE_API_ERROR] timeout", "yes, it is"]})
    recs, counts = label.label_rows(rows, inputs, judge, canonical_label)
    by = {r["traj"]: r for r in recs}
    assert by[0]["label"] == 1 and by[0]["source"] == "rule"
    assert by[1]["label"] == 1 and by[1]["source"] == "judge"
    assert by[2]["label"] == 0 and by[2]["source"] == "judge"
    assert by[3]["label"] == 0 and by[3]["source"] == "noncanonical_as_incorrect" and by[3]["judge_raw"] == "Maybe."
    assert by[4]["label"] == 1 and by[4]["source"] == "judge"
    assert counts["retry_rounds"] == 2 and counts["judge_calls"] == 4 and "transport_error" not in counts
    assert calls[1] == ["response 4"] and calls[2] == ["response 4"]  # only the failed rows go again


def test_row_without_label_after_three_retries():
    rows, inputs = rows_and_inputs()
    err = "[JUDGE_API_ERROR] connection refused"
    judge, _ = fake_judge({"response 1": ["Yes"], "response 2": ["No"], "response 3": ["No"], "response 4": [err] * 4})
    recs, counts = label.label_rows(rows, inputs, judge, canonical_label)
    assert counts["transport_error"] == 1 and counts["retry_rounds"] == 3
    assert [r for r in recs if r["traj"] == 4][0]["label"] is None


def test_rule_row_must_match_task_053():
    rows, inputs = rows_and_inputs()
    rows[0]["correct"] = False
    judge, _ = fake_judge({"response 1": ["Yes"], "response 2": ["No"], "response 3": ["No"], "response 4": ["No"]})
    with pytest.raises(SystemExit, match="differs from TASK-053"):
        label.label_rows(rows, inputs, judge, canonical_label)
