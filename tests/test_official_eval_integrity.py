from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "vision_opd_official"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = load_module("vcd_audit_results", EVAL_DIR / "audit_results.py")
repair_module = load_module("vcd_repair_judges", EVAL_DIR / "repair_judges.py")


def test_judge_label_canonicalization() -> None:
    assert repair_module.canonical_label("Yes") == "Yes"
    assert repair_module.canonical_label("No.") == "No"
    assert repair_module.canonical_label("Yes, the response is correct.") == "Yes"
    assert repair_module.canonical_label("") == ""
    assert repair_module.canonical_label("The response is correct.") == ""


def test_official_audit_accepts_only_complete_canonical_results(tmp_path: Path) -> None:
    benchmark_json = tmp_path / "benchmark.json"
    answer_json = tmp_path / "answers.jsonl"
    judge_json = tmp_path / "judge.json"

    benchmark_rows = [
        {"question_id": 1, "images": ["a.png"], "query": "Q1", "response": "A"},
        {"question_id": 2, "images": ["b.png"], "query": "Q2", "response": "B"},
    ]
    answers = [
        {**benchmark_rows[0], "sample_uid": "vstar:question_id:1", "model_answer": "A"},
        {**benchmark_rows[1], "sample_uid": "vstar:question_id:2", "model_answer": "B"},
    ]
    judges = [
        {**answers[0], "judge": "Yes"},
        {**answers[1], "judge": "No"},
    ]
    benchmark_json.write_text(json.dumps(benchmark_rows), encoding="utf-8")
    answer_json.write_text(
        "".join(json.dumps(item) + "\n" for item in answers),
        encoding="utf-8",
    )
    judge_json.write_text(json.dumps(judges), encoding="utf-8")

    report = audit_module.audit(
        benchmark="vstar",
        benchmark_json=benchmark_json,
        answer_json=answer_json,
        judge_json=judge_json,
        answer_source="generated",
        answer_source_path=str(answer_json),
    )
    assert report["answer_samples"] == 2
    assert report["judge_samples"] == 2
    assert report["noncanonical_judges"] == 0
    assert report["accuracy"] == 50.0

    judges[1]["judge"] = "No, because the answer differs."
    judge_json.write_text(json.dumps(judges), encoding="utf-8")
    with pytest.raises(ValueError, match="noncanonical"):
        audit_module.audit(
            benchmark="vstar",
            benchmark_json=benchmark_json,
            answer_json=answer_json,
            judge_json=judge_json,
            answer_source="generated",
            answer_source_path=str(answer_json),
        )

    judges[1]["judge"] = "No"
    judge_json.write_text(json.dumps(list(reversed(judges))), encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        audit_module.audit(
            benchmark="vstar",
            benchmark_json=benchmark_json,
            answer_json=answer_json,
            judge_json=judge_json,
            answer_source="generated",
            answer_source_path=str(answer_json),
        )
