#!/usr/bin/env python3
"""Audit answer and judge completeness for the official evaluation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--answer-json", required=True, type=Path)
    parser.add_argument("--judge-json", type=Path)
    parser.add_argument("--answer-source", choices=["generated", "reused", "resumed"], default="generated")
    parser.add_argument("--answer-source-path", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def stable_uid(item: dict[str, Any], benchmark: str) -> str:
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    payload = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def read_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"Expected a JSON list of objects: {path}")
    return value


def audit(
    *,
    benchmark: str,
    benchmark_json: Path,
    answer_json: Path,
    judge_json: Path | None,
    answer_source: str,
    answer_source_path: str,
) -> dict[str, Any]:
    benchmark_rows = read_json_list(benchmark_json)
    answers = read_jsonl(answer_json)
    expected = len(benchmark_rows)
    if len(answers) != expected:
        raise ValueError(
            f"{benchmark} answer count mismatch: expected {expected}, found {len(answers)} in {answer_json}"
        )

    uids = [stable_uid(item, benchmark) for item in answers]
    unique_uids = set(uids)
    if len(unique_uids) != expected:
        raise ValueError(f"{benchmark} has {expected - len(unique_uids)} duplicate answer sample IDs.")

    invalid_answers = []
    for index, item in enumerate(answers):
        value = item.get("model_answer")
        stripped = value.strip() if isinstance(value, str) else ""
        if (
            not stripped
            or stripped.startswith("[API_ERROR]")
            or stripped.startswith("[FUTURE_ERROR]")
        ):
            invalid_answers.append(index)
    if invalid_answers:
        raise ValueError(
            f"{benchmark} has {len(invalid_answers)} empty or failed answers; "
            f"first indices: {invalid_answers[:10]}"
        )

    report: dict[str, Any] = {
        "benchmark": benchmark,
        "benchmark_json": str(benchmark_json.resolve()),
        "answer_json": str(answer_json.resolve()),
        "answer_source": answer_source,
        "answer_source_path": answer_source_path,
        "expected_samples": expected,
        "answer_samples": len(answers),
        "unique_answer_samples": len(unique_uids),
        "invalid_answers": 0,
    }

    if judge_json is not None:
        judges = read_json_list(judge_json)
        if len(judges) != expected:
            raise ValueError(
                f"{benchmark} judge count mismatch: expected {expected}, found {len(judges)} in {judge_json}"
            )
        judge_uids = [stable_uid(item, benchmark) for item in judges]
        if len(set(judge_uids)) != expected:
            raise ValueError(f"{benchmark} judge output contains duplicate sample IDs.")
        if judge_uids != uids:
            raise ValueError(f"{benchmark} judge sample order does not match the audited answers.")
        invalid_labels = [
            index
            for index, item in enumerate(judges)
            if str(item.get("judge", "")).strip() not in {"Yes", "No"}
        ]
        if invalid_labels:
            raise ValueError(
                f"{benchmark} has {len(invalid_labels)} noncanonical judge labels; "
                f"first indices: {invalid_labels[:10]}"
            )
        correct = sum(item["judge"] == "Yes" for item in judges)
        report.update(
            {
                "judge_json": str(judge_json.resolve()),
                "judge_samples": len(judges),
                "unique_judge_samples": len(set(judge_uids)),
                "noncanonical_judges": 0,
                "correct": correct,
                "accuracy": 100.0 * correct / expected if expected else 0.0,
            }
        )

    return report


def main() -> None:
    args = parse_args()
    report = audit(
        benchmark=args.benchmark,
        benchmark_json=args.benchmark_json,
        answer_json=args.answer_json,
        judge_json=args.judge_json,
        answer_source=args.answer_source,
        answer_source_path=args.answer_source_path,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
