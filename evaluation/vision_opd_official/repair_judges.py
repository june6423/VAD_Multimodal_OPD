#!/usr/bin/env python3
"""Canonicalize non-Yes/No outputs from the official GPT judge; count the rest as incorrect.

VLM-VAD-ROLLOUT-001 R8 rule: a label that canonical_label maps keeps that mapping. Every other label
becomes No, with no retry and no stop. The raw label stays in judge_raw_before_repair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-json", required=True, type=Path)
    # scripts/evaluate_official.sh passes the judge endpoint arguments. The rule makes no judge call.
    parser.add_argument("--api-base", default=os.getenv("JUDGE_API_BASE", "http://127.0.0.1:8900/v1"))
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    return parser.parse_args()


def canonical_label(value: Any) -> str:
    raw = str(value or "").strip()
    simplified = raw.lower().strip(" \t\r\n.!。")
    if simplified == "yes":
        return "Yes"
    if simplified == "no":
        return "No"
    match = re.match(r"^\s*(yes|no)\b", raw, flags=re.IGNORECASE)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    return ""


def repair(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {path}")

    invalid_indices = [
        index
        for index, item in enumerate(data)
        if str(item.get("judge", "")).strip() not in {"Yes", "No"}
    ]
    if not invalid_indices:
        return {"initial_noncanonical": 0, "canonicalized": 0, "as_incorrect": 0}

    canonicalized = 0
    as_incorrect = 0
    for index in invalid_indices:
        item = data[index]
        original = str(item.get("judge", "")).strip()
        item["judge_raw_before_repair"] = original
        fixed = canonical_label(original)
        if fixed:
            item["judge"] = fixed
            item["judge_source"] = "llm_canonicalized"
            canonicalized += 1
        else:
            item["judge"] = "No"
            item["judge_source"] = "noncanonical_as_incorrect"
            as_incorrect += 1

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)

    return {
        "initial_noncanonical": len(invalid_indices),
        "canonicalized": canonicalized,
        "as_incorrect": as_incorrect,
    }


def main() -> None:
    args = parse_args()
    result = repair(args.judge_json)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
