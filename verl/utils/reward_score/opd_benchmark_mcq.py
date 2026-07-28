"""Fast rule-based MCQ scoring for OPD benchmark validation."""

from __future__ import annotations

import re
from typing import Any


MCQ_DATA_SOURCES = {"vstar", "zoombench", "hrbench-4k", "hrbench-8k"}


def extract_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    value = text.strip()
    think_end = value.rfind("</think>")
    if think_end != -1:
        value = value[think_end + len("</think>") :].strip()

    start = value.rfind("<answer>")
    end = value.find("</answer>", start + len("<answer>")) if start != -1 else -1
    if start != -1 and end != -1 and end > start:
        return value[start + len("<answer>") : end].strip()

    marker_patterns = [
        r"(?:final\s+answer|answer|答案)\s*[:：]\s*([A-F])\b",
        r"(?:final\s+answer|answer|答案)\s+is\s+([A-F])\b",
    ]
    for pattern in marker_patterns:
        matches = re.findall(pattern, value, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return value


def extract_first_option(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    value = extract_answer(text)
    patterns = [
        r"^\s*[\(\[]?\s*([A-F])\s*[\)\].:：]?\s*$",
        r"^\s*[\(\[]?\s*([A-F])\s*[\)\].:：]",
        r"[\(\[]\s*([A-F])\s*[\)\]]",
        r"\b([A-F])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def extract_mcq_option(answer: Any) -> str:
    if not isinstance(answer, str) or not answer.strip():
        return ""
    value = answer.strip()
    match = re.match(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))", value, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-F])\b", value, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def compute_score(solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    pred = extract_first_option(solution_str)
    gt = extract_mcq_option(str(ground_truth))
    is_correct = bool(pred and gt and pred == gt)
    return {
        "score": float(is_correct),
        "acc": float(is_correct),
        "pred": pred,
        "gt": gt,
        "extracted_answer": extract_answer(solution_str),
        "judge_source": "first_letter_rule",
    }
