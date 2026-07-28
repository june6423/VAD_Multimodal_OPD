#!/usr/bin/env python3
"""Canonicalize or retry non-Yes/No outputs from the official GPT judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-json", required=True, type=Path)
    parser.add_argument("--api-base", default=os.getenv("JUDGE_API_BASE", "http://127.0.0.1:8900/v1"))
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
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


def prompt_for(item: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        question=str(item.get("query", "")).replace("<image>", ""),
        gt=item.get("response", ""),
        response=item.get("extracted_answer") or item.get("model_answer", ""),
    )


def repair(
    path: Path,
    *,
    api_base: str,
    api_key: str,
    judge_model: str,
    judge_max_tokens: int,
    max_retries: int,
    retry_delay: float,
) -> dict[str, int]:
    from openai import OpenAI

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {path}")

    invalid_indices = [
        index
        for index, item in enumerate(data)
        if str(item.get("judge", "")).strip() not in {"Yes", "No"}
    ]
    if not invalid_indices:
        return {"initial_noncanonical": 0, "canonicalized": 0, "retried": 0, "remaining": 0}

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
    canonicalized = 0
    retried = 0
    for index in invalid_indices:
        item = data[index]
        original = str(item.get("judge", "")).strip()
        item["judge_raw_before_repair"] = original
        fixed = canonical_label(original)
        if fixed:
            item["judge"] = fixed
            item["judge_source"] = "llm_canonicalized"
            canonicalized += 1
            continue

        last_raw = ""
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt_for(item)}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                )
                last_raw = (response.choices[0].message.content or "").strip()
                fixed = canonical_label(last_raw)
                if fixed:
                    item["judge"] = fixed
                    item["judge_source"] = "llm_single_sample_retry"
                    item["judge_raw_retry"] = last_raw
                    retried += 1
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_delay)

        if not fixed:
            item["judge"] = original
            item["judge_raw_retry"] = last_raw
            item["judge_retry_error"] = last_error
            item["judge_source"] = "llm_retry_failed"

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)

    remaining = sum(str(item.get("judge", "")).strip() not in {"Yes", "No"} for item in data)
    return {
        "initial_noncanonical": len(invalid_indices),
        "canonicalized": canonicalized,
        "retried": retried,
        "remaining": remaining,
    }


def main() -> None:
    args = parse_args()
    result = repair(
        args.judge_json,
        api_base=args.api_base,
        api_key=args.api_key,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    print(json.dumps(result, indent=2))
    if result["remaining"]:
        raise SystemExit(
            f"Formal scoring stopped: {result['remaining']} judge outputs remain noncanonical after retry."
        )


if __name__ == "__main__":
    main()
