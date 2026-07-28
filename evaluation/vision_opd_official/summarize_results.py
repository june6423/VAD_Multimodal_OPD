#!/usr/bin/env python3
"""Build the official six-benchmark summary and reproducibility metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

DEFAULT_BENCHMARKS = (
    "vstar",
    "zoombench",
    "hrbench-4k",
    "hrbench-8k",
    "mme-realworld",
    "mme-realworld-cn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-api-base", default="")
    parser.add_argument("--judge-api-base", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-max-tokens", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--enable-thinking", required=True)
    parser.add_argument("--evaluator-commit", default="c2e345f")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    audits = []
    for benchmark in benchmarks:
        path = args.run_dir / "audits" / f"{benchmark}.json"
        audits.append(json.loads(path.read_text(encoding="utf-8")))

    scores = {item["benchmark"]: float(item["accuracy"]) for item in audits}
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    code_dir = Path(__file__).resolve().parent
    evaluator_files = [
        code_dir / "infer.py",
        code_dir / "judge_qwenlm.py",
        code_dir / "cal_acc.py",
        code_dir / "prepare_data.py",
        code_dir / "repair_judges.py",
        code_dir / "audit_results.py",
    ]
    summary = {
        "protocol": "Vision-OPD official pipeline + GPT-OSS-120B judge",
        "model_tag": args.model_tag,
        "model_id": args.model_id,
        "model_path": args.model_path,
        "model_api_base": args.model_api_base,
        "inference": {
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking,
            "temperature": 0,
            "recommended_server_max_model_len": 32768,
        },
        "judge": {
            "api_base": args.judge_api_base,
            "model": args.judge_model,
            "api_key_recorded": False,
            "max_tokens": args.judge_max_tokens,
            "temperature": 0,
        },
        "benchmarks": audits,
        "scores": scores,
        "avg6": avg,
        "evaluator": {
            "source": "VisionOPD/Vision-OPD",
            "source_commit": args.evaluator_commit,
            "files": {path.name: sha256(path) for path in evaluator_files},
        },
        "completed_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
    }

    json_path = args.run_dir / "summary.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)

    headers = benchmarks + ["Avg6"]
    values = [scores[name] for name in benchmarks] + [avg]
    markdown = [
        "# Official evaluation summary",
        "",
        f"Protocol: {summary['protocol']}",
        "",
        "| Model | " + " | ".join(headers) + " |",
        "| --- | " + " | ".join("---:" for _ in headers) + " |",
        f"| {args.model_tag} | " + " | ".join(f"{value:.2f}" for value in values) + " |",
        "",
        "Completeness audit",
        "",
        "| Benchmark | Answers | Judges | Non-Yes/No | Answer source |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in audits:
        markdown.append(
            f"| {item['benchmark']} | {item['answer_samples']} | {item['judge_samples']} | "
            f"{item['noncanonical_judges']} | {item['answer_source']} |"
        )
    markdown.extend(
        [
            "",
            f"Metadata: `{json_path.name}`",
            "",
        ]
    )
    markdown_path = args.run_dir / "summary.md"
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    markdown_tmp.write_text("\n".join(markdown), encoding="utf-8")
    os.replace(markdown_tmp, markdown_path)
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
