#!/usr/bin/env python3
"""Build a lightweight OPD benchmark validation parquet for in-training eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

BENCHMARK_JSON_MAP = {
    "vstar": "vstar.json",
    "zoombench": "zoombench.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default=None, help="Directory containing Vision-OPD eval JSON files.")
    parser.add_argument(
        "--benchmarks",
        default="vstar,zoombench,hrbench-4k,hrbench-8k",
        help="Comma-separated benchmark names.",
    )
    parser.add_argument("--out", default=None, help="Output parquet path.")
    return parser.parse_args()


def normalize_benchmarks(value: str) -> list[str]:
    benchmarks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in benchmarks if item not in BENCHMARK_JSON_MAP]
    if unknown:
        supported = ", ".join(BENCHMARK_JSON_MAP)
        raise ValueError(f"Unsupported benchmarks: {unknown}. Supported: {supported}")
    return benchmarks


def prompt_with_image_placeholder(query: str) -> str:
    query = (query or "").strip()
    if "<image>" in query:
        return query
    return "<image>\n" + query


def image_cells(images: Any) -> list[dict[str, str]]:
    if not images:
        return []
    if isinstance(images, str):
        images = [images]
    result = []
    for image in images:
        if isinstance(image, dict):
            if image.get("path"):
                result.append({"path": str(image["path"])})
            elif image.get("image"):
                result.append({"path": str(image["image"])})
            else:
                result.append(image)
        else:
            result.append({"path": str(image)})
    return result


def load_benchmark(eval_dir: Path, benchmark: str) -> list[dict[str, Any]]:
    json_path = eval_dir / BENCHMARK_JSON_MAP[benchmark]
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing {benchmark} JSON: {json_path}. Run eval/prepare_data.py for this benchmark first."
        )
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, item in enumerate(data):
        sample_id = item.get("question_id", item.get("index", item.get("id", idx)))
        rows.append(
            {
                "data_source": benchmark,
                "prompt": [{"role": "user", "content": prompt_with_image_placeholder(item.get("query", ""))}],
                "images": image_cells(item.get("images")),
                "ability": "opd_fast_eval",
                "reward_model": {"style": "none", "ground_truth": str(item.get("response", "")).strip()},
                "extra_info": {
                    "index": int(idx),
                    "benchmark": benchmark,
                    "sample_id": str(sample_id),
                    "query": (item.get("query") or "").strip(),
                    "category": str(item.get("category", "") or "unknown"),
                    "question_type": "mcq",
                },
            }
        )
    return rows


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve() if args.eval_dir else script_dir
    out = Path(args.out).resolve() if args.out else eval_dir / "opd_fast_benchmarks.parquet"
    benchmarks = normalize_benchmarks(args.benchmarks)

    rows = []
    counts = {}
    for benchmark in benchmarks:
        bench_rows = load_benchmark(eval_dir, benchmark)
        rows.extend(bench_rows)
        counts[benchmark] = len(bench_rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)

    print(f"Wrote {len(rows)} validation cases to {out}")
    for benchmark in benchmarks:
        print(f"{benchmark}: {counts[benchmark]}")


if __name__ == "__main__":
    main()
