#!/usr/bin/env python3
"""Prepare VCD training data and its aligned visual-evidence views.

The generated parquet contains three image conditions for every sample:

* ``images``: the full image used by the student.
* ``bbox_images``: the clear evidence-present crop used by the teacher.
* ``degraded_bbox_images``: the evidence-degraded counterfactual crop.

The compatibility column names are retained because the training runtime uses
them directly. Their public semantics are recorded in each row's
``extra_info.evidence_views`` metadata and in the generated manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import datasets
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

DEFAULT_REPO = "yuanqianhao/Vision-OPD-6K"
REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)
REQUIRED_COLUMNS = {"prompt", "images", "bbox_images", "degraded_bbox_images"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Dataset download and output root.")
    parser.add_argument("--hf-repo", default=DEFAULT_REPO, help="Hugging Face dataset repository.")
    parser.add_argument("--output", default=None, help="Output parquet. Defaults to DATA_DIR/train_vcd.parquet.")
    parser.add_argument(
        "--degraded-dir",
        default=None,
        help="Evidence-degraded image directory. Defaults to DATA_DIR/evidence_degraded.",
    )
    parser.add_argument("--manifest", default=None, help="Output manifest JSON path.")
    parser.add_argument("--scale", type=float, default=0.10, help="Bilinear downsampling scale in (0, 1].")
    parser.add_argument("--workers", type=int, default=16, help="Parallel image workers.")
    parser.add_argument("--skip-download", action="store_true", help="Use an existing train.jsonl and image tree.")
    parser.add_argument("--verify-only", action="store_true", help="Validate an existing output without rewriting it.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing degraded views and parquet.")
    parser.add_argument("--remove-archives", action="store_true", help="Delete downloaded archives after extraction.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a local smoke test.")
    return parser.parse_args()


class ConcatenatedReader(io.RawIOBase):
    """Stream a sequence of split archive parts as one binary reader."""

    def __init__(self, paths: Iterable[Path]):
        self.paths = iter(paths)
        self.current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        view = memoryview(buffer)
        written = 0
        while written < len(view):
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    break
            size = self.current.readinto(view[written:])
            if size:
                written += size
                continue
            self.current.close()
            self.current = None
        return written

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def extract_split_archive(parts: list[Path], destination: Path) -> None:
    if not parts:
        return
    destination.mkdir(parents=True, exist_ok=True)
    reader = io.BufferedReader(ConcatenatedReader(parts), buffer_size=1024 * 1024)
    process = subprocess.Popen(
        ["tar", "-xf", "-", "-C", str(destination)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        shutil.copyfileobj(reader, process.stdin, length=1024 * 1024)
    finally:
        reader.close()
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


def download_and_extract(repo_id: str, data_dir: Path, remove_archives: bool) -> None:
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(data_dir))

    images_dir = data_dir / "images"
    image_parts = sorted(path for path in images_dir.glob("images.tar.gz*") if path.is_file())
    if image_parts:
        print(f"Extracting {len(image_parts)} student-image archive part(s).")
        extract_split_archive(image_parts, images_dir)

    teacher_dir = data_dir / "teacher_images"
    teacher_parts = sorted(path for path in teacher_dir.glob("teacher_images.tar.gz*") if path.is_file())
    if teacher_parts:
        print(f"Extracting {len(teacher_parts)} evidence-crop archive part(s).")
        extract_split_archive(teacher_parts, teacher_dir)

    if remove_archives:
        for path in image_parts + teacher_parts:
            path.unlink()


def clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    return text.replace(REMOVE_HINT, "").strip()


def resolve_source_path(data_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (data_dir / path).resolve()


def degraded_path(source: Path, data_dir: Path, out_dir: Path) -> Path:
    try:
        stable_name = str(source.relative_to(data_dir))
    except ValueError:
        stable_name = str(source)
    digest = hashlib.sha1(stable_name.encode("utf-8")).hexdigest()[:12]
    return (out_dir / f"{digest}_{source.stem}.png").resolve()


def degrade_image(source: Path, target: Path, scale: float, overwrite: bool) -> dict[str, Any]:
    if target.exists() and not overwrite:
        with Image.open(source) as clear_image, Image.open(target) as degraded_image:
            if clear_image.size != degraded_image.size:
                raise ValueError(f"Cached degraded image has the wrong size: {target}")
            width, height = clear_image.size
        status = "reused"
    else:
        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            low_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            low = image.resize(low_size, Image.Resampling.BILINEAR)
            result = low.resize((width, height), Image.Resampling.NEAREST)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target, format="PNG")
        status = "created"

    return {
        "source": str(source),
        "target": str(target),
        "width": width,
        "height": height,
        "low_width": max(1, round(width * scale)),
        "low_height": max(1, round(height * scale)),
        "status": status,
    }


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            records.append(json.loads(line))
    return records


def build_record(
    item: dict[str, Any],
    data_dir: Path,
    clear_path: Path,
    degraded_path_value: Path,
    scale: float,
) -> dict[str, Any]:
    full_path = resolve_source_path(data_dir, item["images"][0])
    problem = item.get("problem", "")
    return {
        # Retain the upstream identifier because the training-time benchmark
        # reward adapter recognizes this source as multiple choice.
        "data_source": "zwz_rl_vqa_bbox_teacher",
        "prompt": [{"role": "user", "content": problem}],
        "images": [{"path": str(full_path)}],
        "bbox_images": [{"path": str(clear_path)}],
        "degraded_bbox_images": [{"path": str(degraded_path_value)}],
        "ability": "visual_question_answering",
        "reward_model": {"style": "none", "ground_truth": item.get("answer", "")},
        "extra_info": {
            "answer": item.get("answer", ""),
            "question": clean_question(problem),
            "source_extra_info": item.get("extra_info", {}),
            "evidence_views": {
                "student_input": "images",
                "evidence_present": "bbox_images",
                "evidence_degraded": "degraded_bbox_images",
                "degradation_scale": scale,
                "downsample_resampling": "bilinear",
                "upsample_resampling": "nearest",
            },
        },
    }


def extract_single_path(value: Any, column: str, row_index: int) -> Path:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"Row {row_index} column {column!r} must contain exactly one image.")
    cell = value[0]
    if not isinstance(cell, dict) or not cell.get("path"):
        raise ValueError(f"Row {row_index} column {column!r} has no path.")
    return Path(cell["path"])


def verify_parquet(path: Path, expected_rows: int | None = None) -> dict[str, Any]:
    dataset = datasets.Dataset.from_parquet(str(path))
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if expected_rows is not None and len(dataset) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(dataset)} in {path}.")

    for row_index, row in enumerate(tqdm(dataset, desc="Verifying VCD data", unit="row")):
        full_path = extract_single_path(row["images"], "images", row_index)
        clear_path = extract_single_path(row["bbox_images"], "bbox_images", row_index)
        degraded = extract_single_path(row["degraded_bbox_images"], "degraded_bbox_images", row_index)
        missing_paths = [item for item in (full_path, clear_path, degraded) if not item.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Row {row_index} references missing images: {missing_paths}")
        with Image.open(clear_path) as clear_image, Image.open(degraded) as degraded_image:
            if clear_image.size != degraded_image.size:
                raise ValueError(
                    f"Row {row_index} evidence-present and evidence-degraded sizes differ: "
                    f"{clear_image.size} versus {degraded_image.size}."
                )
        metadata = row.get("extra_info", {}).get("evidence_views", {})
        if metadata.get("evidence_present") != "bbox_images":
            raise ValueError(f"Row {row_index} has invalid evidence-present metadata.")
        if metadata.get("evidence_degraded") != "degraded_bbox_images":
            raise ValueError(f"Row {row_index} has invalid evidence-degraded metadata.")

    return {
        "path": str(path.resolve()),
        "rows": len(dataset),
        "columns": dataset.column_names,
        "required_columns": sorted(REQUIRED_COLUMNS),
        "status": "verified",
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 < args.scale <= 1:
        raise ValueError(f"--scale must be in (0, 1], got {args.scale}.")
    if args.workers < 1:
        raise ValueError(f"--workers must be positive, got {args.workers}.")
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be positive, got {args.limit}.")

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve() if args.output else data_dir / "train_vcd.parquet"
    degraded_dir = Path(args.degraded_dir).resolve() if args.degraded_dir else data_dir / "evidence_degraded"
    manifest = Path(args.manifest).resolve() if args.manifest else data_dir / "vcd_data_manifest.json"

    if args.verify_only:
        summary = verify_parquet(output, expected_rows=args.limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_and_extract(args.hf_repo, data_dir, args.remove_archives)

    jsonl_path = data_dir / "train.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {jsonl_path}")
    if output.exists() and not args.overwrite:
        summary = verify_parquet(output, expected_rows=args.limit)
        print(f"Reusing verified output: {output}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    source_records = read_jsonl(jsonl_path, args.limit)
    evidence_present_paths = [
        resolve_source_path(data_dir, item["teacher_images"][0]) for item in source_records
    ]
    missing_sources = [path for path in evidence_present_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing evidence-present source images, first examples: {missing_sources[:5]}")

    degraded_paths = [
        degraded_path(source, data_dir, degraded_dir) for source in evidence_present_paths
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(degrade_image, source, target, args.scale, args.overwrite)
            for source, target in zip(evidence_present_paths, degraded_paths, strict=True)
        ]
        degradation_records = [
            job.result() for job in tqdm(jobs, desc="Preparing degraded evidence", unit="image")
        ]

    output_records = [
        build_record(item, data_dir, clear_path, degraded, args.scale)
        for item, clear_path, degraded in zip(
            source_records,
            evidence_present_paths,
            degraded_paths,
            strict=True,
        )
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(output_records).to_parquet(str(output))

    verification = verify_parquet(output, expected_rows=len(source_records))
    summary = {
        **verification,
        "source_repo": args.hf_repo,
        "source_jsonl": str(jsonl_path),
        "degraded_dir": str(degraded_dir),
        "degradation_scale": args.scale,
        "downsample_resampling": "bilinear",
        "upsample_resampling": "nearest",
        "created": sum(record["status"] == "created" for record in degradation_records),
        "reused": sum(record["status"] == "reused" for record in degradation_records),
        "view_mapping": {
            "student_input": "images",
            "evidence_present": "bbox_images",
            "evidence_degraded": "degraded_bbox_images",
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
