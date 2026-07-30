from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
from PIL import Image

from scripts.prepare_data import prepare, verify_parquet


def test_prepare_three_aligned_views(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "teacher_images").mkdir(parents=True)

    full_path = data_dir / "images" / "full.png"
    clear_path = data_dir / "teacher_images" / "clear.png"
    Image.new("RGB", (32, 24), color=(20, 40, 60)).save(full_path)
    clear = Image.new("RGB", (16, 12), color=(200, 100, 20))
    clear.putpixel((3, 4), (0, 255, 0))
    clear.save(clear_path)

    source = {
        "images": ["images/full.png"],
        "teacher_images": ["teacher_images/clear.png"],
        "problem": "<image>\nWhich color is visible?",
        "answer": "green",
        "extra_info": {"source": "unit-test"},
    }
    (data_dir / "train.jsonl").write_text(json.dumps(source) + "\n", encoding="utf-8")
    output = data_dir / "train_vad.parquet"
    manifest = data_dir / "manifest.json"
    args = argparse.Namespace(
        data_dir=str(data_dir),
        hf_repo="unused",
        output=str(output),
        degraded_dir=None,
        manifest=str(manifest),
        scale=0.10,
        workers=2,
        skip_download=True,
        verify_only=False,
        overwrite=True,
        remove_archives=False,
        limit=None,
    )

    summary = prepare(args)
    assert summary["rows"] == 1
    assert summary["view_mapping"]["evidence_present"] == "bbox_images"
    assert summary["view_mapping"]["evidence_degraded"] == "degraded_bbox_images"

    row = datasets.Dataset.from_parquet(str(output))[0]
    degraded_path = Path(row["degraded_bbox_images"][0]["path"])
    assert degraded_path.is_file()
    with Image.open(clear_path) as clear_image, Image.open(degraded_path) as degraded:
        assert clear_image.size == degraded.size
    assert row["extra_info"]["evidence_views"]["student_input"] == "images"
    assert verify_parquet(output)["status"] == "verified"
    assert json.loads(manifest.read_text(encoding="utf-8"))["rows"] == 1
