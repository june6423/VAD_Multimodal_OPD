"""VIEWMAP.json of VLM-VAD-VIEW-002 (HANDOFF-VLM-VAD-VIEW-002-R1 Part A item 1, Definition 2).

Reuse of the VIEW-001 view images (TASK-VLM-VAD-VIEW-001-006 views/, made by views.py, a copy of which is in this
directory without change): the reuse is allowed when
  1. the sha256 of MANIFEST.jsonl equals EXPECTED (the manifest that VIEW-001 used; argument "-" skips this check for
     a manifest that check_view2.sh made again),
  2. the manifest has one line for each parquet row, each line has the 8 views of hook.VIEW_NAMES, the v0 path
     (p1-z2.0) of each line equals the bbox_images path of its parquet row, and each image file exists,
  3. for SAMPLE_ROWS random rows (seed 20261113) each of the 7 made views is equal in pixels to views.make_view of the
     student image (views.pads with p1 of P1_FIT.json).
Exit 3 when check 1 or 3 fails (check_view2.sh then makes the views again), exit 1 on another failure.
VIEWMAP.json: {"views": VIEW_NAMES, "rows": {<released teacher path>: [8 paths in slot order]}, "manifest_sha256"}.
VIEWMAP_CHECK.json: the result of each check.
Usage: python viewmap.py <train_vad.parquet> <views dir> <EXPECTED manifest sha256 | -> <out dir>
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hook  # noqa: E402
import views  # noqa: E402

SAMPLE_ROWS, SAMPLE_SEED = 30, 20261113


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def build_map(teacher_paths: list[str], manifest: list[dict]) -> tuple[dict, list[str]]:
    """The row map and the list of problems (checks 2)."""
    problems = []
    by_row = {m["row"]: m for m in manifest}
    if len(by_row) != len(manifest) or sorted(by_row) != list(range(len(teacher_paths))):
        problems.append(f"manifest rows {len(by_row)} (lines {len(manifest)}), parquet rows {len(teacher_paths)}")
        return {}, problems
    if len(set(teacher_paths)) != len(teacher_paths):
        problems.append("the bbox_images paths are not unique")
    rows = {}
    for i, key in enumerate(teacher_paths):
        rec = by_row[i]["views"]
        if set(rec) != set(hook.VIEW_NAMES):
            problems.append(f"row {i}: views {sorted(rec)}")
            continue
        paths = [rec[name]["path"] for name in hook.VIEW_NAMES]
        if paths[hook.V0_SLOT - 1] != key:
            problems.append(f"row {i}: v0 path {paths[hook.V0_SLOT - 1]} is not the bbox_images path {key}")
        rows[key] = paths
    return rows, problems


def pixel_check(manifest: list[dict], p1: float) -> list[dict]:
    import numpy as np
    from PIL import Image

    pads = views.pads(p1)
    ordered = sorted(manifest, key=lambda r: r["row"])
    out = []
    for i in sorted(random.Random(SAMPLE_SEED).sample(range(len(ordered)), min(SAMPLE_ROWS, len(ordered)))):
        m = ordered[i]
        with Image.open(m["student"]) as im:
            student = im.convert("RGB")
        for lab, z in views.VIEW_KEYS:
            if (lab, z) == views.V0_KEY:
                continue
            name = views.view_name(lab, z)
            made, _ = views.make_view(student, m["bbox"], pads[lab], z)
            with Image.open(m["views"][name]["path"]) as im:
                stored = im.convert("RGB")
            equal = made.size == stored.size and bool(np.array_equal(np.asarray(made), np.asarray(stored)))
            out.append({"row": m["row"], "view": name, "equal": equal})
    return out


def main():
    parquet, vdir, expected, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    man_path = vdir / "MANIFEST.jsonl"
    got = sha256(man_path)
    manifest = [json.loads(x) for x in open(man_path)]
    df = pd.read_parquet(parquet, columns=["bbox_images"])
    teacher_paths = [hook.row_key(x) for x in df["bbox_images"]]
    rows, problems = build_map(teacher_paths, manifest)
    missing = [p for paths in rows.values() for p in paths if not Path(p).is_file()]
    p1 = json.loads((vdir / "P1_FIT.json").read_text())["p1"]
    pixels = pixel_check(manifest, p1) if not problems else []
    check = {"manifest": str(man_path), "manifest_sha256": got, "expected_sha256": expected,
             "manifest_match": expected == "-" or got == expected, "rows": len(rows), "parquet_rows": len(teacher_paths),
             "problems": problems[:50], "problem_count": len(problems), "missing_files": missing[:50],
             "missing_count": len(missing), "p1": p1, "pixel_check": pixels,
             "pixel_equal": sum(x["equal"] for x in pixels), "pixel_compared": len(pixels)}
    check["reuse"] = bool(check["manifest_match"] and pixels and check["pixel_equal"] == len(pixels))
    (out / "VIEWMAP_CHECK.json").write_text(json.dumps(check, indent=1))
    print(json.dumps({k: v for k, v in check.items() if k not in ("pixel_check", "problems", "missing_files")}))
    if problems or missing:
        sys.exit(1)
    if not check["reuse"]:
        sys.exit(3)
    (out / "VIEWMAP.json").write_text(json.dumps({"views": list(hook.VIEW_NAMES), "rows": rows,
                                                  "manifest_sha256": got, "views_dir": str(vdir)}))


if __name__ == "__main__":
    main()
