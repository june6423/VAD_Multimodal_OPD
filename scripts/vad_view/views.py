"""Teacher views of HANDOFF-VLM-VAD-VIEW-001-R2 Definitions 2 to 4 and Part A item 1.

View v(p, z): the box (bbox of Vision-OPD-6K/train.jsonl, row order = train_vad.parquet row order) clamped to the
student image; the crop adds p x box width on the left and the right and p x box height on the top and the bottom,
clamped to the image, in integer pixels (floor of left and top, ceil of right and bottom); the crop is resized by z
(width and height round(z x crop size), at least 1) with LANCZOS resampling.
p1 (R2 Definition 4): the padding of the released teacher images, fitted by `fit`: on the rows whose box with 0.15
padding stays inside the image, the crop size is the released size / 2 (zoom 2, every released side is even), and
k = sum(cw bw + ch bh) / sum(bw^2 + bh^2) is the least-squares slope through 0 of the crop size over the box size,
p1 = (k - 1) / 2. P1_FIT.json gives p1, the slope of each axis, the residuals cw - k bw and ch - k bh, and the count
of rows where the integer crop of v(p1, 2) has the released size.
View set: padding labels p0.00 (0), p1, p0.30 (0.30) x zoom {1.5, 2, 3}, without (p0.30, 1.5). Names <label>-z<z>.
v0 = p1-z2.0 is the released teacher image (bbox_images), as the handoff says. g3 compares a generated v(p1, 2) with it
on 20 random rows.
Image token count: image_grid_thw of the processor image_processor (the processor settings of the A1 run, not changed)
divided by merge_size^2. teacher_tokens = text tokens of the teacher prompt (chat template, one image placeholder) - 1
+ image tokens. A row with a view above max_reprompt_len (10240, vopd.yaml) is removed from every view and stage.

Usage: python views.py fit <train_vad.parquet> <train.jsonl> <P1_FIT.json>
       python views.py build <train_vad.parquet> <train.jsonl> <P1_FIT.json> <model dir> <out dir> [<rows file>]
       python views.py g3 <out dir> <P1_FIT.json> <G3_VIEWS.json> <IMAGE_TOKENS.tsv>   (20 random rows, seed 20261111;
         the TSV gives the image token count of each view and the largest teacher prompt for each row)
Env: WORKERS (32).
Writes <out>/images/<view>/<row>.png, <out>/MANIFEST.jsonl (one line per row), <out>/REMOVED.json.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

PAD_LABELS = ("p0.00", "p1", "p0.30")
FIXED_PADS = {"p0.00": 0.0, "p0.30": 0.30}
ZOOMS = (1.5, 2.0, 3.0)
VIEW_KEYS = [(lab, z) for lab in PAD_LABELS for z in ZOOMS if (lab, z) != ("p0.30", 1.5)]
V0_KEY = ("p1", 2.0)
MAX_REPROMPT_LEN = 10240
FIT_MARGIN = 0.15
G3_ROWS, G3_SEED = 20, 20261111


def view_name(label: str, z: float) -> str:
    return f"{label}-z{z:.1f}"


V0 = view_name(*V0_KEY)


def pads(p1: float) -> dict[str, float]:
    return {**FIXED_PADS, "p1": p1}


def clamp_box(bbox, size) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = bbox
    return max(0, min(x0, w)), max(0, min(y0, h)), max(0, min(x1, w)), max(0, min(y1, h))


def crop_box(bbox, size, p: float) -> tuple[int, int, int, int]:
    """Definition 3: the padded box, clamped to the image, in integer pixels."""
    w, h = size
    x0, y0, x1, y1 = clamp_box(bbox, size)
    bw, bh = x1 - x0, y1 - y0
    left = max(0, math.floor(x0 - p * bw))
    top = max(0, math.floor(y0 - p * bh))
    right = min(w, math.ceil(x1 + p * bw))
    bottom = min(h, math.ceil(y1 + p * bh))
    if right <= left or bottom <= top:
        raise ValueError(f"empty crop {bbox} in {size}")
    return left, top, right, bottom


def zoom_size(crop, z: float) -> tuple[int, int]:
    return max(1, round((crop[2] - crop[0]) * z)), max(1, round((crop[3] - crop[1]) * z))


def make_view(image, bbox, p: float, z: float):
    from PIL import Image

    crop = crop_box(bbox, image.size, p)
    return image.crop(crop).resize(zoom_size(crop, z), Image.LANCZOS), crop


def inside(bbox, size, p: float) -> bool:
    w, h = size
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    return x0 - p * bw >= 0 and y0 - p * bh >= 0 and x1 + p * bw <= w and y1 + p * bh <= h


def fit_p1(samples: list[dict]) -> dict:
    """samples: bbox, image size, released size. Least-squares slope through 0 of the crop size over the box size."""
    use = [s for s in samples if inside(s["bbox"], s["size"], FIT_MARGIN)]
    odd = sum(s["released"][0] % 2 or s["released"][1] % 2 for s in samples)
    pairs_w = [(s["bbox"][2] - s["bbox"][0], s["released"][0] / 2) for s in use]
    pairs_h = [(s["bbox"][3] - s["bbox"][1], s["released"][1] / 2) for s in use]
    slope = lambda ps: sum(b * c for b, c in ps) / sum(b * b for b, _ in ps)  # noqa: E731
    k = slope(pairs_w + pairs_h)
    p1 = (k - 1) / 2
    res = [c - k * b for b, c in pairs_w + pairs_h]
    exact = sum(zoom_size(crop_box(s["bbox"], s["size"], p1), 2.0) == tuple(s["released"]) for s in use)
    return {"p1": p1, "slope": k, "p_width": (slope(pairs_w) - 1) / 2, "p_height": (slope(pairs_h) - 1) / 2,
            "rows_fit": len(use), "rows_all": len(samples), "fit_margin": FIT_MARGIN, "released_odd_sides_rows": odd,
            "residual_px_rms": math.sqrt(sum(r * r for r in res) / len(res)), "residual_px_max_abs": max(map(abs, res)),
            "integer_crop_exact_rows": exact}


def image_tokens(processor, image) -> tuple[int, list[int]]:
    out = processor.image_processor(images=[image], return_tensors="pt")
    grid = [int(x) for x in out["image_grid_thw"][0]]
    merge = processor.image_processor.merge_size
    return grid[0] * grid[1] * grid[2] // (merge * merge), grid


def text_tokens(processor, prompt_text: str) -> int:
    """Tokens of the teacher prompt text with one image placeholder (the content before and after <image>)."""
    before, _, after = prompt_text.partition("<image>")
    content = ([{"type": "text", "text": before}] if before else []) + [{"type": "image"}] + \
        ([{"type": "text", "text": after}] if after else [])
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    return len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])


_proc = None


def _init(model_dir: str):
    global _proc
    from transformers import AutoProcessor

    _proc = AutoProcessor.from_pretrained(model_dir, use_fast=True)


def build_row(job: dict) -> dict:
    from PIL import Image

    out = Path(job["out"])
    rec = {"row": job["row"], "bbox": job["bbox"], "student": job["student"], "views": {}}
    with Image.open(job["student"]) as im:
        image = im.convert("RGB")
    rec["image_size"] = list(image.size)
    rec["bbox_clamped"] = list(clamp_box(job["bbox"], image.size))
    rec["text_tokens"] = text_tokens(_proc, job["prompt"])
    pad = pads(job["p1"])
    for lab, z in VIEW_KEYS:
        name = view_name(lab, z)
        crop = list(crop_box(job["bbox"], image.size, pad[lab]))
        if (lab, z) == V0_KEY:
            path, crop = job["teacher"], None
            with Image.open(path) as im:
                v = im.convert("RGB")
        else:
            path = str(out / "images" / name / f"{job['row']:05d}.png")
            if Path(path).exists():
                with Image.open(path) as im:
                    v = im.convert("RGB")
            else:
                v, _ = make_view(image, job["bbox"], pad[lab], z)
                tmp = path + ".tmp.png"
                v.save(tmp)
                os.replace(tmp, path)
        n, grid = image_tokens(_proc, v)
        rec["views"][name] = {"path": path, "crop": crop, "size": list(v.size), "image_tokens": n, "grid": grid,
                              "teacher_tokens": rec["text_tokens"] - 1 + n}
    rec["max_teacher_tokens"] = max(x["teacher_tokens"] for x in rec["views"].values())
    return rec


def load_rows(parquet: str, train_jsonl: str):
    import pandas as pd

    df = pd.read_parquet(parquet)
    src = [json.loads(x) for x in open(train_jsonl)]
    if len(src) != len(df):
        raise SystemExit(f"STOP: {len(src)} source lines, {len(df)} parquet rows")
    rows = []
    for i in range(len(df)):
        student = df.iloc[i]["images"][0]["path"]
        teacher = df.iloc[i]["bbox_images"][0]["path"]
        img = src[i]["images"]
        h = Path(img[0] if isinstance(img, list) else img).stem
        if Path(student).stem != h or not Path(teacher).stem.endswith(h):
            raise SystemExit(f"STOP: row {i}: source image {h} does not match {student} / {teacher}")
        rows.append({"row": i, "student": student, "teacher": teacher, "bbox": [int(x) for x in src[i]["bbox"]],
                     "prompt": df.iloc[i]["prompt"][0]["content"]})
    return rows


def fit(argv: list[str]):
    from PIL import Image

    parquet, train_jsonl, dest = argv
    samples = []
    for r in load_rows(parquet, train_jsonl):
        with Image.open(r["student"]) as a, Image.open(r["teacher"]) as b:
            samples.append({"row": r["row"], "bbox": r["bbox"], "size": list(a.size), "released": list(b.size)})
    res = fit_p1(samples)
    Path(dest).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def build(argv: list[str]):
    parquet, train_jsonl, p1_file, model_dir, out = argv[:5]
    out = Path(out)
    p1 = json.loads(Path(p1_file).read_text())["p1"]
    rows = load_rows(parquet, train_jsonl)
    if len(argv) > 5:
        keep = set(json.loads(Path(argv[5]).read_text())["rows"])
        rows = [r for r in rows if r["row"] in keep]
    for lab, z in VIEW_KEYS:
        (out / "images" / view_name(lab, z)).mkdir(parents=True, exist_ok=True)
    done = set()
    man = out / "MANIFEST.jsonl"
    if man.exists():
        done = {json.loads(x)["row"] for x in open(man)}
    jobs = [{**r, "out": str(out), "p1": p1} for r in rows if r["row"] not in done]
    with Pool(int(os.environ.get("WORKERS", "32")), initializer=_init, initargs=(model_dir,)) as pool, \
            open(man, "a") as f:
        for rec in pool.imap_unordered(build_row, jobs, chunksize=4):
            f.write(json.dumps(rec) + "\n")
            f.flush()
    recs = sorted((json.loads(x) for x in open(man)), key=lambda r: r["row"])
    removed = [{"row": r["row"], "max_teacher_tokens": r["max_teacher_tokens"]} for r in recs
               if r["max_teacher_tokens"] > MAX_REPROMPT_LEN]
    tokens = {v: [r["views"][v]["image_tokens"] for r in recs] for v in recs[0]["views"]}
    summary = {"rows": len(recs), "p1": p1, "removed": removed, "removed_fraction": len(removed) / len(recs),
               "max_reprompt_len": MAX_REPROMPT_LEN,
               "image_tokens": {v: {"min": min(t), "mean": sum(t) / len(t), "max": max(t)} for v, t in tokens.items()}}
    (out / "REMOVED.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("rows", "removed_fraction")}))


def g3(argv: list[str]):
    """Mean absolute pixel difference (0-255 scale, RGB) between a generated v(p1, 2) and the released image. When the
    sizes differ, the generated view is resized to the released size (LANCZOS) for the comparison."""
    import numpy as np
    from PIL import Image

    out, p1_file, dest, tokens_tsv = Path(argv[0]), argv[1], argv[2], argv[3]
    p1fit = json.loads(Path(p1_file).read_text())
    recs = sorted((json.loads(x) for x in open(out / "MANIFEST.jsonl")), key=lambda r: r["row"])
    names = [view_name(lab, z) for lab, z in VIEW_KEYS]
    with open(tokens_tsv, "w") as f:
        f.write("\t".join(["row"] + names + ["max_teacher_tokens"]) + "\n")
        for r in recs:
            f.write("\t".join(map(str, [r["row"]] + [r["views"][n]["image_tokens"] for n in names]
                                  + [r["max_teacher_tokens"]])) + "\n")
    pick = sorted(random.Random(G3_SEED).sample([r["row"] for r in recs], G3_ROWS))
    by = {r["row"]: r for r in recs}
    res = []
    for i in pick:
        r = by[i]
        with Image.open(r["views"][V0]["path"]) as im:
            released = im.convert("RGB")
        with Image.open(r["student"]) as im:
            gen, crop = make_view(im.convert("RGB"), r["bbox"], p1fit["p1"], 2.0)
        same = gen.size == released.size
        cmp = gen if same else gen.resize(released.size, Image.LANCZOS)
        diff = float(np.abs(np.asarray(cmp, np.float64) - np.asarray(released, np.float64)).mean())
        res.append({"row": i, "generated_size": list(gen.size), "released_size": list(released.size),
                    "same_size": same, "crop": list(crop), "mean_abs_diff": diff})
    summary = {"p1_fit": p1fit, "rows": res, "mean_abs_diff_mean": sum(x["mean_abs_diff"] for x in res) / len(res),
               "same_size_rows": sum(x["same_size"] for x in res), "seed": G3_SEED}
    Path(dest).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("mean_abs_diff_mean", "same_size_rows")}))


def main():
    cmd = sys.argv[1]
    {"fit": fit, "build": build, "g3": g3}[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()
