"""DIRECTIVE-VLM-VAD-VIEW-001-003 checks P and V (after the FULL task; new file, the FULL scripts do not change).

Letter values of one row or item from its letter log probabilities l (Definition 6): log m = logsumexp(l),
m = exp(log m), log R = l_gt - log m; letter argmax = the first letter with the highest l (stats1.argmax).

p <FULL output root> <CHECK_P.json>: check P. For M0 and A1_80, the bf16 path (heldout.py grad,
  <ck>/HELDOUT_VALUES.jsonl) against the fp32 alpha-0 line search (<ck>/linesearch/points/a0-r1.jsonl) on the same items.
  For each checkpoint and set (A, B): n, and for log R and m (the values that each path recorded) the mean, the median
  and the largest of |bf16 - fp32|, and the count of items with a different letter argmax.
gen <FULL views dir> <P1_FIT.json> <out views dir>: the generated v(p1, 2) of each kept row of the FULL manifest
  (views.make_view of the student image, the row bbox, pad p1, zoom 2) as <out>/images/gen-p1-z2.0/<row>.png, and a
  manifest with two views per row: the released v0 entry of the FULL manifest (stage1.py checks its path) and
  gen-p1-z2.0. REMOVED.json is copied. Resume: rows with an image are not made again. WORKERS (32) processes.
v <FULL FRESH.jsonl> <GEN FRESH.jsonl> <out views dir> <CHECK_V.json> <CHECK_V_SIZES.tsv>: check V. Against the
  released v0 (FULL FRESH p1-z2.0) of each row: the mean absolute difference of log R and of m and the count of rows
  with a different letter argmax, for gen-p1-z2.0 (GEN FRESH), p1-z1.5 and p1-z3.0 (FULL FRESH). Also the same values
  for the released v0 of the GEN run against the FULL run (the repeat of one forward). CHECK_V_SIZES.tsv: row, the
  generated size and the released size; CHECK_V.json has the count of rows with a different size.

Usage: python check_pv.py p|gen|v <args>
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stats1  # noqa: E402
import views  # noqa: E402

GEN = "gen-p1-z2.0"
CHECKPOINTS = ("M0", "A1_80")


def letter_values(logp: list[float], gt_index: int) -> tuple[float, float, int]:
    """(log R, m, letter argmax index) of one row."""
    top = max(logp)
    log_m = top + math.log(sum(math.exp(x - top) for x in logp))
    return logp[gt_index] - log_m, math.exp(log_m), stats1.argmax(logp)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in open(path)]


def diff_stats(pairs: list[tuple[tuple, tuple]]) -> dict:
    """pairs of (log R, m, argmax) for (a, b): n, mean, median and max of |a - b| of log R and m, argmax changes."""
    out = {"n": len(pairs)}
    for k, name in ((0, "log_R"), (1, "m")):
        d = [abs(a[k] - b[k]) for a, b in pairs]
        out[name] = {"mean_abs": statistics.fmean(d), "median_abs": statistics.median(d), "max_abs": max(d)}
    out["argmax_changed"] = sum(a[2] != b[2] for a, b in pairs)
    return out


def check_p(argv: list[str]):
    root, dest = Path(argv[0]), Path(argv[1])
    res = {}
    for ck in CHECKPOINTS:
        bf16 = {x["index"]: x for x in lines(root / ck / "HELDOUT_VALUES.jsonl")}
        fp32 = {x["index"]: x for x in lines(root / ck / "linesearch" / "points" / "a0-r1.jsonl")}
        if set(bf16) != set(fp32):
            raise SystemExit(f"STOP: {ck}: {len(set(bf16) ^ set(fp32))} items in one file only")
        res[ck] = {}
        for s in ("A", "B"):
            # log_R and m as each path recorded them; the letter argmax from its letter log probabilities
            pairs = [((x["log_R"], x["m"], stats1.argmax(x["letter_logp"])),
                      (y["log_R"], y["m"], stats1.argmax(y["letter_logp"])))
                     for x, y in ((bf16[i], fp32[i]) for i in sorted(fp32) if fp32[i]["set"] == s)]
            res[ck][s] = diff_stats(pairs)
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def _gen_row(job: dict) -> dict:
    from PIL import Image

    with Image.open(job["student"]) as im:
        image = im.convert("RGB")
    v, crop = views.make_view(image, job["bbox"], job["p1"], 2.0)
    path = Path(job["path"])
    if not path.exists():
        tmp = str(path) + ".tmp.png"
        v.save(tmp)
        os.replace(tmp, path)
    return {"row": job["row"], "path": str(path), "crop": list(crop), "size": list(v.size)}


def gen(argv: list[str]):
    src, p1_file, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    p1 = json.loads(p1_file.read_text())["p1"]
    man = {r["row"]: r for r in lines(src / "MANIFEST.jsonl")}
    removed = {r["row"] for r in json.loads((src / "REMOVED.json").read_text())["removed"]}
    (out / "images" / GEN).mkdir(parents=True, exist_ok=True)
    jobs = [{"row": r, "student": man[r]["student"], "bbox": man[r]["bbox"], "p1": p1,
             "path": str(out / "images" / GEN / f"{r:05d}.png")} for r in sorted(man) if r not in removed]
    with Pool(int(os.environ.get("WORKERS", "32"))) as pool:
        made = pool.map(_gen_row, jobs, chunksize=8)
    with open(out / "MANIFEST.jsonl", "w") as f:
        for m in made:
            rec = man[m["row"]]
            f.write(json.dumps({"row": m["row"], "bbox": rec["bbox"], "student": rec["student"],
                                "views": {views.V0: rec["views"][views.V0],
                                          GEN: {"path": m["path"], "crop": m["crop"], "size": m["size"]}}}) + "\n")
    (out / "REMOVED.json").write_text((src / "REMOVED.json").read_text())
    print(f"gen: {len(made)} rows")


def check_v(argv: list[str]):
    full_p, gen_p, gen_dir, dest, sizes_dest = map(Path, argv)
    full = {x["row"]: x for x in lines(full_p)}
    new = {x["row"]: x for x in lines(gen_p)}
    if set(full) != set(new):
        raise SystemExit(f"STOP: {len(set(full) ^ set(new))} rows in one file only")

    def vals(x: dict, view: str) -> tuple:
        return letter_values(x["logp"][view], x["letters"].index(x["gt"]))

    rows = sorted(full)
    res = {"rows": len(rows)}
    for name, get in ((GEN, lambda r: vals(new[r], GEN)), ("p1-z1.5", lambda r: vals(full[r], "p1-z1.5")),
                      ("p1-z3.0", lambda r: vals(full[r], "p1-z3.0")),
                      ("released_v0_repeat", lambda r: vals(new[r], views.V0))):
        d = diff_stats([(get(r), vals(full[r], views.V0)) for r in rows])
        res[name] = {"mean_abs_log_R": d["log_R"]["mean_abs"], "mean_abs_m": d["m"]["mean_abs"],
                     "argmax_changed": d["argmax_changed"], "max_abs_log_R": d["log_R"]["max_abs"],
                     "max_abs_m": d["m"]["max_abs"]}
    man = {r["row"]: r for r in lines(gen_dir / "MANIFEST.jsonl")}
    diff = 0
    with open(sizes_dest, "w") as f:
        f.write("row\tgen_w\tgen_h\treleased_w\treleased_h\n")
        for r in rows:
            g, rel = man[r]["views"][GEN]["size"], man[r]["views"][views.V0]["size"]
            diff += g != rel
            f.write(f"{r}\t{g[0]}\t{g[1]}\t{rel[0]}\t{rel[1]}\n")
    res["size_different_rows"] = diff
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def main():
    cmd, argv = sys.argv[1], sys.argv[2:]
    {"p": check_p, "gen": gen, "v": check_v}[cmd](argv)


if __name__ == "__main__":
    main()
