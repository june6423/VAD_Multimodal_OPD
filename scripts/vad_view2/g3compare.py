"""Gate G3 of HANDOFF-VLM-VAD-VIEW-002-R1: the VIEW-002 code (arm B) against the PACKFIX code (VAD_PACKFIX=0).

Both check runs train step 1 from M0 on the same 768 replayed rollout rows (g3hook.py). Reads, in each run directory:
metrics_*.jsonl (the verl file logger, step 1), grad.bin and grad.bin.json (g3hook.py gradient dump). The VIEW-002 run
also has views_log/teacher_1.jsonl (hook.py): each row must have slot 5 and its released teacher image (v0).
loss_rel = |L_view2 - L_packfix| / |L_packfix| for actor/pg_loss (the gate) and actor/vopd_loss (reported);
cosine = <g_view2, g_packfix> / (|g_view2| |g_packfix|) over all gradient elements (fp64 sums over fp32 values).
G3 passes when loss_rel(actor/pg_loss) <= 1e-3, cosine >= 0.999, the gradient layouts are equal and each row has v0.
Usage: python g3compare.py <view2 run dir> <packfix run dir> <out json>. Exit 0 = pass, 1 = fail.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

LOSS_TOL, COS_MIN = 1e-3, 0.999
CHUNK = 1 << 24


def step1(d: Path) -> dict:
    data = {}
    for p in sorted(d.glob("metrics_*.jsonl")):
        for line in p.open():
            if line.strip():
                rec = json.loads(line)
                if int(rec["step"]) == 1:
                    data.update(rec["data"])
    return data


def cosine(a_path: Path, b_path: Path, n: int) -> dict:
    a = np.memmap(a_path, dtype=np.float32, mode="r", shape=(n,))
    b = np.memmap(b_path, dtype=np.float32, mode="r", shape=(n,))
    dot = aa = bb = 0.0
    for i in range(0, n, CHUNK):
        x, y = a[i:i + CHUNK].astype(np.float64), b[i:i + CHUNK].astype(np.float64)
        dot += float(x @ y)
        aa += float(x @ x)
        bb += float(y @ y)
    return {"cosine": dot / math.sqrt(aa * bb), "norm_view2": math.sqrt(aa), "norm_packfix": math.sqrt(bb),
            "rel_norm_diff": abs(math.sqrt(aa) - math.sqrt(bb)) / math.sqrt(bb)}


def main():
    v, p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mv, mp = step1(v), step1(p)
    losses = {}
    for k in ("actor/pg_loss", "actor/vopd_loss", "actor/grad_norm", "rollout_corr/kl"):
        if k in mv and k in mp:
            losses[k] = {"view2": mv[k], "packfix": mp[k],
                         "rel": abs(mv[k] - mp[k]) / abs(mp[k]) if mp[k] else None}
    iv = json.loads((v / "grad.bin.json").read_text())
    ip = json.loads((p / "grad.bin.json").read_text())
    layout_equal = [(x["name"], x["numel"]) for x in iv["params"]] == [(x["name"], x["numel"]) for x in ip["params"]]
    grad = cosine(v / "grad.bin", p / "grad.bin", iv["elements"]) if layout_equal else None
    rows = [json.loads(x) for x in open(v / "views_log" / "teacher_1.jsonl") if x.strip()]
    keys = {}
    for x in (json.loads(y) for y in open(v / "views_log" / "assign_1.jsonl") if y.strip()):
        keys[x["uid"]] = x["key"]
    v0 = sum(r["slot"] == 5 and r["path"] == keys.get(r["uid"]) for r in rows)
    pg = losses.get("actor/pg_loss", {})
    res = {"losses": losses, "layout_equal": layout_equal, "elements": iv["elements"], "gradient": grad,
           "rows": len(rows), "rows_v0": v0, "loss_tol": LOSS_TOL, "cos_min": COS_MIN}
    res["pass"] = bool(pg.get("rel") is not None and pg["rel"] <= LOSS_TOL and grad and grad["cosine"] >= COS_MIN
                       and layout_equal and len(rows) == 768 and v0 == len(rows))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
