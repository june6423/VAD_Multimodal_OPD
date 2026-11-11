"""Gate G2 of HANDOFF-VLM-VAD-VIEW-001-R2: g(B8(1), v0) at M0 from the new code and from the GRAD-003 code.

old: scripts/vad_grad3/grads.py run O, key B8-f1@g2 (the GRAD-003 code at 50801aa6, total only).
new: scripts/vad_view/vgrads.py, key B8-f1@p1-z2.0 (x-plus = the v0 image of the views manifest).
G2.json: update_path.compare(new, old) (rel_diff = ||new - old|| / ||old||, cos, fp64) and pass = rel_diff <= 1e-2 and
cos >= 0.999. Exit 3 when the gate fails.

Usage: python vg2.py <old vector .pt> <new vector .pt> <G2.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vad_grad3"))
REL_MAX, COS_MIN = 1e-2, 0.999
FAIL_EXIT = 3


def gate(cmp: dict) -> bool:
    return cmp["rel_diff"] <= REL_MAX and cmp["cos"] >= COS_MIN


def main():
    import torch
    import update_path

    old, new, dest = sys.argv[1:]
    a, b = torch.load(new, mmap=True), torch.load(old, mmap=True)
    if a.shape != b.shape:
        raise SystemExit(f"STOP: the vectors differ in shape: {tuple(a.shape)} and {tuple(b.shape)}")
    cmp = update_path.compare(a, b)
    res = {"old": old, "new": new, **cmp, "rel_max": REL_MAX, "cos_min": COS_MIN, "pass": gate(cmp),
           "bit_equal": bool(torch.equal(a, b))}
    Path(dest).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if not res["pass"]:
        sys.exit(FAIL_EXIT)


if __name__ == "__main__":
    main()
