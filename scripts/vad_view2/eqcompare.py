"""Equality check of DIRECTIVE-VLM-VAD-VIEW-002-002 Decision 2: this commit against f09a2ac0 (the code of the N1R
outputs of TASK-VLM-VAD-VIEW-002-005), for each of the arms T8, N1V8 and N1R.

check_view2.sh trains step 1 from M0 on the same replayed rollout rows (g3hook.py) with each code: <new>/<arm> (this
commit) and <old>/<arm> (git archive of f09a2ac0). For each arm:
loss_rel = |L_new - L_old| / |L_old| for actor/pg_loss (the rule) and actor/vopd_loss (reported),
cosine of the two step-1 gradients (g3compare.cosine), equal gradient layouts, and equal teacher rows
(views_log/teacher_1.jsonl: position, slot, image path and image sha256 of each row; the uid is random in each run).
An arm passes when loss_rel(actor/pg_loss) <= 1e-3, cosine >= 0.999, the layouts are equal and the teacher rows are equal.
Usage: python eqcompare.py <new dir> <old dir> <out json> <arm> [<arm> ...]. Exit 0 when each arm passes, else 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.vad_view2.g3compare import COS_MIN, LOSS_TOL, cosine, step1


def teacher_rows(d: Path) -> list[tuple]:
    rows = [json.loads(x) for x in open(d / "views_log" / "teacher_1.jsonl") if x.strip()]
    return [(r["position"], r["slot"], r["path"], r["sha256"]) for r in rows]


def compare(new: Path, old: Path) -> dict:
    mn, mo = step1(new), step1(old)
    losses = {}
    for k in ("actor/pg_loss", "actor/vopd_loss", "actor/grad_norm"):
        if k in mn and k in mo:
            losses[k] = {"new": mn[k], "old": mo[k], "rel": abs(mn[k] - mo[k]) / abs(mo[k]) if mo[k] else None}
    i_new = json.loads((new / "grad.bin.json").read_text())
    i_old = json.loads((old / "grad.bin.json").read_text())
    layout_equal = [(x["name"], x["numel"]) for x in i_new["params"]] == [(x["name"], x["numel"]) for x in i_old["params"]]
    grad = cosine(new / "grad.bin", old / "grad.bin", i_new["elements"]) if layout_equal else None
    t_new, t_old = teacher_rows(new), teacher_rows(old)
    pg = losses.get("actor/pg_loss", {})
    res = {"losses": losses, "layout_equal": layout_equal, "gradient": grad, "teacher_rows": len(t_new),
           "teacher_rows_equal": t_new == t_old}
    res["pass"] = bool(pg.get("rel") is not None and pg["rel"] <= LOSS_TOL and grad and grad["cosine"] >= COS_MIN
                       and layout_equal and t_new == t_old and len(t_new) > 0)
    return res


def main() -> int:
    new, old, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    arms = {a: compare(new / a, old / a) for a in sys.argv[4:]}
    res = {"arms": arms, "loss_tol": LOSS_TOL, "cos_min": COS_MIN, "pass": bool(arms) and all(a["pass"] for a in arms.values())}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
