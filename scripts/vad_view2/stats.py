"""STATS_VIEW002.json of VLM-VAD-VIEW-002 (HANDOFF-VLM-VAD-VIEW-002-R1 Part A item 6).

Reads, for each arm directory <root>/<arm> (arms B, T8, N1V8, N1R): STEP_METRICS.json (scripts/vad_packfix/metrics.py),
VAL_ACC.json (scripts/vad_packfix/val_acc.py), validation/<step>.jsonl, heldout/<step>/HELDOUT_B.json (heldout.py) and
views_log/assign_<step>.jsonl (hook.py).
1. validation: for each arm and validated step (FULL under R2: 40; the pilot step 2): the fast-validation accuracy (acc_mean of
   VAL_ACC.json), the row count and the letter-first rate (Definition 7: the fraction of rows whose response, after
   the leading white space, starts with an option letter A to F that is not followed by a letter or a digit).
2. comparisons C1 = T8 - B, C2 = N1V8 - N1R, C3 = N1V8 - B at each validated step: the difference of accuracy in
   points (100 x), n_X = rows with acc_X > acc_Y, n_Y = rows with acc_X < acc_Y, and the two-sided exact sign test
   p = min(1, 2 x P(Binomial(n_X + n_Y, 0.5) <= min(n_X, n_Y))). Rows are paired by position (as in VLM-VAD-PACKFIX-001
   DIRECTIVE-010); the script stops when an arm does not have <rows> rows or the row keys differ at a position.
   detected_rule1 (reading rule 1, filled only at the last validated step): |difference| >= 1.5 points and p < 0.05.
3. step_metrics: for each arm and step: rollout_corr/kl, response_length/mean, actor/grad_norm, actor/entropy,
   actor/pg_loss, actor/vopd_loss and rollout_acc (R6 Definition 8), with the source of each value.
4. heldout: for each arm and validated step: n, mean log R, mean m and accuracy of held-out set B.
5. n1r_view_counts: the count of each slot over the N1R assignment files of all steps (and the steps read).
Usage: python stats.py <root> <out json> [<validated steps, comma list> (default 20,40)] [<rows> (default 2636)]
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ARMS = ("B", "T8", "N1V8", "N1R")
COMPARISONS = {"C1": ("T8", "B"), "C2": ("N1V8", "N1R"), "C3": ("N1V8", "B")}
METRICS = ("rollout_corr/kl", "response_length/mean", "actor/grad_norm", "actor/entropy", "actor/pg_loss",
           "actor/vopd_loss", "rollout_acc")
LETTER_FIRST = re.compile(r"^\s*[A-F](?![A-Za-z0-9])")
RULE1_POINTS, RULE1_P = 1.5, 0.05


def sign_test(n_x: int, n_y: int) -> float:
    n = n_x + n_y
    if n == 0:
        return 1.0
    k = min(n_x, n_y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def letter_first(text) -> bool:
    return bool(LETTER_FIRST.match(str(text or "")))


def compare(x: dict, y: dict, rows: int) -> dict:
    """x, y: VAL_ACC step entries (row_acc, row_key, acc_mean, rows)."""
    for v in (x, y):
        if not v["rows"] == len(v["row_acc"]) == len(v["row_key"]) == rows:
            raise SystemExit(f"rows {v['rows']}, row_acc {len(v['row_acc'])}, row_key {len(v['row_key'])}, "
                             f"expected {rows}")
    bad = [i for i, (a, b) in enumerate(zip(x["row_key"], y["row_key"])) if a != b]
    if bad:
        raise SystemExit(f"row keys differ at positions {bad[:20]}")
    n_x = sum(a > b for a, b in zip(x["row_acc"], y["row_acc"]))
    n_y = sum(a < b for a, b in zip(x["row_acc"], y["row_acc"]))
    return {"difference_points": 100 * (x["acc_mean"] - y["acc_mean"]), "n_x_better": n_x, "n_y_better": n_y,
            "sign_test_p": sign_test(n_x, n_y), "paired_rows": rows}


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    steps = [s for s in (sys.argv[3] if len(sys.argv) > 3 else "20,40").split(",")]
    rows = int(sys.argv[4]) if len(sys.argv) > 4 else 2636
    va = {a: json.loads((root / a / "VAL_ACC.json").read_text())["steps"] for a in ARMS}
    sm = {a: json.loads((root / a / "STEP_METRICS.json").read_text())["steps"] for a in ARMS}

    validation = {}
    for a in ARMS:
        validation[a] = {}
        for s in steps:
            dump = [json.loads(x) for x in open(root / a / "validation" / f"{s}.jsonl") if x.strip()]
            lf = sum(letter_first(r.get("output")) for r in dump)
            validation[a][s] = {"acc": va[a][s]["acc_mean"], "rows": va[a][s]["rows"], "letter_first": lf,
                                "letter_first_rate": lf / len(dump), "dump_rows": len(dump)}

    comparisons = {}
    for c, (x, y) in COMPARISONS.items():
        comparisons[c] = {"x": x, "y": y}
        for s in steps:
            r = compare(va[x][s], va[y][s], rows)
            if s == steps[-1]:
                r["detected_rule1"] = abs(r["difference_points"]) >= RULE1_POINTS and r["sign_test_p"] < RULE1_P
            r["letter_first_rate_x_minus_y"] = (validation[x][s]["letter_first_rate"]
                                                - validation[y][s]["letter_first_rate"])
            comparisons[c][s] = r

    step_metrics = {}
    for a in ARMS:
        step_metrics[a] = {}
        for s in sorted((k for k in sm[a] if int(k) >= 1), key=int):
            step_metrics[a][s] = {m: {"value": sm[a][s][m]["value"], "source": sm[a][s][m]["source"]}
                                  for m in METRICS if m in sm[a][s]}

    heldout = {a: {s: json.loads((root / a / "heldout" / s / "HELDOUT_B.json").read_text()) for s in steps}
               for a in ARMS}
    heldout = {a: {s: {k: v[k] for k in ("n", "mean_log_R", "mean_m", "accuracy")} for s, v in d.items()}
               for a, d in heldout.items()}

    counts, read = Counter(), []
    for p in sorted((root / "N1R" / "views_log").glob("assign_*.jsonl"), key=lambda q: int(q.stem.split("_")[1])):
        read.append(int(p.stem.split("_")[1]))
        counts.update(json.loads(x)["slot"] for x in open(p) if x.strip())
    n1r = {"counts": {str(k): counts.get(k, 0) for k in range(1, 9)}, "steps": read}

    res = {"steps": steps, "rows": rows, "validation": validation, "comparisons": comparisons,
           "step_metrics": step_metrics, "heldout": heldout, "n1r_view_counts": n1r,
           "reading_rule1": {"points": RULE1_POINTS, "p": RULE1_P}}
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({c: {s: (round(v[s]["difference_points"], 3), v[s]["sign_test_p"]) for s in steps}
                      for c, v in comparisons.items()}))


if __name__ == "__main__":
    main()
