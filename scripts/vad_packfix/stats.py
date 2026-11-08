"""STATS_PACKFIX.json of VLM-VAD-PACKFIX-001 (HANDOFF-VLM-VAD-PACKFIX-001-R2 Part A item 6).

Reads only <F>/STEP_METRICS.json, <B>/STEP_METRICS.json, <F>/VAL_ACC.json, <B>/VAL_ACC.json and the A1 reference
scripts/vad_packfix/sources/A1_STEP_METRICS_1_20.json.
1. For each step and arm: each metric of Definition 7 with its source, and F minus B.
2. For each arm: the fast-validation accuracy of each validated step with the row count; F minus B; a paired sign
   test over the rows of the step: n_F = rows with acc_F > acc_B, n_B = rows with acc_F < acc_B, two-sided exact
   binomial p with probability 0.5 over n_F + n_B rows. The rows are paired by position in the dump order
   (DIRECTIVE-VLM-VAD-PACKFIX-001-010): the row key of val_acc.py (sha256 of input and gts) is not unique. The script
   stops with an error when a step does not have <rows> rows in each arm, or when the row keys (input or gts) of the
   arms differ at a position. duplicate_row_keys = rows minus the count of distinct row keys of arm F.
3. For each arm: the mean rollout accuracy (HANDOFF-VLM-VAD-PACKFIX-001-R6 Definition 8) over steps 1-10 and over
   steps 11-20 (the steps that exist), with the source of each step value.
4. Arm B next to A1 for each metric and step (descriptive). A1 has no rollout accuracy and no entropy.
Usage: python stats.py <F dir> <B dir> <A1 reference json> <out json> [rows, default 2636]
"""

import json
import math
import sys
from pathlib import Path

METRICS = ["rollout_corr/kl", "rollout_acc","response_length/mean", "actor/grad_norm", "actor/entropy",
           "actor/pg_loss", "actor/vopd_loss"]


def sign_test(n_f: int, n_b: int):
    n = n_f + n_b
    if n == 0:
        return 1.0
    k = min(n_f, n_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    f_dir, b_dir, a1_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    n_rows = int(sys.argv[5]) if len(sys.argv) > 5 else 2636
    sm = {a: json.loads((d / "STEP_METRICS.json").read_text())["steps"] for a, d in (("F", f_dir), ("B", b_dir))}
    va = {a: json.loads((d / "VAL_ACC.json").read_text())["steps"] for a, d in (("F", f_dir), ("B", b_dir))}
    a1 = json.loads(a1_path.read_text())["steps"]

    steps = sorted({int(s) for a in sm for s in sm[a] if int(s) >= 1})
    per_step = {}
    for s in steps:
        row = {}
        for m in METRICS:
            f, b = sm["F"].get(str(s), {}).get(m), sm["B"].get(str(s), {}).get(m)
            row[m] = {"F": f["value"] if f else None, "B": b["value"] if b else None,
                      "F_source": f["source"] if f else None, "B_source": b["source"] if b else None,
                      "F_minus_B": (f["value"] - b["value"]) if f and b else None}
        per_step[str(s)] = row

    validation = {}
    for s in sorted(set(va["F"]) & set(va["B"]), key=int):
        vf, vb = va["F"][s], va["B"][s]
        for arm, v in (("F", vf), ("B", vb)):
            if not v["rows"] == len(v["row_acc"]) == len(v["row_key"]) == n_rows:
                sys.exit(f"step {s} arm {arm}: rows {v['rows']}, row_acc {len(v['row_acc'])}, "
                         f"row_key {len(v['row_key'])}, expected {n_rows}")
        bad = [i for i, (kf, kb) in enumerate(zip(vf["row_key"], vb["row_key"])) if kf != kb]
        if bad:
            sys.exit(f"step {s}: input or gts differ between the arms at positions {bad}")
        pairs = list(zip(vf["row_acc"], vb["row_acc"]))
        n_f = sum(f > b for f, b in pairs)
        n_b = sum(f < b for f, b in pairs)
        validation[s] = {
            "F": {"acc": vf["acc_mean"], "rows": vf["rows"], "by_benchmark": vf["by_benchmark"]},
            "B": {"acc": vb["acc_mean"], "rows": vb["rows"], "by_benchmark": vb["by_benchmark"]},
            "F_minus_B_points": 100 * (vf["acc_mean"] - vb["acc_mean"]),
            "paired_rows": len(pairs), "duplicate_row_keys": n_rows - len(set(vf["row_key"])),
            "position_mismatches": len(bad), "n_F_better": n_f, "n_B_better": n_b, "sign_test_p": sign_test(n_f, n_b),
        }

    rollout_acc = {}
    for a in ("F", "B"):
        rollout_acc[a] = {}
        for name, lo, hi in (("steps_1_10", 1, 10), ("steps_11_20", 11, 20)):
            vals = [(s, sm[a][str(s)]["rollout_acc"]) for s in range(lo, hi + 1)
                    if "rollout_acc" in sm[a].get(str(s), {})]
            rollout_acc[a][name] = {"mean": sum(v["value"] for _, v in vals) / len(vals) if vals else None,
                                    "steps": [s for s, _ in vals], "sources": sorted({v["source"] for _, v in vals})}

    b_vs_a1 = {}
    for s in steps:
        b_vs_a1[str(s)] = {m: {"B": per_step[str(s)][m]["B"], "A1": a1.get(str(s), {}).get(m)} for m in METRICS}

    res = {"steps": steps, "per_step": per_step, "validation": validation, "rollout_acc_means": rollout_acc,
           "arm_B_vs_A1": b_vs_a1, "a1_reference": str(a1_path)}
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({"steps": steps, "validation": {s: (v["F"]["acc"], v["B"]["acc"], v["sign_test_p"])
                                                     for s, v in validation.items()}}))


if __name__ == "__main__":
    main()
