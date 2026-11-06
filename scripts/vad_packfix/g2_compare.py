"""G2 of HANDOFF-VLM-VAD-PACKFIX-001-R1: arm P with the fix against arm U on the pilot micro-batches.

Reads PACKROWS_*.jsonl of the fixed run and of the reference run (TASK-VLM-VAD-PACK-001-002, arm P without the fix).
For each row: abs = |L_P - L_U|, rel = abs / max(1, |L_U|). G2 passes when each rel is at most 1e-3.
The reference columns show the arm P value without the fix and check that arm U did not change.
Usage: python g2_compare.py <fixed pack dir> <reference pack dir> <out json>. Exit 0 = pass, 1 = fail.
"""

import json
import sys
from pathlib import Path

TOL = 1e-3


def load(d: Path) -> dict:
    rows = {}
    for p in sorted(d.glob("PACKROWS_*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            rows[(r["batch"], r["traj"])] = r
    return rows


def main():
    fixed, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows, ref_rows = load(fixed), load(ref)
    table = []
    for key, r in sorted(rows.items()):
        a = abs(r["L_P"] - r["L_U"])
        rel = a / max(1.0, abs(r["L_U"]))
        q = ref_rows.get(key)
        table.append({
            "batch": r["batch"], "traj": r["traj"], "rank": r["rank"], "micro": r["micro"], "position": r["position"],
            "response_tokens": r["response_tokens"], "L_P": r["L_P"], "L_U": r["L_U"], "abs": a, "rel": rel,
            "d_max": r["d_max"], "kl_student_mean": r["kl_student_sum"] / r["kl_tokens"],
            "kl_teacher_mean": r["kl_teacher_sum"] / r["kl_tokens"],
            "ref_L_P": q["L_P"] if q else None, "ref_L_U": q["L_U"] if q else None,
            "ref_rel": abs(q["L_P"] - q["L_U"]) / max(1.0, abs(q["L_U"])) if q else None,
            "L_U_equal_ref": (q["L_U"] == r["L_U"]) if q else None,
            "ref_kl_teacher_mean": q["kl_teacher_sum"] / q["kl_tokens"] if q else None,
        })
    worst_abs = max(table, key=lambda t: t["abs"])
    worst_rel = max(table, key=lambda t: t["rel"])
    outside = [(t["batch"], t["traj"]) for t in table if t["rel"] > TOL]
    result = {
        "rows": len(table), "reference_rows": len(ref_rows), "rows_in_reference": sum(t["ref_L_P"] is not None for t in table),
        "tolerance": TOL, "outside": outside, "max_abs": worst_abs["abs"], "max_abs_row": [worst_abs["batch"], worst_abs["traj"]],
        "max_rel": worst_rel["rel"], "max_rel_row": [worst_rel["batch"], worst_rel["traj"]],
        "positions": sorted({t["position"] for t in table}),
        "L_U_equal_reference": sum(bool(t["L_U_equal_ref"]) for t in table),
        "reference_max_rel": max((t["ref_rel"] for t in table if t["ref_rel"] is not None), default=None),
        "pass": not outside and len(table) > 0, "table": table,
    }
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "table"}))
    for t in table:
        print(f"{t['batch']} traj {t['traj']} pos {t['position']} ntok {t['response_tokens']} L_P {t['L_P']:.6f} "
              f"L_U {t['L_U']:.6f} rel {t['rel']:.3e} | ref L_P {t['ref_L_P']} ref rel {t['ref_rel']} "
              f"teacher KL {t['kl_teacher_mean']:.3e} ref {t['ref_kl_teacher_mean']}")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
