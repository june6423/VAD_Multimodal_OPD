"""Pilot subset and FULL projection (HANDOFF-VLM-VAD-PACK-001-R1 Part A item 8, gate G3).

select: for each group (A1-S0, A1-S80, A2-S0), over its 3 batches: the micro-batch with the most rows, and the
  micro-batch with the longest row (total tokens). When one micro-batch is both, the second is the micro-batch with the
  most rows among the others. Ties go to the lowest (batch, rank, mini, micro).
check: arm P against arm U at position 1 (G3: |L_P - L_U| <= 1e-3 x max(1, |L_U|)) and, when recorded, arm U against
  the one-row forward (L_1row, O4_1row) with the same tolerance.
project: the projected FULL wall time from the pilot records:
  micro-batches: least squares of seconds (inputs + arm P + arm U) on [1, tokens] over the pilot micro-batches, applied
    to every micro-batch of the 9 batches;
  model loads: the largest pilot load time x 2 (one process for M0, one for A1_80);
  batch ends: the largest pilot batch-end time x 9;
  commits: one commit for each COMMIT_SECONDS of the micro-batch time, at the largest pilot commit time;
  Gram: the pilot Gram seconds x (18 / pilot vectors)^2;
  statistics: the pilot stats seconds x (10000 / pilot resamples) x (FULL rows / pilot rows).
Usage: python pilot.py select <split dir> <out json>
       python pilot.py check <pilot out dir> <out json>
       python pilot.py project <pilot out dir> <split dir> <stats seconds> <pilot resamples> <commit seconds> <out json>
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

LIMIT_H = 16.0
TOL = 1e-3
FULL_VECTORS = 18


def load_split(d: Path) -> dict[tuple, list[dict]]:
    micro = defaultdict(list)
    for p in sorted(d.glob("ROWS_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            micro[(r["batch"], r["rank"], r["mini"], r["micro"])].append(r)
    return micro


def select(micro: dict[tuple, list[dict]]) -> list[list]:
    groups = defaultdict(list)
    for k in sorted(micro):
        groups[k[0].rsplit("-", 1)[0]].append(k)
    chosen = []
    for _, keys in sorted(groups.items()):
        most = sorted(keys, key=lambda k: (-len(micro[k]), k))[0]
        longest = sorted(keys, key=lambda k: (-max(r["total_tokens"] for r in micro[k]), k))[0]
        if longest == most:
            longest = sorted((k for k in keys if k != most), key=lambda k: (-len(micro[k]), k))[0]
        chosen += [list(most), list(longest)]
    return chosen


def check(d: Path) -> dict:
    rows = [json.loads(line) for p in sorted(d.glob("PACKROWS_*.jsonl")) for line in open(p)]
    pos1 = [r for r in rows if r["position"] == 1]
    bad_p = [r["traj"] for r in pos1 if abs(r["L_P"] - r["L_U"]) > TOL * max(1.0, abs(r["L_U"]))]
    one = [r for r in rows if "L_1row" in r]
    bad_u = [r["traj"] for r in one
             if abs(r["L_U"] - r["L_1row"]) > TOL * max(1.0, abs(r["L_1row"]))
             or abs(r["O4_U"] - r["O4_1row"]) > TOL * max(1.0, abs(r["O4_1row"]))]
    return {"rows": len(rows), "position_1_rows": len(pos1), "position_1_outside": bad_p,
            "position_1_max_rel": max((abs(r["L_P"] - r["L_U"]) / max(1.0, abs(r["L_U"])) for r in pos1), default=None),
            "one_row_rows": len(one), "one_row_outside": bad_u,
            "one_row_max_rel_L": max((abs(r["L_U"] - r["L_1row"]) / max(1.0, abs(r["L_1row"])) for r in one), default=None),
            "pass": not bad_p and not bad_u and len(pos1) > 0 and len(one) == len(rows)}


def lstsq(x: list[list[float]], y: list[float]) -> list[float]:
    import numpy as np

    return np.linalg.lstsq(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rcond=None)[0].tolist()


def project(d: Path, split_dir: Path, stats_seconds: float, resamples: int, commit_seconds: float) -> dict:
    prog = [json.loads(p.read_text()) for p in sorted(d.glob("progress_*.json"))]
    timing = [m for p in prog for m in p["micro"]]
    secs = [m["seconds_inputs"] + m["seconds_P"] + m["seconds_U"] for m in timing]
    coef = lstsq([[1.0, m["tokens"]] for m in timing], secs)
    full = load_split(split_dir)
    micro_s = sum(max(0.0, coef[0] + coef[1] * sum(r["total_tokens"] for r in rs)) for rs in full.values())
    runs = [json.loads(line) for line in open(d / "runs.jsonl")]
    ends = [res.get("seconds_end", 0.0) for r in runs for res in r.get("results", [])]
    commits = [res.get("seconds_commit_max", 0.0) for r in runs for res in r.get("results", [])]
    gram = json.loads((d / "GRAM_PACK.json").read_text())
    pilot_rows = sum(len(p["rows"]) for p in prog)
    full_rows = sum(len(rs) for rs in full.values())
    parts = {"micro_batches_s": micro_s,
             "model_loads_s": 2 * max(r.get("load_seconds", 0.0) for r in runs),
             "batch_ends_s": 9 * max(ends, default=0.0),
             "commits_s": math.ceil(micro_s / commit_seconds) * max(commits, default=0.0),
             "gram_s": gram["seconds"] * (FULL_VECTORS / len(gram["names"])) ** 2,
             "stats_s": stats_seconds * (10000 / resamples) * (full_rows / max(1, pilot_rows))}
    total_h = sum(parts.values()) / 3600
    return {"coefficients_seconds": {"const": coef[0], "per_token": coef[1]}, "pilot_micro_batches": len(timing),
            "full_micro_batches": len(full), "parts_s": parts, "total_h": total_h, "limit_h": LIMIT_H,
            "within_limit": total_h <= LIMIT_H, "peak_gpu_gb": max(r.get("peak_gpu_gb", 0.0) for r in runs)}


def main():
    cmd = sys.argv[1]
    if cmd == "select":
        chosen = select(load_split(Path(sys.argv[2])))
        Path(sys.argv[3]).write_text(json.dumps(chosen))
        print(f"pilot micro-batches {chosen}")
    elif cmd == "check":
        res = check(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
    elif cmd == "project":
        d, split_dir, seconds, resamples, commit_s, out = sys.argv[2:8]
        res = project(Path(d), Path(split_dir), float(seconds), int(resamples), float(commit_s))
        Path(out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
