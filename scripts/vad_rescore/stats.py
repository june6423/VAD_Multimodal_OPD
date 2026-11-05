"""Statistics of the rescoring (HANDOFF-VLM-VAD-GRAD-002-R1 Part A item 5, Definitions 7 to 11).

Reads only SCORES_<set>_<model>.jsonl and FIRSTTOK_<model>.jsonl of one directory, and the TRAJ.jsonl files of
TASK-VLM-VAD-GRAD-001-013 for the self-score agreement. Writes STATS_RESCORE.json.

Delta_M(y) = X_M(y) - X_ref(y) for X in L, L/response_tokens, L1, Lc. Reference: M0 on S0; A1_80 for A1_130 on S80;
A2_80 for A2_130 on S80. For one form, the trajectories with a finite Delta count. A contrast prompt has at least one
correct and one incorrect counted sibling. d_p = mean Delta of its correct siblings - mean Delta of its incorrect
siblings, D = mean of d_p over the contrast prompts. Short: response_tokens <= 5, long: > 5.
Interval: 95 percent percentile bootstrap over the prompts of the quantity (resample with replacement, the same
resample for both terms of a paired quantity), RESAMPLES resamples, a new generator with seed 20261105 for each
quantity. A trajectory mean (item 5.4) and a correlation (item 5.7) are ratios of sums over the resampled prompts.
R_gt = P_gt / P_letter, defined for a prompt with P_letter >= 0.01 under both models of a pair.

Usage: python stats.py <score dir> <TRAJ.jsonl step0> <TRAJ.jsonl step80> <out json> [resamples (10000)]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20261105
SHORT_MAX = 5
R_GT_MIN = 0.01
SELF_TOL = 1e-3
MODELS_S0 = ("A1_80", "A1_130", "A2_80", "A2_130")
D_PAIRS = [(m, "S0", "M0") for m in MODELS_S0] + [("A1_130", "S80", "A1_80"), ("A2_130", "S80", "A2_80")]
FIRSTTOK_PAIRS = [(m, "M0") for m in MODELS_S0] + [("A1_130", "A1_80"), ("A2_130", "A2_80")]
ARM_PAIRS = [("A1_80", "A2_80", "S0"), ("A1_130", "A2_130", "S0"), ("A1_130", "A2_130", "S80")]
CORR_PAIRS = [("A1_80", "A2_80"), ("A1_130", "A2_130"), ("A1_80", "A1_130")]
FORMS = {"L": lambda r: r["L"], "L_per_token": lambda r: r["L"] / r["response_tokens"], "L1": lambda r: r["L1"],
         "Lc": lambda r: r["Lc"]}


def counts_matrix(n: int, resamples: int) -> np.ndarray:
    """(resamples, n) multiplicities of n items in each resample (fresh generator, seed SEED)."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(resamples, n))
    flat = idx + np.arange(resamples)[:, None] * n
    return np.bincount(flat.ravel(), minlength=resamples * n).reshape(resamples, n).astype(np.float64)


def interval(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def mean_over_prompts(per_prompt: list[float], resamples: int) -> dict:
    """Mean of per-prompt values with its bootstrap interval over the prompts."""
    x = np.asarray(per_prompt, dtype=np.float64)
    if len(x) == 0:
        return {"value": None, "ci": None, "count": 0}
    c = counts_matrix(len(x), resamples)
    return {"value": float(x.mean()), "ci": interval(c @ x / len(x)), "count": len(x)}


def ratio_over_prompts(num: list[float], den: list[float], resamples: int) -> dict:
    """sum(num) / sum(den) over the prompts, resampling prompts."""
    a, b = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if len(a) == 0:
        return {"value": None, "ci": None, "count": 0}
    c = counts_matrix(len(a), resamples)
    return {"value": float(a.sum() / b.sum()), "ci": interval((c @ a) / (c @ b)), "prompts": len(a), "count": float(b.sum())}


def load_scores(d: Path) -> dict:
    out = {}
    for path in sorted(d.glob("SCORES_*_*.jsonl")):
        _, s, m = path.stem.split("_", 2)
        out[(s, m)] = {r["traj"]: r for r in map(json.loads, open(path))}
    return out


def deltas(scores: dict, s: str, m: str, ref: str, form: str) -> dict[int, float]:
    a, b = scores[(s, m)], scores[(s, ref)]
    f = FORMS[form]
    out = {}
    for t, r in a.items():
        v = f(r) - f(b[t])
        if math.isfinite(v):
            out[t] = v
    return out


def per_prompt_d(delta: dict[int, float], meta: dict[int, dict], keep=lambda r: True) -> dict[int, float]:
    """d_p of each contrast prompt over the kept trajectories with a Delta."""
    groups = defaultdict(lambda: {1: [], 0: []})
    for t, v in delta.items():
        if keep(meta[t]):
            groups[meta[t]["prompt"]][meta[t]["label"]].append(v)
    return {p: sum(g[1]) / len(g[1]) - sum(g[0]) / len(g[0]) for p, g in sorted(groups.items()) if g[1] and g[0]}


def is_short(r: dict) -> bool:
    return r["response_tokens"] <= SHORT_MAX


def d_quantities(scores: dict, resamples: int) -> dict:
    """Items 5.1 to 5.4 and 5.6."""
    out = {"D": {}, "D_forms": {}, "D_length": {}, "mean_delta": {}, "D_arm_difference": {}}
    for m, s, ref in D_PAIRS:
        key = f"{m}|{s}"
        meta = scores[(s, ref)]
        delta = deltas(scores, s, m, ref, "L")
        out["D"][key] = {**mean_over_prompts(list(per_prompt_d(delta, meta).values()), resamples), "reference": ref}
        out["D_forms"][key] = {form: mean_over_prompts(list(per_prompt_d(deltas(scores, s, m, ref, form), meta).values()),
                                                       resamples) for form in ("L_per_token", "L1", "Lc")}
        out["D_length"][key] = {g: mean_over_prompts(list(per_prompt_d(delta, meta, keep).values()), resamples)
                                for g, keep in (("short", is_short), ("long", lambda r: not is_short(r)))}
        cells = {}
        for label, cls in ((1, "C"), (0, "I")):
            for g, keep in (("all", lambda r: True), ("short", is_short), ("long", lambda r: not is_short(r))):
                sums = defaultdict(lambda: [0.0, 0])
                for t, v in delta.items():
                    if meta[t]["label"] == label and keep(meta[t]):
                        sums[meta[t]["prompt"]][0] += v
                        sums[meta[t]["prompt"]][1] += 1
                vals = [sums[p] for p in sorted(sums)]
                cells[f"{cls}|{g}"] = ratio_over_prompts([v[0] for v in vals], [v[1] for v in vals], resamples)
        out["mean_delta"][key] = cells
    for a, b, s in ARM_PAIRS:
        ref_a = "M0" if s == "S0" else "A1_80"
        ref_b = "M0" if s == "S0" else "A2_80"
        meta = scores[(s, ref_a)]
        da = per_prompt_d(deltas(scores, s, a, ref_a, "L"), meta)
        db = per_prompt_d(deltas(scores, s, b, ref_b, "L"), meta)
        common = sorted(set(da) & set(db))
        out["D_arm_difference"][f"{a}-{b}|{s}"] = mean_over_prompts([da[p] - db[p] for p in common], resamples)
    return out


def firsttok_quantities(d: Path, resamples: int) -> dict:
    """Item 5.5: paired changes of the mean P_letter, P_gt, P_wrong and R_gt."""
    ft = {}
    for path in sorted(d.glob("FIRSTTOK_*.jsonl")):
        ft[path.stem.split("_", 1)[1]] = {r["prompt"]: r for r in map(json.loads, open(path))}
    out = {"excluded_prompts": {m: sorted(p for p, r in rows.items() if r["cause"]) for m, rows in ft.items()}}
    for m, ref in FIRSTTOK_PAIRS:
        a, b = ft[m], ft[ref]
        prompts = sorted(p for p in a if not a[p]["cause"] and p in b and not b[p]["cause"])
        res = {}
        for q in ("P_letter", "P_gt", "P_wrong"):
            res[q] = {**mean_over_prompts([a[p][q] - b[p][q] for p in prompts], resamples),
                      "mean_model": float(np.mean([a[p][q] for p in prompts])) if prompts else None,
                      "mean_reference": float(np.mean([b[p][q] for p in prompts])) if prompts else None}
        rp = [p for p in prompts if a[p]["P_letter"] >= R_GT_MIN and b[p]["P_letter"] >= R_GT_MIN]
        ra = {p: a[p]["P_gt"] / a[p]["P_letter"] for p in rp}
        rb = {p: b[p]["P_gt"] / b[p]["P_letter"] for p in rp}
        res["R_gt"] = {**mean_over_prompts([ra[p] - rb[p] for p in rp], resamples),
                       "mean_model": float(np.mean(list(ra.values()))) if rp else None,
                       "mean_reference": float(np.mean(list(rb.values()))) if rp else None,
                       "prompts_defined": len(rp), "prompts_with_letters": len(prompts)}
        out[f"{m}-{ref}"] = res
    return out


def correlations(scores: dict, resamples: int) -> dict:
    """Item 5.7: Pearson correlation of two Delta_L over the S0 trajectories, bootstrap over prompts."""
    meta = scores[("S0", "M0")]
    out = {}
    for a, b in CORR_PAIRS:
        da, db = deltas(scores, "S0", a, "M0", "L"), deltas(scores, "S0", b, "M0", "L")
        sums = defaultdict(lambda: np.zeros(6))
        for t in sorted(set(da) & set(db)):
            x, y = da[t], db[t]
            sums[meta[t]["prompt"]] += np.array([1.0, x, y, x * x, y * y, x * y])
        s = np.array([sums[p] for p in sorted(sums)])

        def pearson(tot):
            n, sx, sy, sxx, syy, sxy = (tot[..., i] for i in range(6))
            return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))

        c = counts_matrix(len(s), resamples)
        out[f"{a}~{b}"] = {"value": float(pearson(s.sum(0))), "ci": interval(pearson(c @ s)),
                           "count": int(s[:, 0].sum()), "prompts": len(s)}
    return out


def self_scores(scores: dict, traj_paths: dict[str, str]) -> dict:
    """Item 5.8: L against logp_sum of TRAJ.jsonl (TASK-VLM-VAD-GRAD-001-013)."""
    out = {}
    for s, m in (("S0", "M0"), ("S80", "A1_80")):
        ref = {r["traj"]: r["logp_sum"] for r in map(json.loads, open(traj_paths[s]))}
        rows = scores[(s, m)]
        ab = [abs(r["L"] - ref[t]) for t, r in rows.items()]
        rel = [abs(r["L"] - ref[t]) / max(1.0, abs(ref[t])) for t, r in rows.items()]
        worst = max(rows, key=lambda t: abs(rows[t]["L"] - ref[t]) / max(1.0, abs(ref[t])))
        out[f"{m}|{s}"] = {"count": len(rows), "max_abs": max(ab), "max_rel": max(rel), "tolerance": SELF_TOL,
                           "within_tolerance": sum(x <= SELF_TOL for x in rel), "worst_traj": worst,
                           "worst": {"L": rows[worst]["L"], "logp_sum": ref[worst]}}
    return out


def compute(d: Path, traj_paths: dict[str, str], resamples: int) -> dict:
    scores = load_scores(d)
    counts = {f"{s}|{m}": {"trajectories": len(v), "prompts": len({r["prompt"] for r in v.values()}),
                          "Lc_nan": sum(1 for r in v.values() if not math.isfinite(r["Lc"]))}
              for (s, m), v in sorted(scores.items())}
    return {"resamples": resamples, "seed": SEED, "counts": counts, **d_quantities(scores, resamples),
            "first_token": firsttok_quantities(d, resamples), "correlation": correlations(scores, resamples),
            "self_score": self_scores(scores, traj_paths)}


def main():
    d, t0, t80, out = sys.argv[1:5]
    resamples = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    result = compute(Path(d), {"S0": t0, "S80": t80}, resamples)
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("counts", "self_score")}))


if __name__ == "__main__":
    main()
