"""Statistics of the packed-forward comparison (HANDOFF-VLM-VAD-PACK-001-R3 Part A item 6, Definitions 9 to 13).

Reads only PACKROWS_<batch>.jsonl and GRAM_PACK.json of one directory. Writes STATS_PACK.json.
Group: the batch name without its index (A1-S0, A1-S80, A2-S0). Position class: "1" (position 1) and "2+" (position 2
or later). Token values: d_abs of each row (|d_t| at each response token); KL sums: kl_student_sum, kl_teacher_sum over
kl_tokens.
Interval (Definition 13): 95 percent percentile bootstrap over the micro-batches of the group (resample with replacement,
a fresh generator with seed 20261106 for each quantity). A token or row mean is a ratio of sums over the resampled
micro-batches. A percentile of |d_t| is np.percentile (linear) of the token values of the resampled micro-batches, each
repeated by its multiplicity.
Gradient quantities (Definitions 9 to 11) come from the Gram matrix: ||e||^2 = G_PP - 2 G_PU + G_UU and so on. They are
batch-level values without a resampling unit: "ci" is null and "count" gives the batches or batch pairs.

Usage: python stats.py <dir> <out json> [resamples (10000)]
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20261106
CLASSES = ("1", "2+")


def counts_matrix(n: int, resamples: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(resamples, n))
    flat = idx + np.arange(resamples)[:, None] * n
    return np.bincount(flat.ravel(), minlength=resamples * n).reshape(resamples, n).astype(np.float64)


def interval(values: np.ndarray) -> list[float] | None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def ratio(num: list[float], den: list[float], resamples: int) -> dict:
    """sum(num) / sum(den) over the micro-batches, with its bootstrap interval."""
    a, b = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if len(a) == 0 or b.sum() == 0:
        return {"value": None, "ci": None, "count": float(b.sum()), "micro_batches": len(a)}
    c = counts_matrix(len(a), resamples)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (c @ a) / (c @ b)
    return {"value": float(a.sum() / b.sum()), "ci": interval(boot), "count": float(b.sum()), "micro_batches": len(a)}


def percentile(token_lists: list[np.ndarray], q: float, resamples: int) -> dict:
    """q-th percentile of the pooled token values of the micro-batches, with its bootstrap interval."""
    keep = [t for t in token_lists]
    vals = np.concatenate(keep) if keep else np.zeros(0)
    if len(vals) == 0:
        return {"value": None, "ci": None, "count": 0, "micro_batches": len(keep)}
    owner = np.concatenate([np.full(len(t), i) for i, t in enumerate(keep)])
    c = counts_matrix(len(keep), resamples).astype(np.int64)
    boot = np.empty(resamples)
    for k in range(resamples):
        rep = c[k][owner]
        pooled = np.repeat(vals, rep)
        boot[k] = np.percentile(pooled, q) if len(pooled) else np.nan
    return {"value": float(np.percentile(vals, q)), "ci": interval(boot), "count": int(len(vals)), "micro_batches": len(keep)}


def load_rows(d: Path) -> dict[str, list[dict]]:
    out = {}
    for p in sorted(d.glob("PACKROWS_*.jsonl")):
        out[p.stem[len("PACKROWS_"):]] = [json.loads(line) for line in open(p)]
    return out


def group_of(batch: str) -> str:
    return batch.rsplit("-", 1)[0]


def by_micro(rows: list[dict]) -> dict[tuple, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[(r["batch"], r["rank"], r["mini"], r["micro"])].append(r)
    return dict(sorted(out.items()))


def cls(r: dict) -> str:
    return "1" if r["position"] == 1 else "2+"


def packing(micro: dict, resamples: int) -> dict:
    """Item 6.1."""
    per = list(micro.values())
    dist = defaultdict(int)
    for rs in per:
        dist[len(rs)] += 1
    return {"micro_batches": len(per), "rows": sum(len(rs) for rs in per),
            "rows_per_micro_batch": {str(k): v for k, v in sorted(dist.items())},
            "share_rows_position_2plus": ratio([sum(r["position"] >= 2 for r in rs) for rs in per],
                                               [len(rs) for rs in per], resamples),
            "share_response_tokens_position_2plus": ratio(
                [sum(r["response_tokens"] for r in rs if r["position"] >= 2) for rs in per],
                [sum(r["response_tokens"] for r in rs) for rs in per], resamples)}


def token_errors(micro: dict, resamples: int) -> dict:
    """Items 6.2 and 6.3 for each position class."""
    out = {}
    per = list(micro.values())
    for c in CLASSES:
        sel = [[r for r in rs if cls(r) == c] for rs in per]
        sel = [rs for rs in sel if rs]
        out[c] = {
            "d_abs_mean": ratio([sum(sum(r["d_abs"]) for r in rs) for rs in sel],
                                [sum(len(r["d_abs"]) for r in rs) for rs in sel], resamples),
            "d_abs_p99": percentile([np.asarray([x for r in rs for x in r["d_abs"]], dtype=np.float64) for rs in sel],
                                    99.0, resamples),
            "kl_student_mean": ratio([sum(r["kl_student_sum"] for r in rs) for rs in sel],
                                     [sum(r["kl_tokens"] for r in rs) for rs in sel], resamples),
            "kl_teacher_mean": ratio([sum(r["kl_teacher_sum"] for r in rs) for rs in sel],
                                     [sum(r["kl_tokens"] for r in rs) for rs in sel], resamples),
            "L_rel_diff_mean": ratio([sum(abs(r["L_P"] - r["L_U"]) / max(1.0, abs(r["L_U"])) for r in rs) for rs in sel],
                                     [len(rs) for rs in sel], resamples),
        }
    return out


def rollout_kl(micro: dict, resamples: int) -> dict:
    """Item 6.7 (R3): rollout_corr/kl of each arm, the A1 formula masked_mean(rollout_log_prob - log_prob,
    response_mask) (rollout_corr_helper.py 889) over the pooled response tokens of the group, for each position class
    and for all rows ("all", the micro-batches as A1 had them). Descriptive."""
    out = {}
    per = list(micro.values())
    for c in (*CLASSES, "all"):
        sel = [[r for r in rs if c == "all" or cls(r) == c] for rs in per]
        sel = [rs for rs in sel if rs]
        out[c] = {arm: ratio([sum(r[f"rkl_{arm}_sum"] for r in rs) for rs in sel],
                             [sum(r["kl_tokens"] for r in rs) for rs in sel], resamples) for arm in "PU"}
    return out


def gradients(gram: dict) -> dict:
    """Items 6.4 to 6.6 from the Gram matrix."""
    names = gram["names"]
    g = np.asarray(gram["gram"], dtype=np.float64)
    ix = {n: i for i, n in enumerate(names)}
    batches = sorted({n.rsplit("_", 1)[0] for n in names if f"{n.rsplit('_', 1)[0]}_P" in ix and f"{n.rsplit('_', 1)[0]}_U" in ix})

    def dot(a: str, b: str) -> float:
        return float(g[ix[a], ix[b]])

    def e_dot(b1: str, b2: str) -> float:
        return dot(f"{b1}_P", f"{b2}_P") - dot(f"{b1}_P", f"{b2}_U") - dot(f"{b1}_U", f"{b2}_P") + dot(f"{b1}_U", f"{b2}_U")

    def sqrt(x: float) -> float:
        return math.sqrt(x) if x >= 0 else math.nan

    per_batch = {}
    for b in batches:
        np_, nu = sqrt(dot(f"{b}_P", f"{b}_P")), sqrt(dot(f"{b}_U", f"{b}_U"))
        ne = sqrt(e_dot(b, b))
        per_batch[b] = {"norm_P": np_, "norm_U": nu, "norm_e": ne, "eps": ne / nu, "cos": dot(f"{b}_P", f"{b}_U") / (np_ * nu),
                        "norm_ratio": np_ / nu}
    groups = defaultdict(list)
    for b in batches:
        groups[group_of(b)].append(b)
    per_group = {}
    for grp, bs in sorted(groups.items()):
        pairs = list(itertools.combinations(bs, 2))
        noise = [sqrt(dot(f"{a}_U", f"{a}_U") - 2 * dot(f"{a}_U", f"{b}_U") + dot(f"{b}_U", f"{b}_U")) for a, b in pairs]
        n_grp = float(np.mean(noise)) if pairs else None
        c_e = [e_dot(a, b) / (per_batch[a]["norm_e"] * per_batch[b]["norm_e"]) for a, b in pairs]
        for b in bs:
            per_batch[b]["rho"] = per_batch[b]["norm_e"] / n_grp if n_grp else None
        rhos = [per_batch[b]["rho"] for b in bs]
        per_group[grp] = {"batches": bs, "pairs": len(pairs), "n_grp": {"value": n_grp, "ci": None, "count": len(pairs)},
                          "pair_noise": noise, "c_e": {"value": float(np.mean(c_e)) if pairs else None, "ci": None,
                                                       "count": len(pairs)}, "pair_error_cos": c_e,
                          "eps_mean": {"value": float(np.mean([per_batch[b]["eps"] for b in bs])), "ci": None, "count": len(bs)},
                          "rho_mean": {"value": float(np.mean(rhos)) if None not in rhos else None, "ci": None,
                                       "count": len(bs)}}
    diff = None
    if "A1-S0" in per_group and "A2-S0" in per_group:
        diff = {"value": per_group["A1-S0"]["eps_mean"]["value"] - per_group["A2-S0"]["eps_mean"]["value"], "ci": None,
                "count": [len(per_group["A1-S0"]["batches"]), len(per_group["A2-S0"]["batches"])]}
    diag = {n: {"gram": dot(n, n), "separate": gram["sq_norm_separate"].get(n)} for n in names}
    return {"per_batch": per_batch, "per_group": per_group, "eps_mean_A1_S0_minus_A2_S0": diff, "diagonal_check": diag}


def compute(d: Path, resamples: int) -> dict:
    rows = load_rows(d)
    groups = defaultdict(list)
    for b, rs in rows.items():
        groups[group_of(b)] += rs
    out = {"resamples": resamples, "seed": SEED, "batches": {b: len(rs) for b, rs in rows.items()}, "groups": {}}
    for grp, rs in sorted(groups.items()):
        micro = by_micro(rs)
        out["groups"][grp] = {"packing": packing(micro, resamples), "token_errors": token_errors(micro, resamples),
                              "rollout_kl": rollout_kl(micro, resamples)}
    gram = json.loads((d / "GRAM_PACK.json").read_text())
    out["gradients"] = gradients(gram)
    return out


def main():
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    resamples = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    res = compute(d, resamples)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({g: v["packing"]["micro_batches"] for g, v in res["groups"].items()}))


if __name__ == "__main__":
    main()
