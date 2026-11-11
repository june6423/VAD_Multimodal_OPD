"""STATS_VIEW1.json of HANDOFF-VLM-VAD-VIEW-001-R1 Part A item 4, from the stage 1 files only.

Inputs (stage1.py):
  FRESH.jsonl: one line per kept training row: row, gt, letters, logp = {view: [log p(letter) for letters]} at the first
    response token of the teacher input with the view.
  PREFIX_step<s>.jsonl: one line per trajectory of step s: traj, prompt_pos, correct, letter (student letter, "" when none),
    gt, flag ("ok" when the teacher-check position rule gives the position and a token for each option letter), logp
    (as above, at the answer-letter position; null when flag is not "ok").
Definitions (Definition 6 and 12): letter mass m = sum of the letter probabilities; letter argmax = the letter with the
  highest probability (the first in letter order at a tie); the ensemble letter distribution = the mean over the views
  of the log letter probabilities (normalizing does not change its argmax).
Item 4:
  1. accuracy[v] = mean of (letter argmax = gt), mass[v] = mean letter mass, on the FRESH rows.
  2. phi[v|w] = the phi correlation of the error indicators of views v and w (null when a margin is 0).
  3. union_accuracy (some view has argmax = gt), ensemble_accuracy, union_minus_v0, ensemble_minus_v0.
  4. v0_wrong_ens_right, v0_right_ens_wrong, sign_test_p (exact two-sided binomial, p = 1/2).
  5. For each step: correction rate on incorrect lettered student rows (correct false, letter not "", flag "ok") =
     mean of (teacher letter argmax = gt), for each view, the union and the ensemble, and the differences union - v0
     and ensemble - v0. SE: prompt-cluster bootstrap (the prompts with at least one counted row drawn with replacement,
     2000 draws, numpy default_rng(1); a draw gives the ratio of the summed counts).

Usage: python stats1.py <stage1 dir> <STATS_VIEW1.json> <step>...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

V0 = "p1-z2.0"
DRAWS = 2000
SEED = 1


def argmax(xs: list[float]) -> int:
    best = 0
    for i, x in enumerate(xs):
        if x > xs[best]:
            best = i
    return best


def right(letters: list[str], logp: list[float], gt: str) -> bool:
    return letters[argmax(logp)] == gt


def ensemble(logps: dict[str, list[float]], views: list[str]) -> list[float]:
    return [sum(logps[v][i] for v in views) / len(views) for i in range(len(logps[views[0]]))]


def phi(x: list[int], y: list[int]) -> float | None:
    n11 = sum(a and b for a, b in zip(x, y))
    n10 = sum(a and not b for a, b in zip(x, y))
    n01 = sum(b and not a for a, b in zip(x, y))
    n00 = len(x) - n11 - n10 - n01
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return (n11 * n00 - n10 * n01) / math.sqrt(den) if den > 0 else None


def sign_test(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fresh_stats(rows: list[dict]) -> dict:
    views = sorted(rows[0]["logp"])
    ok = {v: [right(r["letters"], r["logp"][v], r["gt"]) for r in rows] for v in views}
    ens = [right(r["letters"], ensemble(r["logp"], views), r["gt"]) for r in rows]
    union = [any(ok[v][i] for v in views) for i in range(len(rows))]
    n = len(rows)
    out = {"rows": n, "views": views,
           "accuracy": {v: sum(ok[v]) / n for v in views},
           "mass": {v: sum(sum(math.exp(x) for x in r["logp"][v]) for r in rows) / n for v in views},
           "phi": {f"{v}|{w}": phi([not x for x in ok[v]], [not x for x in ok[w]])
                   for i, v in enumerate(views) for w in views[i + 1:]},
           "union_accuracy": sum(union) / n, "ensemble_accuracy": sum(ens) / n}
    out["union_minus_v0"] = out["union_accuracy"] - out["accuracy"][V0]
    out["ensemble_minus_v0"] = out["ensemble_accuracy"] - out["accuracy"][V0]
    b = sum((not a) and e for a, e in zip(ok[V0], ens))
    c = sum(a and not e for a, e in zip(ok[V0], ens))
    out.update(v0_wrong_ens_right=b, v0_right_ens_wrong=c, sign_test_p=sign_test(b, c))
    return out


def counted(r: dict) -> bool:
    return (not r["correct"]) and r["letter"] != "" and r["flag"] == "ok"


def prefix_stats(rows: list[dict]) -> dict:
    use = [r for r in rows if counted(r)]
    flags = {}
    for r in rows:
        if (not r["correct"]) and r["letter"] != "":
            flags[r["flag"]] = flags.get(r["flag"], 0) + 1
    out = {"trajectories": len(rows), "incorrect_lettered": sum(flags.values()), "flags_incorrect_lettered": flags,
           "counted": len(use)}
    if not use:
        return out
    views = sorted(use[0]["logp"])
    hit = {v: np.array([right(r["letters"], r["logp"][v], r["gt"]) for r in use], float) for v in views}
    hit["union"] = np.array([any(hit[v][i] for v in views) for i in range(len(use))], float)
    hit["ensemble"] = np.array([right(r["letters"], ensemble(r["logp"], views), r["gt"]) for r in use], float)
    prompts = sorted({r["prompt_pos"] for r in use})
    place = {p: i for i, p in enumerate(prompts)}
    cl = np.array([place[r["prompt_pos"]] for r in use])
    sums = {k: np.bincount(cl, weights=h, minlength=len(prompts)) for k, h in hit.items()}
    size = np.bincount(cl, minlength=len(prompts)).astype(float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(prompts), size=(DRAWS, len(prompts)))
    w = np.stack([np.bincount(d, minlength=len(prompts)) for d in draws]).astype(float)
    n_b = w @ size
    rate_b = {k: (w @ s) / n_b for k, s in sums.items()}
    rate = {k: float(h.mean()) for k, h in hit.items()}
    out.update(prompts=len(prompts), draws=DRAWS, seed=SEED,
               rate={k: {"value": rate[k], "se": float(rate_b[k].std(ddof=1))} for k in hit})
    for k in ("union", "ensemble"):
        out[f"{k}_minus_v0"] = {"value": rate[k] - rate[V0], "se": float((rate_b[k] - rate_b[V0]).std(ddof=1))}
    return out


def compute(stage1: Path, steps: list[str]) -> dict:
    fresh = [json.loads(x) for x in open(stage1 / "FRESH.jsonl")]
    res = {"v0": V0, "fresh": fresh_stats(fresh), "prefix": {}}
    for s in steps:
        rows = [json.loads(x) for x in open(stage1 / f"PREFIX_step{s}.jsonl")]
        res["prefix"][f"step{s}"] = prefix_stats(rows)
    return res


def null_if_undefined(x):
    if isinstance(x, dict):
        return {k: null_if_undefined(v) for k, v in x.items()}
    if isinstance(x, list):
        return [null_if_undefined(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main():
    stage1, out, *steps = sys.argv[1:]
    res = compute(Path(stage1), steps)
    Path(out).write_text(json.dumps(null_if_undefined(res), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
