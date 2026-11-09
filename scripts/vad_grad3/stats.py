"""Statistics of HANDOFF-VLM-VAD-GRAD-003-R2 Part A item 7 from the Gram files and the row files only.

Inputs, for each checkpoint ck in <root>/<ck>/: GRAM_<ck>.json (grads.py gram: names, raw, M1 = <x, D y>,
M2 = <x, D^2 y>) and rows/ROWS_<batch>.jsonl (batches.py).
Metrics: "raw" uses the preconditioner P = I, "D" uses P = D (Definition 8.1). The inner product of two gradients is
<x, P y> (raw or M1), the step direction of a gradient part X is u_X = -P g_X (Definition 8.3), so <u_X, Delta> =
-<x, P Delta> (raw or M1) and ||u_X||^2 = <P x, P x> (raw or M2). ||Delta||^2 is raw in both metrics.
Displacements of a checkpoint (Definition 8.2, the segments that start or end at the step of the checkpoint):
  M0: A1_0_80, A2_0_80. A1_80: all 4. A1_130: A1_80_130, A2_80_130.
Cross-fold quantities (Definition 10): <E x, E y> = the mean over folds a != b of <x(a), y(b)> (for n=1, also the mean
  over the siblings j of fold a and k of fold b). SE = delete-one-fold jackknife sqrt((k-1)/k sum (T_a - Tbar)^2).
Per-fold quantities: the value is the mean over the folds, SE = sample sd over the folds / sqrt(k) (sd/2 for k = 4).
Item 7:
  1. s_X = <E g_X, E g> / ||E g||^2 (n=8; X = C, I, L1, L2, L3), f_X = weight fraction of the rows of X (row files),
     s_X / f_X, d_CI = s_C / f_C - s_I / f_I, cos(E g_C, E g_I). The same for n=1 with the B1 parts C and I (n1_*).
  2. Per fold and displacement: t_X = <u_X, Delta> / <u, Delta>, cos(u_X, Delta), d_CI = t_C / f_C - t_I / f_I.
     For n=1 (A2 displacements): t1_X with the sums over the 8 B1 batches of the fold.
  3. m8 = <E g8, E g8>, N8 = mean ||g8||^2 - m8, SNR8 = m8 / N8; m1, N1, SNR1 from the B1 vectors; SNR8 / SNR1.
  4. W1 = mean over f and j != k of ||g1(f, j) - g1(f, k)||^2 / 2, N1 - W1, ICC_step = (N1 - W1) / N1.
  5. H-DIR per fold and displacement: mean over j of cos(-P w(f, j), Delta), w(f, j) = g1(f, j) - mean_k g1(f, k), and
     of cos(-P g1(f, j), Delta).
  6. SNR8 x 19 and SNR1 x 19.
  7. cos(-P E g8, Delta_A1) and cos(-P E g1, Delta_A2): <E u, Delta> = the mean over folds, ||E u||^2 cross-fold.
  8. V_comp_O, V_comp_Fx = mean over the 6 pairs of {A1 split, R8-f1-s1..s3} of ||g_p - g_q||^2 / 2 (fold 1),
     V_num = ||g_rep - g||^2 / 2, each / N8 (the jackknife varies N8 only).
  9. share_X = <g_X(f), g(f)> / ||g(f)||^2 (per fold), f_X and the row counts.
Differences between checkpoints: for each quantity with an SE, value(b) - value(a) and sqrt(se_a^2 + se_b^2).

Usage: python stats.py <root> <STATS_GRAD3.json> <checkpoint>...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PARTS8 = ("C", "I", "L1", "L2", "L3")
PARTS1 = ("C", "I")
SIBLINGS = 8
DELTAS = ("D__A1_0_80", "D__A1_80_130", "D__A2_0_80", "D__A2_80_130")
RELEVANT = {"M0": ("D__A1_0_80", "D__A2_0_80"), "A1_80": DELTAS, "A1_130": ("D__A1_80_130", "D__A2_80_130")}
COMP = {"O": ["O__B8-f1__total"] + [f"O__R8-f1-s{s}__total" for s in (1, 2, 3)],
        "Fx": ["Fx__B8-f1__total"] + [f"Fx__R8-f1-s{s}__total" for s in (1, 2, 3)]}
REPEAT = ("O__B8-f1@rep__total", "O__B8-f1__total")
NAN = float("nan")


def g8(f: int, part: str = "total") -> str:
    return f"O__B8-f{f}__{part}"


def g1(f: int, j: int, part: str = "total") -> str:
    return f"O__B1-f{f}-j{j}__{part}"


def div(a: float, b: float) -> float:
    return a / b if b != 0 and not math.isnan(a) and not math.isnan(b) else NAN


def sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0 else NAN


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else NAN


class Metric:
    def __init__(self, gram: dict, metric: str):
        self.i = {n: k for k, n in enumerate(gram["names"])}
        self.raw = gram["raw"]
        self.g = gram["M1"] if metric == "D" else gram["raw"]
        self.p = gram["M2"] if metric == "D" else gram["raw"]

    def has(self, *names) -> bool:
        return all(n in self.i for n in names)

    def ip(self, x: str, y: str) -> float:
        """<x, P y>; also <P x, Delta> for a displacement y."""
        return self.g[self.i[x]][self.i[y]]

    def pp(self, x: str, y: str) -> float:
        """<P x, P y>."""
        return self.p[self.i[x]][self.i[y]]

    def dd(self, d: str) -> float:
        return self.raw[self.i[d]][self.i[d]]

    def sq_diff(self, x: str, y: str) -> float:
        return self.ip(x, x) + self.ip(y, y) - 2 * self.ip(x, y)


def cross(fn, folds: list[int], xs, ys) -> float:
    vals = [mean(fn(x, y) for x in xs(a) for y in ys(b)) for a in folds for b in folds if a != b]
    return mean(vals)


def member(row: dict, part: str) -> bool:
    if part in ("C", "I"):
        return row["label"] == (1 if part == "C" else 0)
    return row["len_bin"] == part


def fractions(rows: list[dict], parts) -> dict[str, float]:
    total = sum(r["weight"] for r in rows)
    return {x: div(sum(r["weight"] for r in rows if member(r, x)), total) for x in parts}


class Checkpoint:
    def __init__(self, root: Path, ck: str):
        self.ck = ck
        self.gram = json.loads((root / ck / f"GRAM_{ck}.json").read_text())
        rows_dir = root / ck / "rows"
        self.rows8, self.rows1 = {}, {}
        for p in sorted(rows_dir.glob("ROWS_B8-f*.jsonl")):
            self.rows8[int(p.stem.split("-f")[1])] = [json.loads(x) for x in open(p)]
        for p in sorted(rows_dir.glob("ROWS_B1-f*.jsonl")):
            f = int(p.stem.split("-f")[1].split("-j")[0])
            self.rows1.setdefault(f, []).extend(json.loads(x) for x in open(p))
        names = set(self.gram["names"])
        self.folds = sorted(f for f in self.rows8 if g8(f) in names)
        self.deltas = RELEVANT[ck]

    def rows8_of(self, folds):
        return [r for f in folds for r in self.rows8[f]]

    def rows1_of(self, folds):
        return [r for f in folds for r in self.rows1.get(f, [])]


def cos_parts(q: dict, name: str, num: float, nx: float, ny: float):
    """A cross-fold cos with its numerator and both squared-norm estimates (not clipped; DIRECTIVE-003)."""
    q[f"{name}_num"], q[f"{name}_norm2_x"], q[f"{name}_norm2_y"] = num, nx, ny
    q[name] = div(num, sqrt(nx) * sqrt(ny))


def cross_quantities(c: Checkpoint, m: Metric, folds: list[int]) -> dict:
    q = {}
    one8 = lambda a: [g8(a)]  # noqa: E731
    m8 = cross(m.ip, folds, one8, one8)
    n8 = mean(m.ip(g8(a), g8(a)) for a in folds) - m8
    q.update(m8=m8, N8=n8, SNR8=div(m8, n8), SNR8_x19=div(m8, n8) * 19)
    f8 = fractions(c.rows8_of(folds), PARTS8)
    s = {}
    for x in PARTS8:
        s[x] = div(cross(m.ip, folds, lambda a, x=x: [g8(a, x)], one8), m8)
        q[f"s_{x}"], q[f"f_{x}"], q[f"s_per_f_{x}"] = s[x], f8[x], div(s[x], f8[x])
    q["d_CI"] = div(s["C"], f8["C"]) - div(s["I"], f8["I"])
    cc = cross(m.ip, folds, lambda a: [g8(a, "C")], lambda a: [g8(a, "C")])
    ii = cross(m.ip, folds, lambda a: [g8(a, "I")], lambda a: [g8(a, "I")])
    cos_parts(q, "cos_EgC_EgI", cross(m.ip, folds, lambda a: [g8(a, "C")], lambda a: [g8(a, "I")]), cc, ii)
    for d in c.deltas:
        if d.startswith("D__A1"):
            num = -mean(m.ip(g8(a), d) for a in folds)
            cos_parts(q, f"cos_EuE8_{d}", num, cross(m.pp, folds, one8, one8), m.dd(d))
    if all(m.has(g1(a, j)) for a in folds for j in range(SIBLINGS)):
        all1 = lambda a: [g1(a, j) for j in range(SIBLINGS)]  # noqa: E731
        m1 = cross(m.ip, folds, all1, all1)
        n1 = mean(m.ip(g1(a, j), g1(a, j)) for a in folds for j in range(SIBLINGS)) - m1
        w1 = mean(m.sq_diff(g1(a, j), g1(a, k)) / 2 for a in folds for j in range(SIBLINGS) for k in range(SIBLINGS)
                  if j != k)
        q.update(m1=m1, N1=n1, SNR1=div(m1, n1), SNR1_x19=div(m1, n1) * 19, SNR8_over_SNR1=div(div(m8, n8), div(m1, n1)),
                 W1=w1, N1_minus_W1=n1 - w1, ICC_step=div(n1 - w1, n1))
        f1 = fractions(c.rows1_of(folds), PARTS1)
        s1 = {}
        for x in PARTS1:
            s1[x] = div(cross(m.ip, folds, lambda a, x=x: [g1(a, j, x) for j in range(SIBLINGS)], all1), m1)
            q[f"n1_s_{x}"], q[f"n1_f_{x}"], q[f"n1_s_per_f_{x}"] = s1[x], f1[x], div(s1[x], f1[x])
        q["n1_d_CI"] = div(s1["C"], f1["C"]) - div(s1["I"], f1["I"])
        for d in c.deltas:
            if d.startswith("D__A2"):
                num = -mean(m.ip(g1(a, j), d) for a in folds for j in range(SIBLINGS))
                cos_parts(q, f"cos_EuE1_{d}", num, cross(m.pp, folds, all1, all1), m.dd(d))
    for mode, names in COMP.items():
        if m.has(*names):
            pairs = [(p, r) for k, p in enumerate(names) for r in names[k + 1:]]
            q[f"V_comp_{mode}"] = div(mean(m.sq_diff(p, r) / 2 for p, r in pairs), n8)
    if m.has(*REPEAT):
        q["V_num"] = div(m.sq_diff(*REPEAT) / 2, n8)
    if "V_comp_O" in q and "V_comp_Fx" in q:
        q["V_comp_O_minus_Fx"] = q["V_comp_O"] - q["V_comp_Fx"]
    return q


def fold_quantities(c: Checkpoint, m: Metric, f: int) -> dict:
    q = {}
    tot = g8(f)
    fr = fractions(c.rows8[f], PARTS8)
    for x in PARTS8:
        q[f"share_{x}"] = div(m.ip(g8(f, x), tot), m.ip(tot, tot))
    for d in c.deltas:
        dd = m.dd(d)
        base = m.ip(tot, d)
        q[f"cos_u_total_{d}"] = div(-base, sqrt(m.pp(tot, tot)) * sqrt(dd))
        t = {}
        for x in PARTS8:
            t[x] = div(m.ip(g8(f, x), d), base)
            q[f"t_{x}_{d}"] = t[x]
            q[f"cos_u_{x}_{d}"] = div(-m.ip(g8(f, x), d), sqrt(m.pp(g8(f, x), g8(f, x))) * sqrt(dd))
        q[f"d_CI_{d}"] = div(t["C"], fr["C"]) - div(t["I"], fr["I"])
        if not all(m.has(g1(f, j)) for j in range(SIBLINGS)):
            continue
        js = range(SIBLINGS)
        if d.startswith("D__A2"):
            f1 = fractions(c.rows1.get(f, []), PARTS1)
            base1 = sum(m.ip(g1(f, j), d) for j in js)
            t1 = {x: div(sum(m.ip(g1(f, j, x), d) for j in js), base1) for x in PARTS1}
            for x in PARTS1:
                q[f"n1_t_{x}_{d}"] = t1[x]
            q[f"n1_d_CI_{d}"] = div(t1["C"], f1["C"]) - div(t1["I"], f1["I"])
        pd = [m.ip(g1(f, j), d) for j in js]
        pd_mean = mean(pd)
        pp_mean = mean(m.pp(g1(f, k), g1(f, l)) for k in js for l in js)
        cw, cg = [], []
        for j in js:
            w2 = m.pp(g1(f, j), g1(f, j)) - 2 * mean(m.pp(g1(f, j), g1(f, k)) for k in js) + pp_mean
            cw.append(div(-(pd[j] - pd_mean), sqrt(w2) * sqrt(dd)))
            cg.append(div(-pd[j], sqrt(m.pp(g1(f, j), g1(f, j))) * sqrt(dd)))
        q[f"hdir_cos_w_{d}"] = mean(cw)
        q[f"hdir_cos_g1_{d}"] = mean(cg)
    return q


def jackknife(values: list[float]) -> float:
    if len(values) < 2 or any(math.isnan(v) for v in values):
        return NAN
    k = len(values)
    mu = sum(values) / k
    return math.sqrt((k - 1) / k * sum((v - mu) ** 2 for v in values))


def sd_mean(values: list[float]) -> float:
    if len(values) < 2 or any(math.isnan(v) for v in values):
        return NAN
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1) / len(values))


def with_errors(c: Checkpoint, m: Metric) -> dict:
    folds = c.folds
    full = cross_quantities(c, m, folds)
    jack = [cross_quantities(c, m, [b for b in folds if b != a]) for a in folds]
    out = {}
    for key, value in full.items():
        dels = [j.get(key, NAN) for j in jack]
        out[key] = {"value": value, "se": jackknife(dels), "se_rule": "jackknife", "folds": len(folds),
                    "delete_one": dels}
        if f"{key}_norm2_x" in full:
            out[key]["norm_nonpositive"] = bool(full[f"{key}_norm2_x"] <= 0 or full[f"{key}_norm2_y"] <= 0)
    per = [fold_quantities(c, m, f) for f in folds]
    for key in per[0]:
        vals = [p.get(key, NAN) for p in per]
        out[key] = {"value": mean(vals), "se": sd_mean(vals), "se_rule": "sd/sqrt(k)", "folds": len(folds),
                    "per_fold": vals}
    return out


def row_counts(c: Checkpoint) -> dict:
    out = {"folds": c.folds, "displacements": list(c.deltas), "n8": {}, "n1": {}}
    for f in c.folds:
        r8, r1 = c.rows8[f], c.rows1.get(f, [])
        out["n8"][str(f)] = {"rows": len(r8), **{x: sum(member(r, x) for r in r8) for x in PARTS8},
                             "weight_fraction": fractions(r8, PARTS8)}
        out["n1"][str(f)] = {"rows": len(r1), **{x: sum(member(r, x) for r in r1) for x in PARTS1},
                             "weight_fraction": fractions(r1, PARTS1)}
    return out


def differences(res: dict, cks: list[str]) -> dict:
    out = {}
    for a, b in [(cks[i], cks[j]) for i in range(len(cks)) for j in range(i + 1, len(cks))]:
        for metric in ("raw", "D"):
            qa, qb = res[a]["metrics"][metric], res[b]["metrics"][metric]
            for key in sorted(set(qa) & set(qb)):
                diff = qb[key]["value"] - qa[key]["value"]
                se = sqrt(qa[key]["se"] ** 2 + qb[key]["se"] ** 2) if not (
                    math.isnan(qa[key]["se"]) or math.isnan(qb[key]["se"])) else NAN
                out.setdefault(f"{b}-{a}", {}).setdefault(metric, {})[key] = {"value": diff, "se": se}
    return out


def checks(d: Path) -> dict:
    """part_sum_rel, rounding_rel and part_checks of each key (grads.py progress files, DIRECTIVE-VLM-VAD-GRAD-003-005,
    R3 Part A item 3.6)."""
    out = {}
    for p in sorted(d.glob("progress_*.json")):
        state = json.loads(p.read_text())
        out[p.stem[len("progress_"):]] = {k: state.get(k) for k in ("part_sum_rel", "rounding_rel", "part_checks")}
    return out


def compute(root: Path, cks: list[str]) -> dict:
    res = {}
    for ck in cks:
        c = Checkpoint(root, ck)
        res[ck] = {"rows": row_counts(c), "metric_file": c.gram.get("metric"), "checks": checks(root / ck),
                   "metrics": {metric: with_errors(c, Metric(c.gram, metric)) for metric in ("raw", "D")}}
    return {"checkpoints": res, "differences": differences(res, cks)}


def null_if_undefined(x):
    """NaN and infinity become null (undefined values, DIRECTIVE-003); the file is strict JSON."""
    if isinstance(x, dict):
        return {k: null_if_undefined(v) for k, v in x.items()}
    if isinstance(x, list):
        return [null_if_undefined(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write(res: dict, out: Path):
    Path(out).write_text(json.dumps(null_if_undefined(res), indent=1, allow_nan=False))


def main():
    root, out, *cks = sys.argv[1:]
    write(compute(Path(root), cks), Path(out))


if __name__ == "__main__":
    main()
