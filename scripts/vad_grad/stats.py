"""Statistics from one Gram file (HANDOFF-VLM-VAD-GRAD-001-R1 Part A item 6; DIRECTIVE-VLM-VAD-GRAD-001-002 D1-D4).

The Gram file (moments.py) holds, for one checkpoint: the 28 accumulator names, their counts, the 28 x 28 dot
products raw and with D, and for each prompt its fold, siblings, labels, flags and the sibling Gram raw and with D.
This module reads only that file.

Subsets: pooled, C, I, each flag (f1, f0, fx) and each class x flag. For a subset X and a fold set F:
  N, S2 = ||S_X||^2, PP = sum_p ||P_p,X||^2, G2 = sum_i ||g_i||^2, K2 = sum_p k_p^2, KK = sum_p k_p (k_p - 1).
  E2 = G2 / N, m = (S2 - PP) / (N^2 - K2), m_all = (S2 - G2) / (N (N - 1)), s_same = (PP - G2) / KK,
  sigma_b^2 = s_same - m, sigma_w^2 = E2 - s_same, sigma^2 = E2 - m, q = m / E2, ICC = sigma_b^2 / sigma^2,
  B = sigma^2 / m, c_n = m / (m + sigma^2 / n), SNR_A2 = 96 m / (sigma_b^2 + sigma_w^2), SNR_A1 = 96 m /
  (sigma_b^2 + sigma_w^2 / 8). q_in = s_same / E2 over the mixed prompts only (class and class x flag subsets).
Pair cosines inside mixed prompts: cc, ii, ci for the pooled and each flag view. "exact" = mean over prompts of the
  mean pair cosine of the prompt (A8 method); "weighted" = sum of pair dots / sum of ||g_i|| ||g_j|| (option a).
Cross-fold quantities use only pairs of different folds: <mu_X, mu_Y> = sum_{a != b} <S_X(a), S_Y(b)> /
  sum_{a != b} N_X(a) N_Y(b), the same for g_acc with A(b) and the prompt count of fold b.
  mu2_cf, cos(mu_C, mu_I), gacc2_cf, a_X = <-mu_X, g_acc> / sqrt(mu2_cf(X) gacc2_cf) (HANDOFF-VLM-VAD-GRAD-001-R2
  Correction item 1: -mu_X is the step direction, so a_X > 0 means that a step on X raises the expected accuracy).
Standard errors: the value uses all folds. One-fold quantities: sd(Q_1..Q_4) / 2 with Q_a from fold a only (sample
  sd), and also the delete-one-fold jackknife. Cross-fold quantities: the jackknife sqrt(3/4 sum_a (T_a - Tbar)^2),
  where T_a uses the other 3 folds.
Usage: python stats.py <GRAM.json> <STATS.json>
"""

from __future__ import annotations

import json
import math
import sys
from typing import Callable

CLASSES = ("C", "I")
FLAGS = ("f1", "f0", "fx")
C_N = (1, 4, 16, 96, 768)
NAN = float("nan")


def subsets() -> dict[str, Callable[[str, str], bool]]:
    out = {"pooled": lambda c, f: True}
    for c in CLASSES:
        out[c] = lambda cc, f, c=c: cc == c
    for f in FLAGS:
        out[f] = lambda c, ff, f=f: ff == f
    for c in CLASSES:
        for f in FLAGS:
            out[f"{c}|{f}"] = lambda cc, ff, c=c, f=f: cc == c and ff == f
    return out


def prompt_class(labels: list[int]) -> str:
    n_c = sum(labels)
    n_i = len(labels) - n_c
    if n_c >= 2 and n_i >= 2:
        return "mixed"
    if n_i == 0:
        return "all_correct"
    if n_c == 0:
        return "all_wrong"
    return "other"


def _div(a: float, b: float) -> float:
    return a / b if b != 0 and not math.isnan(a) and not math.isnan(b) else NAN


def _sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0 else NAN


class Gram:
    def __init__(self, data: dict):
        self.data = data
        self.k = data["k"]
        self.names = data["names"]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.counts = data["counts"]
        self.prompts = data["prompts"]
        for p in self.prompts:
            p["class"] = prompt_class(p["labels"])
            p["cls"] = ["C" if x else "I" for x in p["labels"]]

    def s_vecs(self, sel, folds) -> list[str]:
        return [f"S|{a}|{c}|{f}" for a in folds for c in CLASSES for f in FLAGS if sel(c, f)]

    def block(self, metric: str, us: list[str], vs: list[str]) -> float:
        g = self.data[metric]
        return sum(g[self.index[u]][self.index[v]] for u in us for v in vs)

    def count(self, names: list[str]) -> int:
        return sum(self.counts.get(n, 0) for n in names)


def one_fold(gr: Gram, metric: str, sel, folds: list[int]) -> dict:
    """Moment quantities of subset sel over the folds (D2)."""
    vecs = gr.s_vecs(sel, folds)
    n = gr.count(vecs)
    s2 = gr.block(metric, vecs, vecs)
    pp = g2 = k2 = kk = 0.0
    n_check = 0
    mix = {"pp": 0.0, "g2": 0.0, "n": 0, "kk": 0.0}
    for p in gr.prompts:
        if p["fold"] not in folds:
            continue
        idx = [i for i, (c, f) in enumerate(zip(p["cls"], p["flags"])) if sel(c, f)]
        if not idx:
            continue
        m = p[metric]
        pp_p = sum(m[i][j] for i in idx for j in idx)
        g2_p = sum(m[i][i] for i in idx)
        k = len(idx)
        pp, g2, k2, kk, n_check = pp + pp_p, g2 + g2_p, k2 + k * k, kk + k * (k - 1), n_check + k
        if p["class"] == "mixed":
            mix["pp"] += pp_p
            mix["g2"] += g2_p
            mix["n"] += k
            mix["kk"] += k * (k - 1)
    if n != n_check:
        raise ValueError(f"accumulator count {n} differs from the sibling count {n_check}")
    e2 = _div(g2, n)
    m_ = _div(s2 - pp, n * n - k2)
    s_same = _div(pp - g2, kk)
    sb2, sw2, s2tot = s_same - m_, e2 - s_same, e2 - m_
    out = {"N": n, "E2": e2, "m": m_, "m_all": _div(s2 - g2, n * (n - 1)), "s_same": s_same,
           "sigma_b2": sb2, "sigma_w2": sw2, "sigma2": s2tot, "q": _div(m_, e2), "ICC": _div(sb2, s2tot),
           "B": _div(s2tot, m_), "SNR_A2": _div(96 * m_, sb2 + sw2), "SNR_A1": _div(96 * m_, sb2 + sw2 / 8),
           "q_in": _div(_div(mix["pp"] - mix["g2"], mix["kk"]), _div(mix["g2"], mix["n"])),
           "same_pair_share": _div(kk, n * (n - 1))}
    for c in C_N:
        out[f"c_{c}"] = _div(m_, m_ + s2tot / c)
    return out


def pair_cosines(gr: Gram, metric: str, view, folds: list[int]) -> dict:
    """Pair cosines inside mixed prompts for one flag view: cc, ii, ci, exact and weighted (D3)."""
    acc = {k: {"prompt_means": [], "dot": 0.0, "normprod": 0.0, "pairs": 0} for k in ("cc", "ii", "ci")}
    for p in gr.prompts:
        if p["fold"] not in folds or p["class"] != "mixed":
            continue
        m = p[metric]
        idx = {c: [i for i, (cc, f) in enumerate(zip(p["cls"], p["flags"])) if cc == c and view(c, f)] for c in CLASSES}
        pairs = {"cc": [(i, j) for a, i in enumerate(idx["C"]) for j in idx["C"][a + 1:]],
                 "ii": [(i, j) for a, i in enumerate(idx["I"]) for j in idx["I"][a + 1:]],
                 "ci": [(i, j) for i in idx["C"] for j in idx["I"]]}
        for key, plist in pairs.items():
            if not plist:
                continue
            cos = [m[i][j] / math.sqrt(m[i][i] * m[j][j]) for i, j in plist]
            acc[key]["prompt_means"].append(sum(cos) / len(cos))
            acc[key]["dot"] += sum(m[i][j] for i, j in plist)
            acc[key]["normprod"] += sum(math.sqrt(m[i][i] * m[j][j]) for i, j in plist)
            acc[key]["pairs"] += len(plist)
    out = {}
    for key, a in acc.items():
        pm = a["prompt_means"]
        out[f"{key}_exact"] = sum(pm) / len(pm) if pm else NAN
        out[f"{key}_weighted"] = _div(a["dot"], a["normprod"])
        out[f"{key}_prompts"] = len(pm)
        out[f"{key}_pairs"] = a["pairs"]
    return out


def cross_fold(gr: Gram, metric: str, folds: list[int]) -> dict:
    """Cross-fold norms, cos(mu_C, mu_I) per flag view, and the alignments a_X (6.2, 6.5)."""
    sets = subsets()

    def inner(sel_x, sel_y):
        num = den = 0.0
        for a in folds:
            for b in folds:
                if a == b:
                    continue
                vx, vy = gr.s_vecs(sel_x, [a]), gr.s_vecs(sel_y, [b])
                num += gr.block(metric, vx, vy)
                den += gr.count(vx) * gr.count(vy)
        return _div(num, den)

    def inner_acc(sel_x):
        num = den = 0.0
        for a in folds:
            for b in folds:
                if a != b:
                    vx = gr.s_vecs(sel_x, [a])
                    num += gr.block(metric, vx, [f"A|{b}"])
                    den += gr.count(vx) * gr.counts.get(f"A|{b}", 0)
        return _div(num, den)

    num = den = 0.0
    for a in folds:
        for b in folds:
            if a != b:
                num += gr.block(metric, [f"A|{a}"], [f"A|{b}"])
                den += gr.counts.get(f"A|{a}", 0) * gr.counts.get(f"A|{b}", 0)
    gacc2 = _div(num, den)
    out = {"gacc2_cf": gacc2}
    mu2 = {}
    for name, sel in sets.items():
        mu2[name] = inner(sel, sel)
        out[f"mu2_cf|{name}"] = mu2[name]
        out[f"a|{name}"] = _div(-inner_acc(sel), _sqrt(mu2[name] * gacc2))  # R2: <-mu_X, g_acc>
    for view in ("pooled",) + FLAGS:
        c_key = "C" if view == "pooled" else f"C|{view}"
        i_key = "I" if view == "pooled" else f"I|{view}"
        out[f"cos_mu_C_mu_I|{view}"] = _div(inner(sets[c_key], sets[i_key]), _sqrt(mu2[c_key] * mu2[i_key]))
    return out


def _sd_half(values: list[float]) -> float:
    if any(math.isnan(v) for v in values) or len(values) < 2:
        return NAN
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) / 2


def _jackknife(values: list[float]) -> float:
    if any(math.isnan(v) for v in values):
        return NAN
    k = len(values)
    mean = sum(values) / k
    return math.sqrt((k - 1) / k * sum((v - mean) ** 2 for v in values))


def with_errors(fn, k: int, cross: bool) -> dict:
    """{quantity: {value, se, se_rule, ...}} for fn(folds) -> {quantity: value}."""
    all_folds = list(range(k))
    full = fn(all_folds)
    jack = [fn([b for b in all_folds if b != a]) for a in all_folds]
    single = None if cross else [fn([a]) for a in all_folds]
    out = {}
    for key, value in full.items():
        jk = _jackknife([j[key] for j in jack])
        if cross:
            out[key] = {"value": value, "se": jk, "se_rule": "jackknife", "delete_one": [j[key] for j in jack]}
        else:
            folds = [s[key] for s in single]
            out[key] = {"value": value, "se": _sd_half(folds), "se_rule": "sd/2", "jackknife_se": jk, "folds": folds}
    return out


def gram12(gr: Gram, metric: str) -> tuple[list[str], list[list[float]]]:
    """The 12-vector Gram (S_correct, S_incorrect, A per fold) from the 28-vector Gram by linearity."""
    groups, names = [], []
    for a in range(gr.k):
        for c in CLASSES:
            names.append(f"S|{a}|{c}")
            groups.append([f"S|{a}|{c}|{f}" for f in FLAGS])
        names.append(f"A|{a}")
        groups.append([f"A|{a}"])
    return names, [[gr.block(metric, u, v) for v in groups] for u in groups]


def compute(data: dict) -> dict:
    gr = Gram(data)
    counts = {"trajectories": sum(len(p["labels"]) for p in gr.prompts), "prompts": len(gr.prompts),
              "label": {c: sum(1 for p in gr.prompts for x in p["cls"] if x == c) for c in CLASSES},
              "flag": {f: sum(1 for p in gr.prompts for x in p["flags"] if x == f) for f in FLAGS},
              "prompt_class": {k: sum(1 for p in gr.prompts if p["class"] == k)
                               for k in ("mixed", "all_correct", "all_wrong", "other")}}
    result = {"checkpoint": data.get("checkpoint"), "counts": counts, "metrics": {}}
    for metric in ("raw", "D"):
        block = {"subsets": {}, "pair_cosines": {}}
        for name, sel in subsets().items():
            block["subsets"][name] = with_errors(lambda f, sel=sel: one_fold(gr, metric, sel, f), gr.k, cross=False)
        for view in ("pooled",) + FLAGS:
            sel = subsets()[view]
            block["pair_cosines"][view] = with_errors(lambda f, sel=sel: pair_cosines(gr, metric, sel, f), gr.k, cross=False)
        block["cross_fold"] = with_errors(lambda f: cross_fold(gr, metric, f), gr.k, cross=True)
        result["metrics"][metric] = block
    return result


def main():
    data = json.loads(open(sys.argv[1]).read())
    out = compute(data)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
