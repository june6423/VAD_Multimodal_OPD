"""STATS_VIEW2.json of HANDOFF-VLM-VAD-VIEW-001-R3 Part A item 8, from the Gram files only.

Input, for each checkpoint ck: <root>/<ck>/GRAM_<ck>.json (gram.py: names, raw = <x, y>, M1 = <x, D y>, M2 = <x, D^2 y>).
Vector names: G__f<f>__<view> = g(B8(f), v) (Definition 9.1), H__skill__h<h> and H__fmt__h<h> = g_skill(h) and g_fmt(h)
(Definition 9.2), D__A1_0_80 and D__A1_80_130 (Definition 9.3), GEO__f<f> = g_geo(B8(f)) (Definition 9.5).
Metrics: "raw" (P = I: inner product raw, step norm raw) and "D" (P = D: inner product M1, step norm M2), as in
scripts/vad_grad3/stats.py. ||Delta||^2 is raw in both metrics.
Estimates (Definition 10, cross-fold): <E x_v, E y_w> = mean over folds a != b of <x(a, v), y(b, w)>; E g_V = mean over v
of E g_v; <E g_v, h> for a fixed vector h = mean over a of <g(a, v), h>. g_skill = mean over the parts h of g_skill(h),
g_fmt the same.
Quantities (item 8), for each checkpoint and metric:
  1. sqnorm_Eg[v] = ||E g_v||^2, sqnorm_EgV, cos_Eg_Egv0[v], view_ICC = ||E g_V||^2 / mean_v ||E g_v||^2.
  2. U_skill[v], U_skill_V = -<E g, g_skill> / ||P E g|| (Definition 11), U_fmt the same; dU_skill = U_skill_V -
     U_skill[v0], dU_fmt.
  3. bias_skill = -<b_v0, g_skill>, bias_fmt, b_v0 = E g_v0 - E g_V.
  4. cos_u_Delta[v or V][Delta] = cos(-P E g, Delta) for each displacement in the Gram file.
  5. g_skill_norm, g_fmt_norm, cos_skill_fmt, mean_cos_parts_skill, mean_cos_parts_fmt (inner product of the metric).
  6. SE: delete-one-fold jackknife over the training folds (se_fold) and delete-one-part jackknife over the parts of
     set A (se_part), sqrt((k-1)/k sum (T_i - Tbar)^2). se_max = the larger.
  7. Each cos and U carries its numerator and its squared-norm estimates (keys <name>_num, <name>_norm2_*) and
     norm_nonpositive. An undefined value (a norm estimate <= 0, a division by 0) is null; undefined_count counts the
     null values of the file.
  8. Only when the Gram file has GEO__ names (E g_geo = cross-fold mean of GEO__f<f>): sqnorm_Eg_geo, cos_Eg_geo_EgV,
     cos_Eg_geo_Egv0, U_skill[geo], U_fmt[geo], dU_geo = U_skill[geo] - U_skill[v0], dU_geo_V = U_skill[geo] -
     U_skill_V, with the SEs of item 6. geo is not a view: it is not in E g_V, view_ICC or cos_u_Delta.

Usage: python stats2.py <root> <STATS_VIEW2.json> <checkpoint>...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

V0 = "p1-z2.0"
GEO = "geo"
NAN = float("nan")


def gname(f: int, v: str) -> str:
    return f"G__f{f}__{v}"


def vname(f: int, v: str) -> str:
    """Gram name of fold f of view v, or of the geometric pool for v == GEO."""
    return f"GEO__f{f}" if v == GEO else gname(f, v)


def hname(kind: str, h: int) -> str:
    return f"H__{kind}__h{h}"


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else NAN


def div(a: float, b: float) -> float:
    return a / b if b != 0 and not math.isnan(a) and not math.isnan(b) else NAN


def sqrt(x: float) -> float:
    return math.sqrt(x) if x > 0 else NAN


class Metric:
    def __init__(self, gram: dict, metric: str):
        self.i = {n: k for k, n in enumerate(gram["names"])}
        self.raw = gram["raw"]
        self.g = gram["M1"] if metric == "D" else gram["raw"]
        self.p = gram["M2"] if metric == "D" else gram["raw"]

    def ip(self, x: str, y: str) -> float:
        return self.g[self.i[x]][self.i[y]]

    def pp(self, x: str, y: str) -> float:
        return self.p[self.i[x]][self.i[y]]

    def dd(self, d: str) -> float:
        return self.raw[self.i[d]][self.i[d]]


def layout(names: list[str]) -> tuple[list[int], list[str], list[int], list[str]]:
    folds = sorted({int(n.split("__")[1][1:]) for n in names if n.startswith("G__")})
    views = sorted({n.split("__")[2] for n in names if n.startswith("G__")})
    parts = sorted({int(n.split("__")[2][1:]) for n in names if n.startswith("H__skill__")})
    deltas = sorted(n for n in names if n.startswith("D__"))
    return folds, views, parts, deltas


def ratio(q: dict, name: str, num: float, norms: dict[str, float], den: float):
    """q[name] = num / den with the numerator, the squared-norm estimates and the non-positive flag."""
    q[f"{name}_num"] = num
    for k, v in norms.items():
        q[f"{name}_norm2_{k}"] = v
    q[f"{name}_norm_nonpositive"] = any(v <= 0 for v in norms.values())
    q[name] = NAN if q[f"{name}_norm_nonpositive"] else div(num, den)


def quantities(m: Metric, folds: list[int], views: list[str], parts: list[int], deltas: list[str]) -> dict:
    q = {}
    pairs = [(a, b) for a in folds for b in folds if a != b]

    def cross(fn, vs: list[str], ws: list[str]) -> float:
        return mean(mean(fn(vname(a, v), vname(b, w)) for v in vs for w in ws) for a, b in pairs)

    def with_h(kind: str, vs: list[str]) -> float:
        """<E g_vs, g_kind> = mean over a, v in vs and h of <g(a, v), g_kind(h)>."""
        return mean(m.ip(vname(a, v), hname(kind, h)) for a in folds for v in vs for h in parts)

    has_geo = any(n.startswith("GEO__") for n in m.i)
    sets = {v: [v] for v in views}
    sets["V"] = list(views)
    if has_geo:
        sets[GEO] = [GEO]
    sq = {k: cross(m.ip, vs, vs) for k, vs in sets.items()}
    sqp = {k: cross(m.pp, vs, vs) for k, vs in sets.items()}
    for v in views:
        q[f"sqnorm_Eg[{v}]"] = sq[v]
        ratio(q, f"cos_Eg_Egv0[{v}]", cross(m.ip, [v], [V0]), {"x": sq[v], "y": sq[V0]}, sqrt(sq[v]) * sqrt(sq[V0]))
    q["sqnorm_EgV"] = sq["V"]
    q["view_ICC"] = div(sq["V"], mean(sq[v] for v in views))
    for kind in ("skill", "fmt"):
        hh = {k: with_h(kind, vs) for k, vs in sets.items()}
        for k in sets:
            name = f"U_{kind}_V" if k == "V" else f"U_{kind}[{k}]"
            ratio(q, name, -hh[k], {"step": sqp[k]}, sqrt(sqp[k]))
        q[f"dU_{kind}"] = q[f"U_{kind}_V"] - q[f"U_{kind}[{V0}]"]
        q[f"bias_{kind}"] = -(hh[V0] - hh["V"])
    if has_geo:
        q["sqnorm_Eg_geo"] = sq[GEO]
        for name, ws in (("cos_Eg_geo_EgV", "V"), ("cos_Eg_geo_Egv0", V0)):
            ratio(q, name, cross(m.ip, [GEO], sets[ws]), {"x": sq[GEO], "y": sq[ws]}, sqrt(sq[GEO]) * sqrt(sq[ws]))
        q["dU_geo"] = q[f"U_skill[{GEO}]"] - q[f"U_skill[{V0}]"]
        q["dU_geo_V"] = q[f"U_skill[{GEO}]"] - q["U_skill_V"]
    for d in deltas:
        dd = m.dd(d)
        for k, vs in sets.items():
            if k == GEO:
                continue
            num = -mean(m.ip(gname(a, v), d) for a in folds for v in vs)
            ratio(q, f"cos_u_Delta[{k}][{d}]", num, {"u": sqp[k], "delta": dd}, sqrt(sqp[k]) * sqrt(dd))

    def hh_ip(kx: str, ky: str) -> float:
        return mean(m.ip(hname(kx, h), hname(ky, k)) for h in parts for k in parts)

    ss, ff, sf = hh_ip("skill", "skill"), hh_ip("fmt", "fmt"), hh_ip("skill", "fmt")
    q["g_skill_norm"], q["g_fmt_norm"] = sqrt(ss), sqrt(ff)
    ratio(q, "cos_skill_fmt", sf, {"skill": ss, "fmt": ff}, sqrt(ss) * sqrt(ff))
    for kind in ("skill", "fmt"):
        cs = [div(m.ip(hname(kind, h), hname(kind, k)),
                  sqrt(m.ip(hname(kind, h), hname(kind, h))) * sqrt(m.ip(hname(kind, k), hname(kind, k))))
              for h in parts for k in parts if h < k]
        q[f"mean_cos_parts_{kind}"] = mean(cs)
    return q


def jackknife(values: list[float]) -> float:
    if len(values) < 2 or any(math.isnan(v) for v in values):
        return NAN
    k = len(values)
    mu = sum(values) / k
    return math.sqrt((k - 1) / k * sum((v - mu) ** 2 for v in values))


def with_errors(m: Metric, folds, views, parts, deltas) -> dict:
    full = quantities(m, folds, views, parts, deltas)
    jf = [quantities(m, [b for b in folds if b != a], views, parts, deltas) for a in folds]
    jp = [quantities(m, folds, views, [k for k in parts if k != h], deltas) for h in parts]
    out = {}
    for key, value in full.items():
        if isinstance(value, bool) or "_num" in key or "_norm2_" in key:
            out[key] = value
            continue
        df = [j.get(key, NAN) for j in jf]
        dp = [j.get(key, NAN) for j in jp]
        se_f, se_p = jackknife(df), jackknife(dp)
        out[key] = {"value": value, "se_fold": se_f, "se_part": se_p,
                    "se_max": NAN if math.isnan(se_f) or math.isnan(se_p) else max(se_f, se_p),
                    "delete_one_fold": df, "delete_one_part": dp}
    return out


def compute(root: Path, cks: list[str]) -> dict:
    res = {}
    for ck in cks:
        gram = json.loads((root / ck / f"GRAM_{ck}.json").read_text())
        folds, views, parts, deltas = layout(gram["names"])
        if V0 not in views:
            raise SystemExit(f"STOP: {ck}: the Gram file has no vectors of v0 = {V0}")
        res[ck] = {"folds": folds, "views": views, "parts": parts, "displacements": deltas,
                   "metric_file": gram.get("metric"),
                   "metrics": {mt: with_errors(Metric(gram, mt), folds, views, parts, deltas) for mt in ("raw", "D")}}
    return {"v0": V0, "checkpoints": res}


def null_if_undefined(x, counter: list):
    if isinstance(x, dict):
        return {k: null_if_undefined(v, counter) for k, v in x.items()}
    if isinstance(x, list):
        return [null_if_undefined(v, counter) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        counter[0] += 1
        return None
    return x


def write(res: dict, out: Path):
    counter = [0]
    clean = null_if_undefined(res, counter)
    clean["undefined_count"] = counter[0]
    Path(out).write_text(json.dumps(clean, indent=1, allow_nan=False))


def main():
    root, out, *cks = sys.argv[1:]
    write(compute(Path(root), cks), Path(out))


if __name__ == "__main__":
    main()
