"""stats2.py on synthetic vectors: each value equals the direct computation from the vectors."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import stats2  # noqa: E402

VIEWS = ["p0.00-z1.5", "p1-z2.0", "p0.30-z3.0"]
FOLDS = [1, 2, 3, 4]
PARTS = [1, 2, 3, 4]
DIM = 50


def make(tmp_path: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.5, 2.0, DIM)
    vec = {}
    for v in VIEWS:
        mu = rng.normal(size=DIM)
        for f in FOLDS:
            vec[stats2.gname(f, v)] = mu + rng.normal(size=DIM)
    for kind in ("skill", "fmt"):
        base = rng.normal(size=DIM)
        for h in PARTS:
            vec[stats2.hname(kind, h)] = base + 0.5 * rng.normal(size=DIM)
    vec["D__A1_0_80"] = rng.normal(size=DIM)
    names = list(vec)
    x = np.stack([vec[n] for n in names])
    gram = {"names": names, "raw": (x @ x.T).tolist(), "M1": ((x * d) @ x.T).tolist(),
            "M2": ((x * d) @ (x * d).T).tolist(), "metric": {"test": True}}
    (tmp_path / "M0").mkdir()
    (tmp_path / "M0" / "GRAM_M0.json").write_text(json.dumps(gram))
    return vec, d


def cross_ip(vec, P1, vs, ws, folds=FOLDS):
    vals = []
    for a in folds:
        for b in folds:
            if a != b:
                vals.append(np.mean([vec[stats2.gname(a, v)] @ (P1 * vec[stats2.gname(b, w)]) for v in vs for w in ws]))
    return float(np.mean(vals))


def test_values_match_vectors(tmp_path):
    vec, d = make(tmp_path)
    res = stats2.compute(tmp_path, ["M0"])
    for metric, p1, p2 in (("raw", np.ones(DIM), np.ones(DIM)), ("D", d, d ** 2)):
        q = res["checkpoints"]["M0"]["metrics"][metric]
        gs = np.mean([vec[stats2.hname("skill", h)] for h in PARTS], axis=0)
        gf = np.mean([vec[stats2.hname("fmt", h)] for h in PARTS], axis=0)
        for v in VIEWS:
            assert q[f"sqnorm_Eg[{v}]"]["value"] == pytest.approx(cross_ip(vec, p1, [v], [v]), rel=1e-10)
            eg = np.mean([vec[stats2.gname(a, v)] for a in FOLDS], axis=0)
            num = -float(eg @ (p1 * gs))
            den = math.sqrt(cross_ip(vec, p2, [v], [v]))
            assert q[f"U_skill[{v}]"]["value"] == pytest.approx(num / den, rel=1e-10)
            assert q[f"U_skill[{v}]_num"] == pytest.approx(num, rel=1e-10)
        egv = np.mean([vec[stats2.gname(a, v)] for a in FOLDS for v in VIEWS], axis=0)
        u_v = -float(egv @ (p1 * gs)) / math.sqrt(cross_ip(vec, p2, VIEWS, VIEWS))
        assert q["U_skill_V"]["value"] == pytest.approx(u_v, rel=1e-10)
        assert q["dU_skill"]["value"] == pytest.approx(u_v - q["U_skill[p1-z2.0]"]["value"], rel=1e-10)
        eg0 = np.mean([vec[stats2.gname(a, "p1-z2.0")] for a in FOLDS], axis=0)
        assert q["bias_fmt"]["value"] == pytest.approx(-float((eg0 - egv) @ (p1 * gf)), rel=1e-9)
        sqv = cross_ip(vec, p1, VIEWS, VIEWS)
        assert q["view_ICC"]["value"] == pytest.approx(sqv / np.mean([cross_ip(vec, p1, [v], [v]) for v in VIEWS]))
        delta = vec["D__A1_0_80"]
        c = -float(egv @ (p1 * delta)) / (math.sqrt(cross_ip(vec, p2, VIEWS, VIEWS)) * math.sqrt(delta @ delta))
        assert q["cos_u_Delta[V][D__A1_0_80]"]["value"] == pytest.approx(c, rel=1e-10)
        cs = float(gs @ (p1 * gf)) / math.sqrt(float(gs @ (p1 * gs)) * float(gf @ (p1 * gf)))
        assert q["cos_skill_fmt"]["value"] == pytest.approx(cs, rel=1e-10)
        assert q["g_skill_norm"]["value"] == pytest.approx(math.sqrt(float(gs @ (p1 * gs))), rel=1e-10)


def test_jackknife(tmp_path):
    vec, d = make(tmp_path, seed=1)
    res = stats2.compute(tmp_path, ["M0"])
    q = res["checkpoints"]["M0"]["metrics"]["D"]["U_skill_V"]
    # delete-one-part: the value with parts other than h
    gs3 = np.mean([vec[stats2.hname("skill", h)] for h in (2, 3, 4)], axis=0)
    egv = np.mean([vec[stats2.gname(a, v)] for a in FOLDS for v in VIEWS], axis=0)
    u = -float(egv @ (d * gs3)) / math.sqrt(cross_ip(vec, d ** 2, VIEWS, VIEWS))
    assert q["delete_one_part"][0] == pytest.approx(u, rel=1e-10)
    # delete-one-fold: folds other than 1
    f3 = [2, 3, 4]
    gs = np.mean([vec[stats2.hname("skill", h)] for h in PARTS], axis=0)
    egv3 = np.mean([vec[stats2.gname(a, v)] for a in f3 for v in VIEWS], axis=0)
    u3 = -float(egv3 @ (d * gs)) / math.sqrt(cross_ip(vec, d ** 2, VIEWS, VIEWS, folds=f3))
    assert q["delete_one_fold"][0] == pytest.approx(u3, rel=1e-10)
    vals = q["delete_one_fold"]
    mu = sum(vals) / 4
    assert q["se_fold"] == pytest.approx(math.sqrt(3 / 4 * sum((x - mu) ** 2 for x in vals)))
    assert q["se_max"] == max(q["se_fold"], q["se_part"])


def test_undefined_values_are_null(tmp_path):
    vec, d = make(tmp_path, seed=2)
    gram = json.loads((tmp_path / "M0" / "GRAM_M0.json").read_text())
    i = gram["names"].index(stats2.gname(1, "p0.30-z3.0"))
    # a zero vector of one view in all folds gives a zero norm estimate for that view
    for f in FOLDS:
        j = gram["names"].index(stats2.gname(f, "p0.30-z3.0"))
        for key in ("raw", "M1", "M2"):
            for k in range(len(gram["names"])):
                gram[key][j][k] = gram[key][k][j] = 0.0
    assert i >= 0
    (tmp_path / "M0" / "GRAM_M0.json").write_text(json.dumps(gram))
    out = tmp_path / "STATS_VIEW2.json"
    stats2.write(stats2.compute(tmp_path, ["M0"]), out)
    res = json.loads(out.read_text())
    q = res["checkpoints"]["M0"]["metrics"]["D"]
    assert q["U_skill[p0.30-z3.0]"]["value"] is None
    assert q["U_skill[p0.30-z3.0]_norm_nonpositive"] is True
    assert res["undefined_count"] > 0
