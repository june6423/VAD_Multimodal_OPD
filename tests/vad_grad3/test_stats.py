"""Tests of stats.py (HANDOFF-VLM-VAD-GRAD-003-R2 Part A item 8.4): synthetic vectors with a known mean and covariance,
and hand computations from the vectors (not from the Gram matrices)."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

DIR = Path(__file__).resolve().parents[2] / "scripts" / "vad_grad3"


def load(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stats = load("vad_grad3_stats", DIR / "stats.py")
FOLDS = (1, 2, 3, 4)
DISP = ("D__A1_0_80", "D__A1_80_130", "D__A2_0_80", "D__A2_80_130")


def write_ck(root: Path, ck: str, vecs: dict[str, np.ndarray], d: np.ndarray, rows8: dict, rows1: dict):
    names = sorted(vecs)
    x = np.stack([vecs[n] for n in names])
    xd = x * d
    (root / ck / "rows").mkdir(parents=True)
    (root / ck / f"GRAM_{ck}.json").write_text(json.dumps(
        {"names": names, "raw": (x @ x.T).tolist(), "M1": (xd @ x.T).tolist(), "M2": (xd @ xd.T).tolist()}))
    for f, rows in rows8.items():
        (root / ck / "rows" / f"ROWS_B8-f{f}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for (f, j), rows in rows1.items():
        (root / ck / "rows" / f"ROWS_B1-f{f}-j{j}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def rows_of(rng, n: int) -> list[dict]:
    return [{"weight": float(rng.uniform(0.1, 1.0)), "label": int(rng.integers(0, 2)),
             "len_bin": ["L1", "L2", "L3"][int(rng.integers(0, 3))]} for _ in range(n)]


def test_known_mean_and_covariance(tmp_path):
    """Orthogonal construction: g8(f) = mu + s e_f, g1(f, j) = mu1 + sb b_f + sw w_fj. Then m8 = ||mu||^2, N8 = s^2,
    m1 = ||mu1||^2, N1 = sb^2 + sw^2 (7/8 of sw^2 for the within part: mean over j != k of ||w_j - w_k||^2 / 2 = sw^2),
    ICC_step = sb^2 / (sb^2 + sw^2)."""
    dim = 64
    e = np.eye(dim)
    mu, mu1, s, sb, sw = 3 * e[0], 2 * e[1], 1.5, 0.8, 1.2
    vecs = {}
    for f in FOLDS:
        g = mu + s * e[1 + f]
        vecs[stats.g8(f)] = g
        vecs[stats.g8(f, "C")], vecs[stats.g8(f, "I")] = 0.25 * g, 0.75 * g
        for x in ("L1", "L2", "L3"):
            vecs[stats.g8(f, x)] = g / 3
        for j in range(8):
            g1 = mu1 + sb * e[6 + f] + sw * e[11 + 8 * (f - 1) + j]
            vecs[stats.g1(f, j)] = g1
            vecs[stats.g1(f, j, "C")], vecs[stats.g1(f, j, "I")] = 0.5 * g1, 0.5 * g1
    for k, dn in enumerate(DISP):
        vecs[dn] = e[-1 - k]
    rng = np.random.default_rng(0)
    rows8 = {f: rows_of(rng, 6) for f in FOLDS}
    rows1 = {(f, j): rows_of(rng, 2) for f in FOLDS for j in range(8)}
    write_ck(tmp_path, "A1_80", vecs, np.ones(dim), rows8, rows1)
    q = stats.compute(tmp_path, ["A1_80"])["checkpoints"]["A1_80"]["metrics"]["raw"]
    assert q["m8"]["value"] == pytest.approx(9.0) and q["N8"]["value"] == pytest.approx(s ** 2)
    assert q["SNR8"]["value"] == pytest.approx(9.0 / s ** 2)
    assert q["m1"]["value"] == pytest.approx(4.0) and q["N1"]["value"] == pytest.approx(sb ** 2 + sw ** 2)
    assert q["W1"]["value"] == pytest.approx(sw ** 2)
    assert q["ICC_step"]["value"] == pytest.approx(sb ** 2 / (sb ** 2 + sw ** 2))
    assert q["SNR8_over_SNR1"]["value"] == pytest.approx((9 / s ** 2) / (4 / (sb ** 2 + sw ** 2)))
    assert q["s_C"]["value"] == pytest.approx(0.25) and q["s_L2"]["value"] == pytest.approx(1 / 3)
    assert q["cos_EgC_EgI"]["value"] == pytest.approx(1.0)
    assert q["m8"]["se"] == pytest.approx(0.0, abs=1e-9)  # exact in every delete-one subset
    assert q["share_I"]["value"] == pytest.approx(0.75) and q["share_I"]["se"] == pytest.approx(0.0, abs=1e-12)


def make_random(rng, dim: int = 40):
    vecs = {}
    mu = {k: rng.normal(size=dim) for k in ("C", "I", "L1", "L2", "C1", "I1")}  # common means, so that the
    noise = lambda: 0.5 * rng.normal(size=dim)  # noqa: E731  cross-fold squared norms are positive
    for f in FOLDS:
        c, i = mu["C"] + noise(), mu["I"] + noise()
        vecs[stats.g8(f, "C")], vecs[stats.g8(f, "I")], vecs[stats.g8(f)] = c, i, c + i
        l1, l2 = mu["L1"] + noise(), mu["L2"] + noise()
        vecs[stats.g8(f, "L1")], vecs[stats.g8(f, "L2")], vecs[stats.g8(f, "L3")] = l1, l2, c + i - l1 - l2
        for j in range(8):
            c1, i1 = mu["C1"] + noise(), mu["I1"] + noise()
            vecs[stats.g1(f, j, "C")], vecs[stats.g1(f, j, "I")], vecs[stats.g1(f, j)] = c1, i1, c1 + i1
    for mode in ("O", "Fx"):
        for name in stats.COMP[mode]:
            vecs.setdefault(name, vecs[stats.g8(1)] + 0.1 * rng.normal(size=dim))
    vecs["O__B8-f1@rep__total"] = vecs[stats.g8(1)] + 0.01 * rng.normal(size=dim)
    for dn in DISP:
        vecs[dn] = rng.normal(size=dim)
    return vecs


def ref_cross(xs, ys, folds):
    """Mean over a != b of the mean over the vectors of xs[a] and ys[b] of their inner product (metric applied)."""
    return np.mean([np.mean([x @ y for x in xs[a] for y in ys[b]]) for a in folds for b in folds if a != b])


@pytest.fixture(scope="module")
def random_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("stats")
    rng = np.random.default_rng(5)
    vecs = make_random(rng)
    d = rng.uniform(0.5, 2.0, size=40)
    rows8 = {f: rows_of(rng, 12) for f in FOLDS}
    rows1 = {(f, j): rows_of(rng, 3) for f in FOLDS for j in range(8)}
    write_ck(root, "A1_80", vecs, d, rows8, rows1)
    vecs0 = {k: v * 1.1 for k, v in vecs.items()}
    write_ck(root, "M0", vecs0, d, rows8, rows1)
    res = stats.compute(root, ["M0", "A1_80"])
    return res, vecs, d, rows8, rows1


def frac(rows, part):
    w = sum(r["weight"] for r in rows)
    return sum(r["weight"] for r in rows if stats.member(r, part)) / w


@pytest.mark.parametrize("metric", ["raw", "D"])
def test_cross_fold_quantities_by_hand(random_case, metric):
    res, vecs, d, rows8, rows1 = random_case
    q = res["checkpoints"]["A1_80"]["metrics"][metric]
    p = d if metric == "D" else np.ones_like(d)
    m = {k: v * np.sqrt(p) for k, v in vecs.items()}  # <x, P y> = <sqrt(P) x, sqrt(P) y>
    one = {a: [m[stats.g8(a)]] for a in FOLDS}
    m8 = ref_cross(one, one, FOLDS)
    n8 = np.mean([m[stats.g8(a)] @ m[stats.g8(a)] for a in FOLDS]) - m8
    assert q["m8"]["value"] == pytest.approx(m8) and q["N8"]["value"] == pytest.approx(n8)
    all1 = {a: [m[stats.g1(a, j)] for j in range(8)] for a in FOLDS}
    m1 = ref_cross(all1, all1, FOLDS)
    n1 = np.mean([m[stats.g1(a, j)] @ m[stats.g1(a, j)] for a in FOLDS for j in range(8)]) - m1
    w1 = np.mean([np.sum((m[stats.g1(a, j)] - m[stats.g1(a, k)]) ** 2) / 2
                  for a in FOLDS for j in range(8) for k in range(8) if j != k])
    assert q["SNR1"]["value"] == pytest.approx(m1 / n1) and q["W1"]["value"] == pytest.approx(w1)
    assert q["ICC_step"]["value"] == pytest.approx((n1 - w1) / n1)
    assert q["SNR8_x19"]["value"] == pytest.approx(19 * m8 / n8)
    all8 = [r for f in FOLDS for r in rows8[f]]
    s = {}
    for x in stats.PARTS8:
        s[x] = ref_cross({a: [m[stats.g8(a, x)]] for a in FOLDS}, one, FOLDS) / m8
        assert q[f"s_{x}"]["value"] == pytest.approx(s[x])
        assert q[f"s_per_f_{x}"]["value"] == pytest.approx(s[x] / frac(all8, x))
    assert sum(s[x] for x in ("L1", "L2", "L3")) == pytest.approx(1.0)
    assert q["d_CI"]["value"] == pytest.approx(s["C"] / frac(all8, "C") - s["I"] / frac(all8, "I"))
    cc = ref_cross({a: [m[stats.g8(a, "C")]] for a in FOLDS}, {a: [m[stats.g8(a, "C")]] for a in FOLDS}, FOLDS)
    ii = ref_cross({a: [m[stats.g8(a, "I")]] for a in FOLDS}, {a: [m[stats.g8(a, "I")]] for a in FOLDS}, FOLDS)
    ci = ref_cross({a: [m[stats.g8(a, "C")]] for a in FOLDS}, {a: [m[stats.g8(a, "I")]] for a in FOLDS}, FOLDS)
    assert q["cos_EgC_EgI"]["value"] == pytest.approx(ci / np.sqrt(cc * ii))
    all1rows = [r for (f, j), rr in rows1.items() for r in rr]
    s1c = ref_cross({a: [m[stats.g1(a, j, "C")] for j in range(8)] for a in FOLDS}, all1, FOLDS) / m1
    assert q["n1_s_per_f_C"]["value"] == pytest.approx(s1c / frac(all1rows, "C"))
    # 7.7: cos(-P E g8, Delta_A1), u = -P g, ||E u||^2 cross-fold of <P g(a), P g(b)>
    pg = {a: [p * vecs[stats.g8(a)]] for a in FOLDS}
    for dn in ("D__A1_0_80", "D__A1_80_130"):
        num = -np.mean([pg[a][0] @ vecs[dn] for a in FOLDS])
        assert q[f"cos_EuE8_{dn}"]["value"] == pytest.approx(num / np.sqrt(ref_cross(pg, pg, FOLDS) * vecs[dn] @ vecs[dn]))
    pg1 = {a: [p * vecs[stats.g1(a, j)] for j in range(8)] for a in FOLDS}
    num = -np.mean([x @ vecs["D__A2_0_80"] for a in FOLDS for x in pg1[a]])
    assert q["cos_EuE1_D__A2_0_80"]["value"] == pytest.approx(
        num / np.sqrt(ref_cross(pg1, pg1, FOLDS) * vecs["D__A2_0_80"] @ vecs["D__A2_0_80"]))
    # 7.8
    for mode in ("O", "Fx"):
        names = stats.COMP[mode]
        v = np.mean([np.sum((m[a] - m[b]) ** 2) / 2 for a, b in itertools.combinations(names, 2)])
        assert q[f"V_comp_{mode}"]["value"] == pytest.approx(v / n8)
    assert q["V_num"]["value"] == pytest.approx(np.sum((m["O__B8-f1@rep__total"] - m[stats.g8(1)]) ** 2) / 2 / n8)
    # jackknife SE of SNR8: delete one fold, recompute
    t = []
    for a in FOLDS:
        rest = [b for b in FOLDS if b != a]
        mm = ref_cross(one, one, rest)
        t.append(mm / (np.mean([m[stats.g8(b)] @ m[stats.g8(b)] for b in rest]) - mm))
    assert q["SNR8"]["se"] == pytest.approx(np.sqrt(3 / 4 * np.sum((np.array(t) - np.mean(t)) ** 2)))
    assert q["SNR8"]["delete_one"] == pytest.approx(t)


@pytest.mark.parametrize("metric", ["raw", "D"])
def test_per_fold_quantities_by_hand(random_case, metric):
    res, vecs, d, rows8, rows1 = random_case
    q = res["checkpoints"]["A1_80"]["metrics"][metric]
    p = d if metric == "D" else np.ones_like(d)
    for dn in DISP:  # A1_80: all 4 displacements
        delta = vecs[dn]
        t = {x: [] for x in stats.PARTS8}
        cos_c, dci, cw, cg = [], [], [], []
        for f in FOLDS:
            u = {x: -p * vecs[stats.g8(f, x)] for x in stats.PARTS8 + ("total",)}
            for x in stats.PARTS8:
                t[x].append((u[x] @ delta) / (u["total"] @ delta))
            cos_c.append(u["C"] @ delta / np.linalg.norm(u["C"]) / np.linalg.norm(delta))
            dci.append(t["C"][-1] / frac(rows8[f], "C") - t["I"][-1] / frac(rows8[f], "I"))
            g1s = np.stack([vecs[stats.g1(f, j)] for j in range(8)])
            w = g1s - g1s.mean(axis=0)
            cw.append(np.mean([(-p * wj) @ delta / np.linalg.norm(p * wj) / np.linalg.norm(delta) for wj in w]))
            cg.append(np.mean([(-p * gj) @ delta / np.linalg.norm(p * gj) / np.linalg.norm(delta) for gj in g1s]))
        for x in stats.PARTS8:
            assert q[f"t_{x}_{dn}"]["value"] == pytest.approx(np.mean(t[x]))
            assert q[f"t_{x}_{dn}"]["per_fold"] == pytest.approx(t[x])
        assert q[f"cos_u_C_{dn}"]["value"] == pytest.approx(np.mean(cos_c))
        assert q[f"cos_u_C_{dn}"]["se"] == pytest.approx(np.std(cos_c, ddof=1) / 2)
        assert q[f"d_CI_{dn}"]["value"] == pytest.approx(np.mean(dci))
        assert q[f"hdir_cos_w_{dn}"]["value"] == pytest.approx(np.mean(cw))
        assert q[f"hdir_cos_g1_{dn}"]["value"] == pytest.approx(np.mean(cg))
    sq = np.sqrt(p)
    share = [(sq * vecs[stats.g8(f, "L3")]) @ (sq * vecs[stats.g8(f)]) / np.sum((sq * vecs[stats.g8(f)]) ** 2)
             for f in FOLDS]
    assert q["share_L3"]["value"] == pytest.approx(np.mean(share))
    f = 2
    t1 = [sum(-p * vecs[stats.g1(f, j, "C")] @ vecs["D__A2_80_130"] for j in range(8)) /
          sum(-p * vecs[stats.g1(f, j)] @ vecs["D__A2_80_130"] for j in range(8))]
    assert q["n1_t_C_D__A2_80_130"]["per_fold"][1] == pytest.approx(t1[0])
    assert "n1_t_C_D__A1_0_80" not in q


def test_displacements_of_each_checkpoint_and_differences(random_case):
    res = random_case[0]
    m0 = res["checkpoints"]["M0"]["metrics"]["raw"]
    assert "t_C_D__A1_0_80" in m0 and "t_C_D__A1_80_130" not in m0 and "cos_EuE8_D__A1_80_130" not in m0
    assert res["checkpoints"]["M0"]["rows"]["displacements"] == ["D__A1_0_80", "D__A2_0_80"]
    a1 = res["checkpoints"]["A1_80"]["metrics"]["raw"]
    diff = res["differences"]["A1_80-M0"]["raw"]["N8"]
    assert diff["value"] == pytest.approx(a1["N8"]["value"] - m0["N8"]["value"])
    assert diff["value"] == pytest.approx(a1["N8"]["value"] * (1 - 1.21))  # M0 vectors are 1.1 x A1_80
    assert diff["se"] == pytest.approx(np.hypot(a1["N8"]["se"], m0["N8"]["se"]))
    rows = res["checkpoints"]["A1_80"]["rows"]["n8"]["1"]
    assert rows["rows"] == 12 and rows["C"] + rows["I"] == 12


def test_two_folds_give_values_and_nan_jackknife(tmp_path):
    """The pilot has 2 folds: cross-fold values exist; a delete-one subset has one fold, so the jackknife SE is NaN."""
    rng = np.random.default_rng(9)
    vecs = {k: v for k, v in make_random(rng).items()
            if not any(f"f{f}" in k for f in (3, 4))}
    write_ck(tmp_path, "M0", vecs, np.ones(40), {f: rows_of(rng, 4) for f in (1, 2)},
             {(f, j): rows_of(rng, 2) for f in (1, 2) for j in range(8)})
    q = stats.compute(tmp_path, ["M0"])["checkpoints"]["M0"]["metrics"]["raw"]
    assert np.isfinite(q["SNR8"]["value"]) and np.isnan(q["SNR8"]["se"]) and q["SNR8"]["folds"] == 2
    assert np.isfinite(q["t_C_D__A1_0_80"]["se"])


def test_cross_fold_cos_parts_flag_and_null(tmp_path):
    """DIRECTIVE-VLM-VAD-GRAD-003-003: each cross-fold cos has its numerator and both squared-norm estimates, a flag
    for an estimate <= 0 (not clipped), and an undefined value is null in the written file (no NaN)."""
    rng = np.random.default_rng(11)
    vecs = {k: v for k, v in make_random(rng).items() if not any(f"f{f}" in k for f in (3, 4))}
    vecs[stats.g8(2, "C")] = -vecs[stats.g8(1, "C")]  # cross-fold ||E g_C||^2 = <C1, C2> < 0
    write_ck(tmp_path, "M0", vecs, np.ones(40), {f: rows_of(rng, 4) for f in (1, 2)},
             {(f, j): rows_of(rng, 2) for f in (1, 2) for j in range(8)})
    q = stats.compute(tmp_path, ["M0"])["checkpoints"]["M0"]["metrics"]["raw"]
    c1, c2, i1, i2 = (vecs[stats.g8(f, x)] for x in ("C", "I") for f in (1, 2))
    assert q["cos_EgC_EgI_num"]["value"] == pytest.approx((c1 @ i2 + c2 @ i1) / 2)
    assert q["cos_EgC_EgI_norm2_x"]["value"] == pytest.approx(c1 @ c2) and c1 @ c2 < 0
    assert q["cos_EgC_EgI_norm2_y"]["value"] == pytest.approx(i1 @ i2)
    assert q["cos_EgC_EgI"]["norm_nonpositive"] is True and np.isnan(q["cos_EgC_EgI"]["value"])
    assert "cos_EuE8_D__A1_0_80_num" in q and q["cos_EuE8_D__A1_0_80"]["norm_nonpositive"] is False
    out = tmp_path / "STATS_GRAD3.json"
    stats.write(stats.compute(tmp_path, ["M0"]), out)
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    w = json.loads(text)["checkpoints"]["M0"]["metrics"]["raw"]
    assert w["cos_EgC_EgI"]["value"] is None and w["SNR8"]["se"] is None and w["SNR8"]["delete_one"] == [None, None]


def test_checks_hold_part_sum_and_rounding_of_each_key(tmp_path):
    """DIRECTIVE-VLM-VAD-GRAD-003-005: part_sum_rel and rounding_rel of each key (progress files) in STATS."""
    rng = np.random.default_rng(12)
    vecs = {k: v for k, v in make_random(rng).items() if not any(f"f{f}" in k for f in (3, 4))}
    write_ck(tmp_path, "M0", vecs, np.ones(40), {f: rows_of(rng, 4) for f in (1, 2)},
             {(f, j): rows_of(rng, 2) for f in (1, 2) for j in range(8)})
    (tmp_path / "M0" / "progress_O__B8-f1.json").write_text(json.dumps(
        {"part_sum_rel": {"C+I": 0.01, "L1+L2+L3": 0.02}, "rounding_rel": 0.03, "micro": []}))
    (tmp_path / "M0" / "progress_Fx__R8-f1-s1.json").write_text(json.dumps(
        {"part_sum_rel": None, "rounding_rel": 0.025, "micro": []}))
    c = stats.compute(tmp_path, ["M0"])["checkpoints"]["M0"]["checks"]
    assert c == {"O__B8-f1": {"part_sum_rel": {"C+I": 0.01, "L1+L2+L3": 0.02}, "rounding_rel": 0.03},
                 "Fx__R8-f1-s1": {"part_sum_rel": None, "rounding_rel": 0.025}}
