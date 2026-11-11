"""stats1.py on synthetic stage 1 files with known answers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import stats1  # noqa: E402

L = ["A", "B", "C", "D"]
V0 = "p1-z2.0"
VW = "p0.30-z3.0"


def lp(winner: int, mass: float = 0.8) -> list[float]:
    p = [0.1 * mass / 3] * 4
    p[winner] = 0.9 * mass
    p = [x / sum(p) * mass for x in p]
    return [math.log(x) for x in p]


def test_fresh(tmp_path):
    # gt is A (index 0). Rows: (v0 right, vw right), (v0 wrong, vw right), (v0 right, vw wrong), (both wrong).
    pairs = [(0, 0), (1, 0), (0, 2), (1, 2)]
    rows = [{"row": i, "gt": "A", "letters": L, "logp": {V0: lp(a), VW: lp(b)}} for i, (a, b) in enumerate(pairs)]
    res = stats1.fresh_stats(rows)
    assert res["accuracy"] == {V0: 0.5, VW: 0.5}
    assert res["union_accuracy"] == 0.75
    assert res["mass"][V0] == pytest.approx(0.8)
    # errors v0 = [0,1,0,1], vw = [0,0,1,1]: phi = 0
    a, b = sorted([V0, VW])
    assert res["phi"][f"{a}|{b}"] == pytest.approx(0.0)
    # ensemble of a row with two different winners of equal strength: a tie of A with the other letter only when
    # the other letter is first; here the other letters come after A, so the ensemble picks A.
    assert res["ensemble_accuracy"] == 0.75
    assert res["v0_wrong_ens_right"] == 1 and res["v0_right_ens_wrong"] == 0
    assert res["sign_test_p"] == 1.0


def test_sign_test():
    assert stats1.sign_test(0, 0) == 1.0
    assert stats1.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert stats1.sign_test(3, 7) == pytest.approx(stats1.sign_test(7, 3))
    assert stats1.sign_test(3, 7) == pytest.approx(2 * sum(math.comb(10, i) for i in range(4)) / 1024)


def test_phi_undefined():
    assert stats1.phi([1, 1, 1], [0, 1, 0]) is None


def test_prefix_rates_and_bootstrap(tmp_path):
    rows = []
    t = 0
    rng = np.random.default_rng(0)
    for p in range(30):
        for s in range(4):
            correct = s == 0
            v0_ok, vw_ok = bool(rng.integers(2)), bool(rng.integers(2))
            rows.append({"traj": t, "prompt_pos": p, "correct": correct, "letter": "B" if s < 3 else "", "gt": "A",
                         "letters": L, "flag": "ok", "logp": {V0: lp(0 if v0_ok else 1), VW: lp(0 if vw_ok else 2)}})
            t += 1
    rows.append({"traj": t, "prompt_pos": 99, "correct": False, "letter": "C", "gt": "A", "flag": "fx", "logp": None})
    res = stats1.prefix_stats(rows)
    use = [r for r in rows if not r["correct"] and r["letter"] and r["flag"] == "ok"]
    assert res["counted"] == len(use) == 60
    assert res["flags_incorrect_lettered"] == {"ok": 60, "fx": 1}
    v0 = np.mean([stats1.right(L, r["logp"][V0], "A") for r in use])
    un = np.mean([stats1.right(L, r["logp"][V0], "A") or stats1.right(L, r["logp"][VW], "A") for r in use])
    assert res["rate"][V0]["value"] == pytest.approx(v0)
    assert res["rate"]["union"]["value"] == pytest.approx(un)
    assert res["union_minus_v0"]["value"] == pytest.approx(un - v0)
    # bootstrap by hand: the same draws
    prompts = sorted({r["prompt_pos"] for r in use})
    draws = np.random.default_rng(1).integers(0, len(prompts), size=(2000, len(prompts)))
    by = {p: [stats1.right(L, r["logp"][V0], "A") for r in use if r["prompt_pos"] == p] for p in prompts}
    vals = []
    for d in draws:
        hits = [h for i in d for h in by[prompts[i]]]
        vals.append(np.mean(hits))
    assert res["rate"][V0]["se"] == pytest.approx(np.std(vals, ddof=1), rel=1e-9)
    assert res["rate"]["union"]["se"] > 0


def test_compute_files(tmp_path):
    rows = [{"row": i, "gt": "A", "letters": L, "logp": {V0: lp(i % 2), VW: lp(0)}} for i in range(6)]
    (tmp_path / "FRESH.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pre = [{"traj": 0, "prompt_pos": 0, "correct": True, "letter": "A", "gt": "A", "flag": "ok", "letters": L,
            "logp": {V0: lp(0), VW: lp(0)}}]
    (tmp_path / "PREFIX_step0.jsonl").write_text(json.dumps(pre[0]) + "\n")
    out = tmp_path / "S.json"
    sys.argv = ["stats1.py", str(tmp_path), str(out), "0"]
    stats1.main()
    res = json.loads(out.read_text())
    assert res["fresh"]["accuracy"][V0] == 0.5
    assert res["prefix"]["step0"]["counted"] == 0
