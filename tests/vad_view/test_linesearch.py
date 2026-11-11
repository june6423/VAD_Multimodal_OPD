"""linesearch.py helpers: points, items, direction norms from the Gram file, the summary table, resume."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import linesearch as ls  # noqa: E402


def test_points():
    pts = ls.point_names(["MEAN__V", "MEAN__p1-z2.0"], [1.0, 4.0, 16.0])
    assert [p for p, _, _ in pts] == ["a0-r1", "a0-r2", "MEAN__V@a1", "MEAN__V@a4", "MEAN__V@a16",
                                      "MEAN__p1-z2.0@a1", "MEAN__p1-z2.0@a4", "MEAN__p1-z2.0@a16"]
    assert pts[0][1] is None and pts[3] == ("MEAN__V@a4", "MEAN__V", 4.0)
    assert len(ls.point_names([f"MEAN__{i}" for i in range(9)], [1, 4, 16])) == 29


def test_items_of():
    sel = {"items": [{"index": i, "set": "A" if i < 6 else "B"} for i in range(10)]}
    assert [it["index"] for it in ls.items_of(sel, 0)] == list(range(10))
    assert [it["index"] for it in ls.items_of(sel, 2)] == [0, 1, 6, 7]


def test_direction_norm_from_gram():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 2, 20)
    names = ["G__f1__a", "G__f2__a", "G__f1__b", "G__f2__b", "D__A1_0_80"]
    x = {n: rng.normal(size=20) for n in names}
    m = np.stack([x[n] for n in names])
    gram = {"names": names, "M2": ((m * d) @ (m * d).T).tolist(),
            "fold_means": {"a": ["G__f1__a", "G__f2__a"], "b": ["G__f1__b", "G__f2__b"]}}
    for direction, members in (("MEAN__a", ["G__f1__a", "G__f2__a"]), ("MEAN__V", names[:4])):
        assert sorted(ls.direction_members(gram, direction)) == sorted(members)
        mean = np.mean([x[n] for n in members], axis=0)
        assert math.sqrt(ls.gram_norm2(gram, members)) == pytest.approx(np.linalg.norm(d * mean), rel=1e-12)


def lines_for(values: dict[int, tuple[float, float, bool]], sets: dict[int, str]) -> list[dict]:
    return [{"index": i, "set": sets[i], "log_R": v[0], "m": v[1], "correct": v[2]} for i, v in values.items()]


def test_summary_floor_marks_and_bf16():
    sets = {1: "A", 2: "A", 3: "B", 4: "B"}
    base = {1: (-0.5, 0.6, True), 2: (-1.0, 0.4, False), 3: (-0.2, 0.9, True), 4: (-2.0, 0.3, False)}
    rep = {**base, 1: (-0.5 + 1e-6, 0.6, True)}
    big = {i: (r + 0.1, m + 0.01, c) for i, (r, m, c) in base.items()}
    small = {**base, 2: (-1.0 + 1e-6, 0.4, False)}
    points = [("a0-r1", None, 0.0), ("a0-r2", None, 0.0), ("MEAN__V@a1", "MEAN__V", 1.0),
              ("MEAN__V@a4", "MEAN__V", 4.0)]
    lines = {"a0-r1": lines_for(base, sets), "a0-r2": lines_for(rep, sets), "MEAN__V@a1": lines_for(small, sets),
             "MEAN__V@a4": lines_for(big, sets)}
    bf16 = [{**x, "log_R": x["log_R"] + 0.01, "m": x["m"] - 0.002} for x in lines_for(base, sets)]
    bf16.append({"index": 99, "set": "A", "log_R": 0.0, "m": 1.0, "correct": True})
    metas = {"MEAN__V@a1": {"bf16_changed_fraction": 0.001}}
    res = ls.summarize(points, lines, metas, bf16)
    assert res["floor"]["A"]["mean_log_R"] == pytest.approx(0.5e-6)
    assert res["floor"]["B"]["mean_log_R"] == 0.0
    assert res["item_floor"]["log_R"] == pytest.approx(1e-6)
    t = {r["point"]: r for r in res["table"]}
    assert t["MEAN__V@a1"]["sets"]["A"]["change"]["mean_log_R"] == pytest.approx(0.5e-6)
    assert t["MEAN__V@a1"]["sets"]["A"]["below_2x_floor"]["mean_log_R"] is True
    assert t["MEAN__V@a4"]["sets"]["A"]["change"]["mean_log_R"] == pytest.approx(0.1)
    assert t["MEAN__V@a4"]["sets"]["A"]["below_2x_floor"]["mean_log_R"] is False
    assert t["MEAN__V@a4"]["sets"]["B"]["below_2x_floor"]["mean_log_R"] is False  # floor 0: nothing is below
    assert t["MEAN__V@a1"]["bf16_changed_fraction"] == 0.001 and t["MEAN__V@a4"]["bf16_changed_fraction"] is None
    fb = res["fp32_minus_bf16_alpha0"]
    assert fb["items"] == 4 and fb["means"]["A"]["mean_log_R"] == pytest.approx(-0.01)
    assert fb["means"]["B"]["mean_m"] == pytest.approx(0.002) and fb["item_max_abs"]["m"] == pytest.approx(0.002)
    assert fb["argmax_disagree"] == 0
    with pytest.raises(SystemExit):
        ls.summarize(points, lines, metas, bf16[1:])


def test_pending_resume(tmp_path):
    items = [{"index": i} for i in range(5)]
    path = tmp_path / "p.jsonl"
    assert ls.pending(path, items) == items
    path.write_text("".join(json.dumps({"index": i}) + "\n" for i in (0, 3)))
    assert [it["index"] for it in ls.pending(path, items)] == [1, 2, 4]
