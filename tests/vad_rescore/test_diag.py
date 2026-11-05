"""Test of scripts/vad_rescore/diag_prompt.py compare (DIRECTIVE-VLM-VAD-GRAD-002-001) on synthetic runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vad_rescore_diag", ROOT / "scripts" / "vad_rescore" / "diag_prompt.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def run(name: str, tok_a: list[float], tok_b: list[float]) -> dict:
    sibs = [{"traj": t, "sibling": s, "tokens": tok, "L": diag.fp32_sum(tok)} for t, s, tok in ((10, 0, tok_a), (11, 1, tok_b))]
    return {"name": name, "siblings": sibs}


def test_compare_finds_the_differing_token_and_the_groups():
    a, b = [-0.5, -1.25, -0.125], [-2.0, -0.25]
    odd = [-0.5, -1.5, -0.125]
    runs = [run("s1", a, b), run("s2", a, b), run("m", odd, b)]
    traj = {10: {"logp_sum": diag.fp32_sum(odd)}, 11: {"logp_sum": diag.fp32_sum(b)}}
    scores = {10: {"L": diag.fp32_sum(a)}, 11: {"L": diag.fp32_sum(b)}}
    res = diag.compare(traj, scores, runs)
    t10 = res["trajectories"][10]
    assert t10["differing_tokens"] == [{"index": 1, "s1": -1.25, "s2": -1.25, "m": -1.5}]
    assert sorted(map(sorted, t10["equal_groups"])) == [["FULL_L", "s1", "s2"], ["TRAJ_logp_sum", "m"]]
    assert res["trajectories"][11]["differing_tokens"] == [] and len(res["trajectories"][11]["equal_groups"]) == 1
    assert res["trajectories_with_any_difference"] == [10]


def test_fp32_sum_rounds_as_fp32():
    assert diag.fp32_sum([1e8, 1.0]) == 1e8  # 1e8 + 1 is not representable in fp32
