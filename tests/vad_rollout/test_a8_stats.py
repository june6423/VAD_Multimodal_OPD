"""HANDOFF-VLM-VAD-ROLLOUT-001-R9 gate G2: A8 statistics on synthetic gradient vectors with known mu and sigma^2."""

from __future__ import annotations

import importlib.util
import math
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

STATS = Path(__file__).resolve().parents[2] / "scripts" / "vad_rollout" / "a8" / "stats.py"
spec = importlib.util.spec_from_file_location("a8_stats", STATS)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)

D = 20000
MU_SQ = {True: 1.0, False: 1.0}
SIGMA_SQ = {True: 3.0, False: 1.0}
COS_MU = 0.6


def mean_vector(correct: bool) -> torch.Tensor:
    mu = torch.zeros(D, dtype=torch.float64)
    if correct:
        mu[0] = 1.0
    else:
        mu[0], mu[1] = COS_MU, math.sqrt(1 - COS_MU**2)
    return mu


def gradient(traj: int, correct: bool) -> list[torch.Tensor]:
    """Deterministic synthetic gradient of one trajectory, split in two tensors like two parameters."""
    gen = torch.Generator().manual_seed(1000 + traj)
    noise = torch.randn(D, generator=gen, dtype=torch.float64) * math.sqrt(SIGMA_SQ[correct] / D)
    vec = (mean_vector(correct) + noise).float()
    return [vec[: D // 2], vec[D // 2:]]


def make_pool(prompts: int = 375, siblings: int = 8, p: float = 0.6, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    return [{"id": pr * siblings + s, "prompt": pr, "correct": rng.random() < p, "flag": rng.choice(["f1", "f0", "fx"])}
            for pr in range(prompts) for s in range(siblings)]


def group_sum(ids, correct_of):
    total = None
    for traj in ids:
        g = gradient(traj, correct_of[traj])
        total = [x.clone() for x in g] if total is None else [t + x for t, x in zip(total, g)]
    return total


def close(value: float, expected: float, rel: float = 0.02) -> bool:
    return abs(value - expected) <= rel * abs(expected)


def fine_class(row: dict) -> str:
    return ("C_" if row["correct"] else "I_") + row["flag"]


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    return math.degrees(math.acos(float(a @ b / (a.norm() * b.norm()))))


def test_sampler_groups_are_disjoint_with_one_sibling_per_prompt():
    pool = make_pool()
    prompt_of = {r["id"]: r["prompt"] for r in pool}
    correct_of = {r["id"]: r["correct"] for r in pool}
    n_max = stats.largest_n(pool)
    for n, count in ((4, 64), (16, 8), (n_max, 2)):
        for trial in stats.draw_trials(pool, n, count, seed=42):
            members = [traj for g in stats.GROUPS for traj in trial[g]]
            assert len(members) == len(set(members)) == 4 * n
            for g in stats.GROUPS:
                assert len(trial[g]) == n
                assert max(Counter(prompt_of[t] for t in trial[g]).values()) == 1
                assert all(correct_of[t] == g.startswith("C") for t in trial[g])
    with pytest.raises(ValueError):
        stats.draw_trials(pool, n_max + 1, 1, seed=42)


def test_largest_n_closed_form():
    # Correct: prompt 0 has 3, prompt 1 has 1, prompt 2 has 1 -> min(k, 2) sum 4 -> n 2. Incorrect: 2 + 2 + 1 -> 2.
    pool = [{"id": i, "prompt": p, "correct": c} for i, (p, c) in enumerate(
        [(0, True)] * 3 + [(1, True), (2, True)] + [(0, False)] * 2 + [(1, False)] * 2 + [(2, False)])]
    assert stats.largest_n(pool) == 2


def test_trial_plan_marks_sizes_the_pool_cannot_give():
    pool = make_pool()
    n_max = stats.largest_n(pool)
    plan = stats.trial_plan(pool, seed=42)
    assert [(e["n"], e["count"], len(e["trials"])) for e in plan] == [(4, 64, 64), (16, 8, 8), (n_max, 2, 2)]
    small = make_pool(prompts=12)
    small_max = stats.largest_n(small)
    assert 4 < small_max < 16
    plan = stats.trial_plan(small, seed=42)
    assert [(e["n"], e["trials"] is None) for e in plan] == [(4, False), (16, True), (small_max, False)]
    tiny = make_pool(prompts=2)
    assert stats.largest_n(tiny) < 4
    assert all(e["trials"] is None for e in stats.trial_plan(tiny, seed=42) if e["n"] in (4, 16))


def test_across_prompt_statistics_match_closed_form():
    pool = make_pool()
    correct_of = {r["id"]: r["correct"] for r in pool}
    records = []
    for entry in stats.trial_plan(pool, seed=42):
        for trial in entry["trials"]:
            sums = {g: group_sum(trial[g], correct_of) for g in stats.GROUPS}
            records.append(stats.trial_scalars(sums, entry["n"]))
    norms = {"C": [], "I": []}
    for r in pool:
        g = gradient(r["id"], r["correct"])
        norms["C" if r["correct"] else "I"].append(math.sqrt(stats.dot(g, g)))
    p = sum(correct_of.values()) / len(pool)
    result = stats.summarize_trials(records, p, norms)

    c4 = {pop: MU_SQ[pop] / (MU_SQ[pop] + SIGMA_SQ[pop] / 4) for pop in (True, False)}
    q = {pop: MU_SQ[pop] / (MU_SQ[pop] + SIGMA_SQ[pop]) for pop in (True, False)}
    at4 = result["per_n"][4]
    assert close(at4["cos_correct"], c4[True])
    assert close(at4["cos_incorrect"], c4[False])
    assert close(at4["cos_cross"], COS_MU * math.sqrt(c4[True] * c4[False]))
    assert close(at4["cos_cross_corrected"], COS_MU)
    for pop, primary, secondary in ((True, "primary_correct", "secondary_ols_correct"),
                                    (False, "primary_incorrect", "secondary_ols_incorrect")):
        fit = result[primary]
        assert close(fit["c_4"], c4[pop])
        assert close(fit["q"], q[pop])
        assert close(fit["mean_sq_norm"], MU_SQ[pop] + SIGMA_SQ[pop])
        assert close(fit["mu_sq"], MU_SQ[pop])
        assert close(fit["sigma_sq"], SIGMA_SQ[pop])
        assert close(fit["snr_1"], math.sqrt(MU_SQ[pop] / SIGMA_SQ[pop]))
        for n, check in fit["c_n_check"].items():
            expected = MU_SQ[pop] / (MU_SQ[pop] + SIGMA_SQ[pop] / n)
            assert close(check["predicted"], expected)
            assert close(check["measured"], expected)
        ols = result[secondary]
        assert close(ols["mu_sq"], MU_SQ[pop])
        assert close(ols["sigma_sq"], SIGMA_SQ[pop])
        assert close(ols["q"], q[pop])
    assert close(result["q_ratio_incorrect_over_correct"], q[False] / q[True])

    m = p * mean_vector(True) + (1 - p) * mean_vector(False)
    mixture = result["mixture"]
    assert close(mixture["cos_mu_correct_mu_incorrect"], COS_MU)
    assert close(mixture["phi_angle_mixture_incorrect_deg"], angle_deg(m, mean_vector(False)))
    assert close(mixture["angle_mixture_correct_deg"], angle_deg(m, mean_vector(True)))
    assert mixture["phi_angle_mixture_incorrect_deg"] + mixture["angle_mixture_correct_deg"] == pytest.approx(
        mixture["theta_deg"], rel=1e-9)


def unit_records(pool):
    """Per prompt: Gram record of the fine-class sums of unit gradients, and the unit vectors themselves."""
    rows_by_prompt = {}
    for r in pool:
        rows_by_prompt.setdefault(r["prompt"], []).append(r)
    out = []
    for rows in rows_by_prompt.values():
        sums, counts, units = {}, Counter(), []
        for r in rows:
            g = gradient(r["id"], r["correct"])
            norm = math.sqrt(stats.dot(g, g))
            unit = [x.double() / norm for x in g]
            key = fine_class(r)
            sums[key] = unit if key not in sums else [s + u for s, u in zip(sums[key], unit)]
            counts[key] += 1
            units.append((r, torch.cat(unit)))
        out.append((stats.prompt_scalars(sums, counts), units))
    return out


def test_within_prompt_unit_sums_equal_direct_pairwise_means():
    selections = {"pooled": lambda r: True, "flag_f1": lambda r: r["flag"] == "f1", "flag_f0": lambda r: r["flag"] == "f0"}
    for rec, units in unit_records(make_pool(prompts=6)):
        for view, keep in selections.items():
            select = ({"C": [k for k in rec["counts"] if k.startswith("C_")], "I": [k for k in rec["counts"] if k.startswith("I_")]}
                      if view == "pooled" else {"C": ["C_" + view[-2:]], "I": ["I_" + view[-2:]]})
            merged = stats.merge_classes(rec, select)
            by = {"C": [u for r, u in units if r["correct"] and keep(r)], "I": [u for r, u in units if not r["correct"] and keep(r)]}
            for a, b in (("C", "C"), ("I", "I"), ("C", "I")):
                pairs = [float(x @ y) for i, x in enumerate(by[a]) for j, y in enumerate(by[b]) if (a != b or i < j)]
                if pairs:
                    assert stats.pair_mean(merged, a, b) == pytest.approx(sum(pairs) / len(pairs), rel=1e-9, abs=1e-12)
                else:
                    assert math.isnan(stats.pair_mean(merged, a, b))


def test_within_prompt_statistics_match_closed_form():
    out = stats.summarize_prompts([rec for rec, _ in unit_records(make_pool())])
    q = {pop: MU_SQ[pop] / (MU_SQ[pop] + SIGMA_SQ[pop]) for pop in (True, False)}
    pooled = out["pooled"]["mixed"]
    assert close(pooled["cc"]["mean"], q[True])
    assert close(pooled["ii"]["mean"], q[False])
    assert close(pooled["ci"]["mean"], COS_MU * math.sqrt(q[True] * q[False]))
    assert close(pooled["q_ratio_incorrect_over_correct"], q[False] / q[True])
    assert pooled["cc"]["prompts"] > 100
    assert sum(out["pooled"]["prompt_counts"].values()) == 375
    assert out["flag_f1"]["prompt_counts"] == out["pooled"]["prompt_counts"]
    assert 0 < out["flag_f1"]["mixed"]["cc"]["prompts"] < pooled["cc"]["prompts"]


def test_prompt_class():
    assert stats.prompt_class(2, 2) == "mixed"
    assert stats.prompt_class(8, 0) == "all_correct"
    assert stats.prompt_class(0, 8) == "all_wrong"
    assert stats.prompt_class(1, 7) == "other"
    assert stats.prompt_class(7, 1) == "other"


def test_cross_teacher_and_reading_rule_summaries():
    rows = [{"correct": c, "prompt_class": k, "flag": f, "cos": {"O1|O2": v, "O3|O4": -v}}
            for c, k, f, v in ((True, "mixed", "f1", 0.2), (False, "mixed", "f0", 0.4), (False, "all_wrong", "f1", 0.6))]
    cross = stats.summarize_cross(rows)
    assert cross["all"]["all"]["O1|O2"] == {"count": 3, "mean": pytest.approx(0.4), "median": 0.4}
    assert cross["correctness"]["I"]["O3|O4"]["mean"] == pytest.approx(-0.5)
    assert cross["correctness_prompt_class"]["I|mixed"]["O1|O2"]["count"] == 1
    assert set(cross["flag"]) == {"f0", "f1"}

    teacher = stats.summarize_teacher([{"correct": True, "flag": "f1", "gap": 0.0}, {"correct": False, "flag": "f0", "gap": -1.5},
                                       {"correct": False, "flag": "fx", "gap": None}])
    assert teacher["I|f0"]["mean"] == -1.5
    assert teacher["I|fx"] == {"count": 0, "mean": teacher["I|fx"]["mean"], "median": teacher["I|fx"]["median"], "trajectories": 1}

    trials = {"primary_correct": {"q": 0.1}, "primary_incorrect": {"q": 0.25}, "q_ratio_incorrect_over_correct": 2.5}
    prompts = {"pooled": {"mixed": {"cc": {"mean": 0.1}, "ii": {"mean": 0.12}, "q_ratio_incorrect_over_correct": 1.2}}}
    checks = stats.reading_rule_checks(trials, prompts)
    assert checks["across_ratio_ge_2.0"] and checks["within_ratio_lt_1.5"] and not checks["within_ratio_ge_2.0"]
    assert not checks["across_q_correct_gt_q_incorrect"] and not checks["within_q_correct_gt_q_incorrect"]
