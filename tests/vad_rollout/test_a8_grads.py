"""HANDOFF-VLM-VAD-ROLLOUT-001-R9 Part A: the bookkeeping of a8/grads.py on synthetic vectors (CPU, no model)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

A8 = Path(__file__).resolve().parents[2] / "scripts" / "vad_rollout" / "a8"
sys.path.insert(0, str(A8))
spec = importlib.util.spec_from_file_location("a8_grads", A8 / "grads.py")
grads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grads)
stats = grads.stats

TEST_STATS = Path(__file__).resolve().parent / "test_a8_stats.py"
spec = importlib.util.spec_from_file_location("a8_test_stats", TEST_STATS)
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)


class CpuPool(grads.HostPool):
    def take(self, src):
        return [x.clone() for x in src]


def test_gram_matches_direct_dots():
    vecs = {k: synthetic.gradient(i, True) for i, k in enumerate(("C1", "C2", "I1", "I2"))}
    gram = grads.Gram(CpuPool([v.shape for v in vecs["C1"]]))
    for i, (k, v) in enumerate(vecs.items()):
        gram.add(k, v, keep=i < 3)
    assert set(gram.host) == {"C1", "C2", "I1"}
    for a in vecs:
        for b in vecs:
            key = "|".join(sorted((a, b)))
            assert gram.values[key] == pytest.approx(stats.dot(vecs[a], vecs[b]), rel=1e-9)


def test_summarize_on_synthetic_pass_files(tmp_path):
    pool = synthetic.make_pool(prompts=375)
    rows = [{"traj": r["id"], "prompt_pos": r["prompt"], "correct": r["correct"]} for r in pool]
    (tmp_path / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    flag = {r["id"]: r["flag"] for r in pool}
    (tmp_path / "teacher_0.jsonl").write_text("".join(
        json.dumps({"traj": r["traj"], "correct": r["correct"], "flag": flag[r["traj"]], "gap": 0.0 if r["correct"] else -1.0}) + "\n"
        for r in rows))
    by_prompt = {}
    for r in pool:
        by_prompt.setdefault(r["prompt"], []).append(r)
    with open(tmp_path / "within_O4_0.jsonl", "w") as f:
        for p, members in by_prompt.items():
            classes = {}
            for r in members:
                classes.setdefault(synthetic.fine_class(r), []).append(r["id"])
            gram, norms = grads.Gram(CpuPool(None)), {}
            for key in sorted(classes):
                acc = None
                for t in classes[key]:
                    g = synthetic.gradient(t, pool[t]["correct"])
                    norms[t] = math.sqrt(stats.dot(g, g))
                    unit = [x.double() / norms[t] for x in g]
                    acc = unit if acc is None else [a + u for a, u in zip(acc, unit)]
                gram.add(key, acc)
            f.write(json.dumps({"prompt": p, "counts": {k: len(v) for k, v in classes.items()}, "gram": gram.values,
                                "norms": {str(t): v for t, v in norms.items()}, "losses": {}}) + "\n")
    correct_of = {r["id"]: r["correct"] for r in pool}
    plan = stats.trial_plan(grads.pool_rows(type("C", (), {"rows": rows})(), flag, "pooled"), grads.TRIAL_SEED)
    with open(tmp_path / "trials_O4_pooled_0.jsonl", "w") as f:
        f.write(json.dumps({"plan": [{"n": e["n"], "count": e["count"], "largest_n": e["largest_n"], "possible": True} for e in plan]}) + "\n")
        for e in plan:
            for trial in e["trials"]:
                f.write(json.dumps(stats.trial_scalars({g: synthetic.group_sum(trial[g], correct_of) for g in stats.GROUPS}, e["n"])) + "\n")
    with open(tmp_path / "cross_0.jsonl", "w") as f:
        for r in rows[:50]:
            f.write(json.dumps({"traj": r["traj"], "norms": {}, "cos": {"O1|O4": 0.1, "O3|O4": 0.9}}) + "\n")

    grads.summarize(tmp_path, str(tmp_path / "rollouts.jsonl"))
    out = json.loads((tmp_path / "STATS.json").read_text())
    q = {pop: synthetic.MU_SQ[pop] / (synthetic.MU_SQ[pop] + synthetic.SIGMA_SQ[pop]) for pop in (True, False)}
    o4 = out["objectives"]["O4"]
    assert o4["trials"]["pooled"]["primary_correct"]["q"] == pytest.approx(q[True], rel=0.02)
    assert o4["trials"]["pooled"]["primary_incorrect"]["q"] == pytest.approx(q[False], rel=0.02)
    assert o4["within"]["pooled"]["mixed"]["cc"]["mean"] == pytest.approx(q[True], rel=0.02)
    assert o4["reading_rule"]["across_ratio"] == pytest.approx(q[False] / q[True], rel=0.03)
    assert out["cross"]["all"]["all"]["O3|O4"]["count"] == 50
    assert out["fx"]["count"] == sum(1 for r in pool if r["flag"] == "fx")
    assert out["trajectories"] == 3000
