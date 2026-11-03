"""G3 (HANDOFF-VLM-VAD-GRAD-001-R1; DIRECTIVE-VLM-VAD-GRAD-001-002 D7): estimators, folds, resume, 12-from-28 Gram."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_grad"))
import accum  # noqa: E402
import stats  # noqa: E402
import synthetic  # noqa: E402


@pytest.fixture(scope="module")
def drawn():
    truth = synthetic.make_truth()
    data = synthetic.draw(truth, seed=0)
    return truth, data, stats.compute(data)


def within(est: dict, true: float, k: float = 3.0) -> bool:
    return abs(est["value"] - true) <= k * est["se"]


@pytest.mark.parametrize("metric", ["raw", "D"])
@pytest.mark.parametrize("cls", ["C", "I"])
def test_moments_within_3_se(drawn, metric, cls):
    truth, _, out = drawn
    got = out["metrics"][metric]["subsets"][cls]
    true = synthetic.true_values(truth, cls, metric)
    for key in ("m", "E2", "q", "sigma_b2", "sigma_w2", "ICC", "s_same"):
        assert within(got[key], true[key]), (key, got[key]["value"], got[key]["se"], true[key])


@pytest.mark.parametrize("cls", ["C", "I"])
def test_m_all_has_the_predicted_same_prompt_bias(drawn, cls):
    truth, _, out = drawn
    got = out["metrics"]["raw"]["subsets"][cls]
    true = synthetic.true_values(truth, cls, "raw")
    share = got["same_pair_share"]["value"]
    predicted = true["m"] + share * true["sigma_b2"]
    assert within(got["m_all"], predicted)
    assert predicted - true["m"] > 3 * got["m"]["se"]  # the bias is visible at this ICC and q


@pytest.mark.parametrize("metric", ["raw", "D"])
def test_cross_fold_within_3_se(drawn, metric):
    truth, _, out = drawn
    cross = out["metrics"][metric]["cross_fold"]
    for key, value in synthetic.true_cross(truth, metric).items():
        assert cross[key]["se_rule"] == "jackknife"
        assert within(cross[key], value), (key, cross[key]["value"], cross[key]["se"], value)


def test_alignment_sign_follows_the_step_direction(drawn):
    """HANDOFF-VLM-VAD-GRAD-001-R2 Correction item 1: a_X = <-mu_X, g_acc> / norms. The truth has cos(mu_C, nu) = 0.5
    and cos(mu_I, nu) = -0.2, so a step on C (along -mu_C) lowers the accuracy (a_C < 0) and a step on I raises it."""
    _, _, out = drawn
    cross = out["metrics"]["raw"]["cross_fold"]
    assert cross["a|C"]["value"] < -3 * cross["a|C"]["se"]
    assert cross["a|I"]["value"] > 3 * cross["a|I"]["se"]


def test_folds_are_disjoint_and_cover_all_prompts():
    folds = accum.fold_of(list(range(375)))
    assert sorted(folds) == list(range(375))
    sizes = [sum(1 for f in folds.values() if f == a) for a in range(4)]
    assert sorted(sizes) == [93, 94, 94, 94]
    assert accum.fold_of(list(range(375))) == folds


def test_counts_match_siblings(drawn):
    _, data, out = drawn
    assert out["counts"]["trajectories"] == 3000
    s_total = sum(v for k, v in data["counts"].items() if k.startswith("S|"))
    assert s_total == 3000
    assert sum(data["counts"][f"A|{a}"] for a in range(4)) == 375


def test_gram12_equals_direct_computation(drawn):
    _, data, _ = drawn
    gr = stats.Gram(data)
    names12, g12 = stats.gram12(gr, "raw")
    # Direct: rebuild the 12 vectors from a fresh synthetic run is costly; use the linear identity on the stored
    # 28 x 28 Gram against an independent sum over the 28 names.
    index = {n: i for i, n in enumerate(data["names"])}
    for u, nu in enumerate(names12):
        for v, nv in enumerate(names12):
            us = [n for n in data["names"] if n == nu or n.startswith(nu + "|")]
            vs = [n for n in data["names"] if n == nv or n.startswith(nv + "|")]
            direct = sum(data["raw"][index[a]][index[b]] for a in us for b in vs)
            assert g12[u][v] == pytest.approx(direct, rel=1e-12, abs=1e-9)


def test_gram12_equals_vectors_summed_first():
    """The 12-vector Gram from the 28-vector Gram equals the Gram of the 12 summed vectors."""
    torch.manual_seed(0)
    shapes = [torch.Size([5, 3]), torch.Size([7])]
    store = accum.Store(shapes)
    for name in accum.vector_names():
        store.add(name, [torch.randn(s) for s in shapes], 1)
    d = [torch.rand(s) + 0.5 for s in shapes]
    data = accum.gram_file(store, d, "cpu", [], "t")
    for name in data["names"]:
        data["counts"][name] = 0
    names12, g12 = stats.gram12(stats.Gram(data), "D")
    summed = {}
    for n12 in names12:
        parts = [n for n in data["names"] if n == n12 or n.startswith(n12 + "|")]
        summed[n12] = [sum(store.get(p)[t].double() for p in parts) for t in range(len(shapes))]
    for u, nu in enumerate(names12):
        for v, nv in enumerate(names12):
            assert g12[u][v] == pytest.approx(accum.dot(summed[nu], summed[nv], d), rel=1e-10, abs=1e-12)


def test_resume_gives_the_same_accumulators(tmp_path):
    """Save after some prompts, drop the state, restore, continue: equal to one run without a stop."""
    torch.manual_seed(1)
    shapes = [torch.Size([4, 2]), torch.Size([3])]
    d = [torch.rand(s) + 0.5 for s in shapes]
    prompts = list(range(12))
    folds = accum.fold_of(prompts)
    order = sorted(prompts, key=lambda p: (folds[p], p))
    data = {p: ([[torch.randn(s) for s in shapes] for _ in range(4)], [1, 0, 1, 1], ["f1", "f0", "fx", "f1"]) for p in prompts}

    def run(store, progress, todo, records, stop_after=None):
        buf = [torch.zeros(s) for s in shapes]
        for n, p in enumerate(todo):
            grads, labels, flags = data[p]
            records.append(accum.finish_prompt(store, p, folds[p], [p], labels, flags, grads, d, grads[0], buf))
            if progress is not None and (n + 1) % 3 == 0:
                progress.save(store, records)
            if stop_after is not None and n + 1 == stop_after:
                return

    ref = accum.Store(shapes)
    run(ref, None, order, [])
    store, records = accum.Store(shapes), []
    progress = accum.Progress(tmp_path)
    run(store, progress, order, records, stop_after=7)  # saved after 6 prompts; prompt 7 is lost
    store2 = accum.Store(shapes)
    progress2 = accum.Progress(tmp_path)
    progress2.restore(store2)
    records2 = list(progress2.state["prompts"])
    assert len(records2) == 6
    run(store2, progress2, [p for p in order if p not in progress2.done], records2)
    for name in accum.vector_names():
        if name in ref.vec:
            for a, b in zip(ref.get(name), store2.get(name)):
                assert torch.allclose(a, b, atol=1e-6)
            assert ref.count[name] == store2.count[name]
    assert sorted(r["prompt"] for r in records2) == prompts
    files = sorted(p.name for p in tmp_path.glob("*.safetensors"))
    assert len(files) == len(set(f.split(".v")[0] for f in files))  # one version of each vector is left
