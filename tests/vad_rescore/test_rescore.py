"""HANDOFF-VLM-VAD-GRAD-002-R1 Part A item 6: the scorer and the statistics code against direct computations, and the
resume logic."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[2] / "scripts" / "vad_rescore"


def load(name: str):
    """Load scripts/vad_rescore/<name>.py as vad_rescore_<name> (scripts/vad_grad has its own stats module)."""
    spec = importlib.util.spec_from_file_location(f"vad_rescore_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = load("score")
stats = load("stats")

V, LEN, H = 12, 5, 4


class ToyActor:
    """_forward_micro_batch with the dp_actor semantics: log_probs of the responses and topk_logps at given indices,
    both log-softmax over the full vocabulary."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = torch.nn.Linear(H, V, bias=False)

    def _forward_micro_batch(self, mi, temperature, calculate_entropy=False, topk_indices=None):
        logits = self.model(mi["input_ids"].reshape(mi["input_ids"].shape[0], LEN, H)) / temperature
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        out = {"log_probs": (logits.gather(-1, mi["responses"].unsqueeze(-1)) - lse).squeeze(-1)}
        if topk_indices is not None:
            out["topk_logps"] = logits.gather(-1, topk_indices) - lse
        return out


def test_scores_equal_direct_log_softmax():
    gen = torch.Generator().manual_seed(1)
    actor = ToyActor()
    mi = {"input_ids": torch.randn(3, LEN * H, generator=gen), "responses": torch.randint(0, V, (3, LEN), generator=gen),
          "response_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])}
    ids, positions = [2, 5, 7], [4, None, 0]
    recs, first = score.score_forward(actor, mi, ids, positions)
    with torch.no_grad():
        logp = torch.log_softmax(actor.model(mi["input_ids"].reshape(3, LEN, H)).double(), dim=-1)
    for i in range(3):
        n = int(mi["response_mask"][i].sum())
        tok = [float(logp[i, j, int(mi["responses"][i, j])]) for j in range(LEN)]
        assert recs[i]["L"] == pytest.approx(sum(tok[:n]), abs=1e-5)
        assert recs[i]["L1"] == pytest.approx(tok[0], abs=1e-5)
        assert recs[i]["response_tokens"] == n
        if positions[i] is None:
            assert math.isnan(recs[i]["Lc"])
        else:
            assert recs[i]["Lc"] == pytest.approx(tok[positions[i]], abs=1e-5)
    assert first == pytest.approx([float(logp[0, 0, k].exp()) for k in ids], abs=1e-6)


def test_first_token_record():
    rec = score.first_token_record(3, "B", {"A": 10, "B": 11, "C": 12, "D": 13}, [0.1, 0.4, 0.2, 0.05])
    assert rec["P_letter"] == pytest.approx(0.75) and rec["P_gt"] == pytest.approx(0.4)
    assert rec["P_wrong"] == pytest.approx(0.35) and rec["cause"] is None
    assert score.first_token_record(3, "B", {}, None)["cause"] == "no option letter"


def test_option_letters():
    prompt = [{"role": "user", "content": "<image>\nQ?\n\nA. concrete\nB. sand\nC. metal\nD. water\n\nAnswer with the option's letter."}]
    assert score.option_letters(prompt) == ["A", "B", "C", "D"]


def test_option_letters_content_list():
    """RLHFDataset._build_messages gives the user content as a list of image and text parts (pilot TASK-002/003)."""
    prompt = [{"role": "user", "content": [{"type": "image", "image": "img"},
                                           {"type": "text", "text": "\nQ?\n\nA. concrete\nB. sand\nC. metal\nD. water\n"}]}]
    assert score.option_letters(prompt) == ["A", "B", "C", "D"]


def test_d_per_prompt_by_hand():
    meta = {0: (0, 1, 3), 1: (0, 1, 20), 2: (0, 0, 2), 3: (1, 1, 2), 4: (1, 0, 9), 5: (1, 0, 2), 6: (2, 1, 1), 7: (2, 1, 1)}
    meta = {t: {"prompt": p, "label": lab, "response_tokens": n} for t, (p, lab, n) in meta.items()}
    delta = {0: 1.0, 1: 3.0, 2: 0.0, 3: 2.0, 4: 1.0, 5: -1.0, 6: 5.0, 7: 5.0}
    assert stats.per_prompt_d(delta, meta) == {0: 2.0, 1: 2.0}  # prompt 2 has no incorrect sibling
    assert stats.per_prompt_d(delta, meta, stats.is_short) == {0: 1.0, 1: 3.0}  # short: traj 0, 2, 3, 5


def test_bootstrap_by_hand():
    # Two prompts with d = 1 and 3: a resample mean is 1, 2 or 3 with probabilities 1/4, 1/2, 1/4.
    res = stats.mean_over_prompts([1.0, 3.0], 20000)
    assert res["value"] == 2.0 and res["ci"] == [1.0, 3.0] and res["count"] == 2
    # Paired: the per-prompt difference is constant, so the interval has width 0.
    a, b = [1.0, 5.0, 2.0], [0.0, 4.0, 1.0]
    paired = stats.mean_over_prompts([x - y for x, y in zip(a, b)], 2000)
    assert paired["ci"] == [1.0, 1.0]
    assert stats.mean_over_prompts(a, 2000)["ci"][1] - stats.mean_over_prompts(a, 2000)["ci"][0] > 1.0
    # A resample is a multiset of the prompts: each row of the counts matrix sums to n.
    c = stats.counts_matrix(7, 50)
    assert c.shape == (50, 7) and np.all(c.sum(1) == 7)
    assert np.array_equal(c, stats.counts_matrix(7, 50))


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def synthetic_dir(tmp_path: Path):
    """2 prompts x 4 siblings per set. Scores: L = base + model shift + 1 for correct siblings of trained models."""
    labels = {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0}
    shift = {"M0": 0.0, "A1_80": 1.0, "A1_130": 2.0, "A2_80": 0.5, "A2_130": 1.5}
    gain = {"M0": 0.0, "A1_80": 1.0, "A1_130": 3.0, "A2_80": 0.5, "A2_130": 2.0}
    for s, models in (("S0", list(shift)), ("S80", list(shift)[1:])):
        for m in models:
            rows = []
            for t, lab in labels.items():
                base = -10.0 - t
                L = base + shift[m] + gain[m] * lab
                rows.append({"traj": t, "prompt": t // 4, "sibling": t % 4, "label": lab, "response_tokens": 2 if t % 2 else 8,
                             "L": L, "L1": L / 4, "Lc": L / 2 if t != 7 else math.nan})
            write_jsonl(tmp_path / f"SCORES_{s}_{m}.jsonl", rows)
    pgt = {"M0": 0.4, "A1_80": 0.6, "A1_130": 0.7, "A2_80": 0.5, "A2_130": 0.5}
    for m in shift:
        write_jsonl(tmp_path / f"FIRSTTOK_{m}.jsonl",
                    [{"prompt": p, "P_letter": 0.8, "P_gt": pgt[m] * (1 + p), "P_wrong": 0.8 - pgt[m] * (1 + p), "cause": None}
                     for p in (0, 1)] + [{"prompt": 2, "P_letter": None, "P_gt": None, "P_wrong": None, "cause": "no option letter"}])
    traj = {s: [{"traj": t, "logp_sum": -10.0 - t + (0.001 if t == 3 else 0.0)} for t in labels] for s in ("S0", "S80")}
    write_jsonl(tmp_path / "TRAJ0.jsonl", traj["S0"])
    write_jsonl(tmp_path / "TRAJ80.jsonl", [{"traj": t, "logp_sum": -10.0 - t + 1.0 + lab} for t, lab in labels.items()])
    return gain


def test_statistics_by_hand(tmp_path):
    gain = synthetic_dir(tmp_path)
    out = stats.compute(tmp_path, {"S0": str(tmp_path / "TRAJ0.jsonl"), "S80": str(tmp_path / "TRAJ80.jsonl")}, 500)
    # Delta = shift + gain x label, so d_p = gain(M) - gain(ref) on each contrast prompt.
    assert out["D"]["A1_80|S0"]["value"] == pytest.approx(1.0) and out["D"]["A1_80|S0"]["count"] == 2
    assert out["D"]["A1_130|S80"]["value"] == pytest.approx(gain["A1_130"] - gain["A1_80"])
    assert out["D"]["A2_130|S80"]["value"] == pytest.approx(gain["A2_130"] - gain["A2_80"])
    assert out["D"]["A1_130|S0"]["ci"] == pytest.approx([3.0, 3.0])
    assert out["D_forms"]["A1_80|S0"]["L1"]["value"] == pytest.approx(0.25)
    assert out["D_forms"]["A1_80|S0"]["Lc"]["value"] == pytest.approx(0.5)
    # Per token (Delta 2 for correct, 1 for incorrect; odd traj 2 tokens, even 8 tokens):
    # prompt 0: mean(2/8, 2/2) - mean(1/8, 1/2) = 0.3125; prompt 1: 2/8 - mean(1/2, 1/8, 1/2) = -0.125.
    assert out["D_forms"]["A1_80|S0"]["L_per_token"]["value"] == pytest.approx((0.3125 - 0.125) / 2)
    # Short siblings (odd traj): prompt 0 has 1 (C) and 3 (I); prompt 1 has 5 and 7 (I only).
    assert out["D_length"]["A1_80|S0"]["short"]["count"] == 1
    assert out["mean_delta"]["A1_80|S0"]["C|all"]["value"] == pytest.approx(2.0)
    assert out["mean_delta"]["A1_80|S0"]["I|all"]["value"] == pytest.approx(1.0)
    assert out["D_arm_difference"]["A1_80-A2_80|S0"]["value"] == pytest.approx(0.5)
    ft = out["first_token"]["A1_80-M0"]
    assert ft["P_gt"]["value"] == pytest.approx(((0.6 - 0.4) + (1.2 - 0.8)) / 2)
    assert ft["R_gt"]["value"] == pytest.approx(((0.6 - 0.4) / 0.8 + (1.2 - 0.8) / 0.8) / 2)
    assert ft["R_gt"]["prompts_defined"] == 2 and out["first_token"]["excluded_prompts"]["M0"] == [2]
    assert out["self_score"]["M0|S0"]["max_abs"] == pytest.approx(0.001)
    assert out["self_score"]["A1_80|S80"]["max_abs"] == pytest.approx(0.0)  # TRAJ80 equals the A1_80 scores
    assert out["counts"]["S0|M0"]["Lc_nan"] == 1
    # Delta_A1_80 and Delta_A2_80 are both affine in the label: correlation 1.
    assert out["correlation"]["A1_80~A2_80"]["value"] == pytest.approx(1.0)


def test_resume_gives_the_same_outputs(tmp_path):
    rows = [{"traj": t, "prompt_pos": t // 2, "sibling": t % 2, "prompt_len": 10, "response_ids": [1, 2]} for t in range(8)]
    by_prompt = {p: [r for r in rows if r["prompt_pos"] == p] for p in range(4)}
    labels = {t: t % 2 for t in range(8)}
    positions = {0: 1, 5: 0}
    calls = []

    def fake(p, rs, pos):
        calls.append(p)
        recs = [{"L": -1.0 * r["traj"], "L1": -0.5, "Lc": -0.1 if q is not None else math.nan, "response_tokens": 2,
                 "batch_size": 2} for r, q in zip(rs, pos)]
        return recs, {"prompt": p, "cause": None}

    ref = tmp_path / "ref"
    ref.mkdir()
    score.run_set("S0", "M0", ref, by_prompt, list(range(4)), labels, positions, fake, {"prompts": 0}, 0)
    run = tmp_path / "run"
    run.mkdir()
    calls.clear()
    with pytest.raises(score.Stop):
        score.run_set("S0", "M0", run, by_prompt, list(range(4)), labels, positions, fake, {"prompts": 0}, 2)
    assert calls == [0, 1] and not (run / "SCORES_S0_M0.jsonl").exists()
    calls.clear()
    score.run_set("S0", "M0", run, by_prompt, list(range(4)), labels, positions, fake, {"prompts": 0}, 0)
    assert calls == [2, 3]
    for name in ("SCORES_S0_M0.jsonl", "FIRSTTOK_M0.jsonl"):
        assert (run / name).read_text() == (ref / name).read_text()
    recs = [json.loads(line) for line in open(run / "SCORES_S0_M0.jsonl")]
    assert [r["traj"] for r in recs] == list(range(8))
    assert recs[1]["Lc_cause"] == score.NO_POSITION and recs[0]["Lc_cause"] is None
    assert not list(run.glob("*.tmp"))
