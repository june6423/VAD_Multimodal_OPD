"""Tests of HANDOFF-VLM-VAD-PACK-001-R1 Part A item 7.

The toy actor replaces DataParallelPPOActor._forward_micro_batch with a linear model over a small vocabulary (the
semantics of test_o4_equality.py). When one forward holds more than one row, row j gets a state term from rows 0 to j-1:
the cross-row state of the packed forward. Row 1 of a forward has no such term. The O4 loss is the real
compute_self_distillation_loss with the vopd config. The modules of scripts/vad_pack load by file path, because
scripts/vad_grad and scripts/vad_rollout/a8 have their own stats modules.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = ROOT / "scripts" / "vad_pack"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACK_DIR))


def load(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pack = load("vad_pack_pack", PACK_DIR / "pack.py")
split = load("vad_pack_split", PACK_DIR / "split.py")
pstats = load("vad_pack_stats", PACK_DIR / "stats.py")
pilot = load("vad_pack_pilot", PACK_DIR / "pilot.py")

V, L, H = 16, 6, 8
RANKS, B_MINI = 6, 128  # A1 values (trainer_values.py)


class ToyActor:
    def __init__(self, policy_config):
        torch.manual_seed(0)
        self.student = torch.nn.Linear(H, V, bias=False)
        self.teacher_module = torch.nn.Linear(H, V, bias=False).requires_grad_(False)
        self.config = policy_config

    def _forward_micro_batch(self, inputs, temperature, calculate_entropy=False, return_all_logps=False,
                             distill_topk=None, topk_indices=None, return_response_logits=False, module=None):
        model = module or self.student
        feats = inputs["input_ids"].float().reshape(-1, L, H)
        if feats.shape[0] > 1:  # packed rows: state from the earlier rows of the forward
            prev = torch.cumsum(feats.mean(dim=1), dim=0)
            feats = feats + 0.5 * torch.cat([torch.zeros_like(prev[:1]), prev[:-1]]).unsqueeze(1)
        logits = model(feats) / temperature
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        out = {"log_probs": (logits.gather(-1, inputs["responses"].unsqueeze(-1)) - lse).squeeze(-1)}
        if distill_topk is not None or topk_indices is not None:
            k = min(distill_topk or 0, V)
            idx = torch.topk(logits, k, dim=-1).indices if topk_indices is None else topk_indices
            out["topk_logps"] = logits.gather(-1, idx) - lse
            if topk_indices is None:
                out["topk_indices"] = idx
        if return_response_logits:
            out["response_logits"] = logits
        return out


class ToyCtx:
    def __init__(self, actor, data: dict[int, dict]):
        self.actor = actor
        self.params = list(actor.student.parameters())
        self.device = "cpu"
        self.classes = pack.make_capture(pack._objectives())
        self.data = data

    def inputs(self, ros):
        rows = [self.data[ro["traj"]] for ro in ros]
        mi = {k: (torch.cat([r[k] for r in rows]) if torch.is_tensor(rows[0][k]) else None) for k in rows[0]}
        return mi, [dict(r) for r in rows]


def toy_row(gen, n_tokens: int) -> dict:
    r = {"responses": torch.randint(0, V, (1, L), generator=gen),
         "response_mask": (torch.arange(L) < n_tokens).float().unsqueeze(0)}
    for prefix in ("", "teacher_", "teacher_minus_"):
        r[prefix + "input_ids"] = torch.randn(1, L * H, generator=gen)
        r[prefix + "attention_mask"] = torch.ones(1, L)
        r[prefix + "position_ids"] = torch.zeros(1, L, dtype=torch.long)
        r[prefix + "response_start_idx"] = torch.zeros(1, dtype=torch.long)
        r[prefix + "multi_modal_inputs"] = None
    return r


@pytest.fixture(scope="module")
def setup():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from verl.utils.config import omega_conf_to_dataclass

    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="vopd")
    OmegaConf.resolve(cfg)
    actor = ToyActor(omega_conf_to_dataclass(cfg.actor_rollout_ref.actor))
    gen = torch.Generator().manual_seed(1)
    data = {t: toy_row(gen, n) for t, n in enumerate([6, 3, 5, 4, 2, 6])}
    return actor, data


def split_rows(batch: str, micro_sizes: list[int], start: int = 0) -> list[dict]:
    """Split rows of one batch: micro-batches of the given sizes on rank 0, trajectories start, start + 1, ..."""
    rows, t = [], start
    for u, size in enumerate(micro_sizes):
        for pos in range(1, size + 1):
            rows.append({"batch": batch, "arm": "A1", "set": "S0", "rank": 0, "mini": 0, "micro": u, "position": pos,
                         "micro_rows": size, "traj": t, "prompt": t, "sibling": 0, "label": t % 2, "total_tokens": 100 + t,
                         "response_tokens": L})
            t += 1
    return rows


def rollout_lp(t: int, n: int) -> list[float]:
    """Synthetic TASK-053 rollout_logprobs of trajectory t with n response tokens."""
    return [-0.1 * (k + 1) - 0.01 * t for k in range(n)]


def run(ctx, out: Path, rows: list[dict], **opts):
    out.mkdir(parents=True, exist_ok=True)
    o = {"pilot": None, "one_row_check": False, "stop_after": 0, "commit_seconds": 1e9, **opts}
    rollouts = {r["traj"]: {"traj": r["traj"], "rollout_logprobs": rollout_lp(
        r["traj"], int(ctx.data[r["traj"]]["response_mask"].sum()))} for r in rows}
    return pack.run_batch(ctx, rows[0]["batch"], rows, rollouts, out, o, {"micro": 0})


def objectives(actor):
    return pack._objectives().Objectives(actor, list(actor.student.parameters()))


def records(out: Path, batch: str) -> list[dict]:
    return [json.loads(line) for line in open(out / f"PACKROWS_{batch}.jsonl")]


def test_arm_u_rows_equal_one_row_forward(setup, tmp_path):
    """Item 7.1: each arm U row value equals a forward of that row alone."""
    actor, data = setup
    ctx = ToyCtx(actor, data)
    rows = split_rows("A1-S0-1", [3, 2, 1])
    run(ctx, tmp_path, rows, one_row_check=True)
    ob = objectives(actor)
    for rec in records(tmp_path, "A1-S0-1"):
        mi = data[rec["traj"]]
        with torch.no_grad():
            o4 = float(ob.losses(mi, ["O4"])["O4"])
            lp = actor._forward_micro_batch(mi, temperature=1.0)["log_probs"]
        assert rec["O4_U"] == pytest.approx(o4, rel=1e-6, abs=1e-7)
        assert rec["L_U"] == pytest.approx(float((lp * mi["response_mask"]).sum()), rel=1e-6, abs=1e-7)
        assert rec["L_1row"] == pytest.approx(rec["L_U"], rel=1e-6, abs=1e-7)
        assert rec["O4_1row"] == pytest.approx(rec["O4_U"], rel=1e-6, abs=1e-7)


def test_arm_p_equals_arm_u_for_one_row(setup, tmp_path):
    """Item 7.3: with one row in the micro-batch, arm P equals arm U (values and gradient). With more rows, position 1
    is equal and position 2 differs (the toy state term)."""
    actor, data = setup
    ctx = ToyCtx(actor, data)
    rows = split_rows("A1-S0-1", [1])
    run(ctx, tmp_path, rows)
    rec = records(tmp_path, "A1-S0-1")[0]
    assert rec["L_P"] == rec["L_U"] and rec["O4_P"] == rec["O4_U"]
    assert rec["d_max"] == 0.0 and rec["kl_student_sum"] == pytest.approx(0.0, abs=1e-6)
    vp = torch.load(tmp_path / "vectors" / "A1-S0-1_P.pt")
    vu = torch.load(tmp_path / "vectors" / "A1-S0-1_U.pt")
    assert torch.allclose(vp, vu, rtol=1e-6, atol=1e-8)

    out2 = tmp_path / "two"
    run(ctx, out2, split_rows("A1-S0-1", [2]))
    r1, r2 = records(out2, "A1-S0-1")
    assert r1["L_P"] == pytest.approx(r1["L_U"], rel=1e-6) and r1["d_max"] < 1e-5
    assert abs(r2["L_P"] - r2["L_U"]) > 1e-3 and r2["kl_student_sum"] > 1e-4


def test_rollout_differences_equal_a_direct_computation(setup, tmp_path):
    """R3 Part A item 3: rdiff_X = rollout_logprobs - the student log-probability of arm X at each response token, and
    rkl_X_sum / kl_tokens equals the A1 rollout_corr/kl (compute_rollout_corr_metrics_from_logprobs) of the row."""
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

    actor, data = setup
    ctx = ToyCtx(actor, data)
    run(ctx, tmp_path, split_rows("A1-S0-1", [3]))
    members = [0, 1, 2]
    mi = {k: (torch.cat([data[t][k] for t in members]) if torch.is_tensor(data[0][k]) else None) for k in data[0]}
    with torch.no_grad():
        lp_packed = actor._forward_micro_batch(mi, temperature=1.0)["log_probs"]
    for j, rec in enumerate(records(tmp_path, "A1-S0-1")):
        t = rec["traj"]
        mask = data[t]["response_mask"]
        n = int(mask.sum())
        with torch.no_grad():
            lp_row = actor._forward_micro_batch(data[t], temperature=1.0)["log_probs"]
        rl = torch.tensor(rollout_lp(t, n), dtype=torch.float64)
        for arm, lp in (("P", lp_packed[j:j + 1]), ("U", lp_row)):
            want = rl - lp[0, :n].double()
            assert len(rec[f"rdiff_{arm}"]) == n == rec["kl_tokens"]
            assert np.allclose(rec[f"rdiff_{arm}"], want.numpy(), rtol=1e-5, atol=1e-7)
            assert rec[f"rkl_{arm}_sum"] == pytest.approx(float(want.sum()), rel=1e-6, abs=1e-7)
            rollout_full = torch.zeros_like(lp)
            rollout_full[0, :n] = rl.float()
            a1 = compute_rollout_corr_metrics_from_logprobs(log_prob=lp, rollout_log_prob=rollout_full,
                                                            response_mask=mask)["rollout_corr/kl"]
            assert rec[f"rkl_{arm}_sum"] / rec["kl_tokens"] == pytest.approx(a1, rel=1e-5, abs=1e-7)
        if rec["position"] == 1:
            assert rec["rdiff_P"] == rec["rdiff_U"]
        else:  # the toy state term of the packed forward reaches the rollout record
            assert abs(rec["rkl_P_sum"] - rec["rkl_U_sum"]) > 1e-3


def test_rollout_diff_rejects_a_length_mismatch():
    with pytest.raises(RuntimeError):
        pack.rollout_diff(torch.zeros(1, 4), [-0.1, -0.2], torch.tensor([[1.0, 1.0, 1.0, 0.0]]))


def test_gradients_follow_the_a1_aggregation(setup, tmp_path):
    """g_P = mean over ranks of grad(O4_m x B_m / B_mini); g_U = mean over ranks of the sum of
    grad(O4_i x (B_m / B_mini) x T_i / T_m)."""
    actor, data = setup
    ctx = ToyCtx(actor, data)
    rows = split_rows("A1-S0-1", [3, 2])
    run(ctx, tmp_path, rows)
    ob = objectives(actor)
    params = list(actor.student.parameters())
    gp = torch.zeros(sum(p.numel() for p in params))
    gu = torch.zeros_like(gp)
    for members in ([0, 1, 2], [3, 4]):
        mi = {k: (torch.cat([data[t][k] for t in members]) if torch.is_tensor(data[0][k]) else None) for k in data[0]}
        scale = len(members) / B_MINI
        g = torch.autograd.grad(ob.losses(mi, ["O4"])["O4"] * scale, params)
        gp += torch.cat([x.reshape(-1) for x in g]) / RANKS
        t_m = sum(float(data[t]["response_mask"].sum()) for t in members)
        for t in members:
            w = scale * float(data[t]["response_mask"].sum()) / t_m
            g = torch.autograd.grad(ob.losses(data[t], ["O4"])["O4"] * w, params)
            gu += torch.cat([x.reshape(-1) for x in g]) / RANKS
    assert torch.allclose(torch.load(tmp_path / "vectors" / "A1-S0-1_P.pt"), gp, rtol=1e-5, atol=1e-9)
    assert torch.allclose(torch.load(tmp_path / "vectors" / "A1-S0-1_U.pt"), gu, rtol=1e-5, atol=1e-9)
    prog = json.loads((tmp_path / "progress_A1-S0-1.json").read_text())
    assert prog["sq_norm"]["P"] == pytest.approx(float(gp.double().pow(2).sum()), rel=1e-5)


def test_row_weights_equal_the_a1_formula(setup, tmp_path):
    """weight_P = B_m / B_mini and weight_U = (B_m / B_mini) x T_i / T_m, with T = the tokens of
    response_mask x self_distillation_mask (core_algos.py 1207-1209), on synthetic rows."""
    actor, data = setup
    sd = {0: 1.0, 1: 0.0, 2: 1.0}
    rows_data = {t: {**data[t], "self_distillation_mask": torch.tensor([sd[t]])} for t in sd}
    ctx = ToyCtx(actor, rows_data)
    run(ctx, tmp_path, split_rows("A1-S0-1", [3]))
    tokens = {t: float(data[t]["response_mask"].sum()) * sd[t] for t in sd}
    t_m = sum(tokens.values())
    for rec in records(tmp_path, "A1-S0-1"):
        assert rec["weight_P"] == 3 / B_MINI
        assert rec["loss_tokens"] == tokens[rec["traj"]]
        assert rec["weight_U"] == pytest.approx(3 / B_MINI * tokens[rec["traj"]] / t_m, rel=1e-12)
    assert records(tmp_path, "A1-S0-1")[1]["weight_U"] == 0.0


def test_resume_gives_the_same_outputs(setup, tmp_path):
    """Item 7.5: a forced stop after one micro-batch and a restart give the files of a run without a stop."""
    actor, data = setup
    ctx = ToyCtx(actor, data)
    rows = split_rows("A1-S0-1", [3, 2, 1])
    run(ctx, tmp_path / "a", rows)
    with pytest.raises(pack.Stop):
        run(ctx, tmp_path / "b", rows, stop_after=1)
    prog = json.loads((tmp_path / "b" / "progress_A1-S0-1.json").read_text())
    assert prog["committed"] == [[0, 0, 0]] and len(prog["rows"]) == 3
    assert (tmp_path / "b" / "vectors" / "partial_A1-S0-1_P.pt").exists()
    run(ctx, tmp_path / "b", rows)
    assert records(tmp_path / "a", "A1-S0-1") == records(tmp_path / "b", "A1-S0-1")
    for arm in "PU":
        assert torch.equal(torch.load(tmp_path / "a" / "vectors" / f"A1-S0-1_{arm}.pt"),
                           torch.load(tmp_path / "b" / "vectors" / f"A1-S0-1_{arm}.pt"))
    assert not (tmp_path / "b" / "vectors" / "partial_A1-S0-1_P.pt").exists()


def test_pilot_subset_skips_a_batch_without_selected_micro_batches(setup, tmp_path):
    actor, data = setup
    ctx = ToyCtx(actor, data)
    rows = split_rows("A1-S0-3", [3, 2])
    res = run(ctx, tmp_path, rows, pilot=[["A1-S0-1", 0, 0, 0]])
    assert res["skipped"] and not (tmp_path / "PACKROWS_A1-S0-3.jsonl").exists()
    assert not list((tmp_path / "vectors").glob("*.pt"))
    res = run(ctx, tmp_path, rows, pilot=[["A1-S0-3", 0, 0, 1]])
    assert [r["micro"] for r in records(tmp_path, "A1-S0-3")] == [1, 1]


def test_gram_diagonal_equals_separate_norms(setup, tmp_path):
    """G5 form: the Gram diagonal equals the squared norms computed at the batch end; the vector files are removed."""
    actor, data = setup
    ctx = ToyCtx(actor, data)
    run(ctx, tmp_path, split_rows("A1-S0-1", [3, 1]))
    run(ctx, tmp_path, split_rows("A1-S0-2", [2], start=4))
    pack.gram(tmp_path)
    g = json.loads((tmp_path / "GRAM_PACK.json").read_text())
    for i, n in enumerate(g["names"]):
        assert g["gram"][i][i] == pytest.approx(g["sq_norm_separate"][n], rel=1e-6)
    assert np.allclose(np.asarray(g["gram"]), np.asarray(g["gram"]).T)
    assert not list((tmp_path / "vectors").glob("*.pt"))
    assert (tmp_path / "GRAM_PACK.sha256").read_text().split()[0] == pack.sha256_file(tmp_path / "GRAM_PACK.json")
    res = pstats.compute(tmp_path, 50)  # end to end: pack.py outputs -> stats.py, 2 batches as in the pilot
    pg = res["gradients"]["per_group"]["A1-S0"]
    assert pg["pairs"] == 1 and pg["n_grp"]["value"] > 0 and math.isfinite(pg["c_e"]["value"])
    assert res["groups"]["A1-S0"]["packing"]["micro_batches"] == 3


def test_split_matches_the_a1_partition_functions():
    """Item 7.2: balance_order equals RayPPOTrainer._balance_batch and micro_partition equals rearrange_micro_batches."""
    from tensordict import TensorDict
    from types import SimpleNamespace

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.seqlen_balancing import rearrange_micro_batches

    rng = np.random.default_rng(3)
    seq = [int(x) for x in rng.integers(2284, 7775, size=48)]
    width = max(seq)
    mask = torch.zeros(len(seq), width, dtype=torch.long)
    for i, s in enumerate(seq):
        mask[i, width - s:] = 1
    batch = DataProto.from_dict(tensors={"attention_mask": mask, "idx": torch.arange(len(seq))})
    stub = SimpleNamespace(_get_dp_size=lambda wg, role: RANKS, actor_rollout_wg=None, use_prefix_grouper=False,
                           config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={})))
    RayPPOTrainer._balance_batch(stub, batch, metrics={})
    ours = split.balance_order(seq, RANKS)
    assert batch.batch["idx"].tolist() == [i for part in ours for i in part]

    for part in ours:
        s = [seq[i] for i in part]
        td = TensorDict({"attention_mask": mask[part], "input_ids": mask[part]}, batch_size=[len(part)])
        _, verl_parts = rearrange_micro_batches(td, max_token_len=9216)
        assert split.micro_partition(s, split.micro_count(s, 9216)) == verl_parts


def test_split_batch_uses_the_largest_micro_count_of_the_ranks():
    # 96 rows, 16 per rank: the rank with the 9010-token row needs 2 micro-batches, the other ranks need 1.
    rows = [{"prompt_len": p, "response_ids": [0] * 10} for p in [9000] + [100] * 95]
    recs = split.split_batch(rows, "A2")
    counts = {}
    for r in recs:
        counts.setdefault(r["rank"], set()).add(r["micro"])
    per_rank = [math.ceil(sum(r["total_tokens"] for r in recs if r["rank"] == k) / 9216) for k in range(RANKS)]
    assert min(per_rank) < max(per_rank)  # the ranks differ, so the rule matters
    assert {len(v) for v in counts.values()} == {max(per_rank)}


def test_statistics_by_hand(tmp_path):
    """Item 7.4: stats.py on synthetic rows and a synthetic Gram matrix."""
    rows = []
    for b, scale in (("A1-S0-1", 1.0), ("A1-S0-2", 2.0), ("A1-S0-3", 3.0)):
        for u in range(2):
            for pos in (1, 2):
                rows.append({"batch": b, "rank": 0, "mini": 0, "micro": u, "position": pos, "response_tokens": 2 * pos,
                             "traj": len(rows), "d_abs": [0.0, 0.0] if pos == 1 else [scale, 2 * scale],
                             "kl_student_sum": 0.0 if pos == 1 else 3 * scale, "kl_teacher_sum": 0.0 if pos == 1 else scale,
                             "kl_tokens": 2, "L_P": -10.0 if pos == 1 else -10.0 - scale, "L_U": -10.0,
                             "rkl_U_sum": 0.2, "rkl_P_sum": 0.2 if pos == 1 else 0.2 + scale})
    for b in ("A1-S0-1", "A1-S0-2", "A1-S0-3"):
        with open(tmp_path / f"PACKROWS_{b}.jsonl", "w") as f:
            for r in rows:
                if r["batch"] == b:
                    f.write(json.dumps(r) + "\n")
    vec = np.random.default_rng(0).normal(size=(6, 5))
    names = [f"A1-S0-{k}_{a}" for k in (1, 2, 3) for a in "PU"]
    gram = vec @ vec.T
    (tmp_path / "GRAM_PACK.json").write_text(json.dumps(
        {"names": names, "gram": gram.tolist(), "sq_norm_separate": {n: float(gram[i, i]) for i, n in enumerate(names)}}))
    res = pstats.compute(tmp_path, 200)
    grp = res["groups"]["A1-S0"]
    assert grp["packing"]["micro_batches"] == 6 and grp["packing"]["rows_per_micro_batch"] == {"2": 6}
    assert grp["packing"]["share_rows_position_2plus"]["value"] == pytest.approx(0.5)
    assert grp["packing"]["share_response_tokens_position_2plus"]["value"] == pytest.approx(4 / 6)
    te = grp["token_errors"]
    assert te["1"]["d_abs_mean"]["value"] == 0.0 and te["1"]["kl_student_mean"]["value"] == 0.0
    pooled = [x for s in (1, 2, 3) for x in [s, 2 * s] * 2]
    assert te["2+"]["d_abs_mean"]["value"] == pytest.approx(np.mean(pooled))
    assert te["2+"]["d_abs_p99"]["value"] == pytest.approx(np.percentile(pooled, 99))
    assert te["2+"]["kl_student_mean"]["value"] == pytest.approx(3 * 2 * 2 * 6 / 2 / 12)  # sum 3s x 2 micro / tokens
    assert te["2+"]["L_rel_diff_mean"]["value"] == pytest.approx(np.mean([s / 10 for s in (1, 2, 3)]))
    rk = grp["rollout_kl"]  # R3 item 6.7: sum of rkl over sum of tokens
    assert rk["1"]["U"]["value"] == pytest.approx(0.1) and rk["1"]["P"]["value"] == pytest.approx(0.1)
    assert rk["2+"]["U"]["value"] == pytest.approx(0.1)
    assert rk["2+"]["P"]["value"] == pytest.approx((6 * 0.2 + 2 * (1 + 2 + 3)) / 12)
    assert rk["all"]["P"]["value"] == pytest.approx((12 * 0.2 + 2 * (1 + 2 + 3)) / 24)
    assert rk["all"]["P"]["count"] == 24 and rk["all"]["P"]["micro_batches"] == 6
    gr = res["gradients"]
    v = {n: vec[i] for i, n in enumerate(names)}
    for k in (1, 2, 3):
        gp, gu = v[f"A1-S0-{k}_P"], v[f"A1-S0-{k}_U"]
        pb = gr["per_batch"][f"A1-S0-{k}"]
        assert pb["eps"] == pytest.approx(np.linalg.norm(gp - gu) / np.linalg.norm(gu))
        assert pb["cos"] == pytest.approx(gp @ gu / np.linalg.norm(gp) / np.linalg.norm(gu))
        assert pb["norm_ratio"] == pytest.approx(np.linalg.norm(gp) / np.linalg.norm(gu))
    pairs = [(1, 2), (1, 3), (2, 3)]
    n_grp = np.mean([np.linalg.norm(v[f"A1-S0-{a}_U"] - v[f"A1-S0-{b}_U"]) for a, b in pairs])
    e = {k: v[f"A1-S0-{k}_P"] - v[f"A1-S0-{k}_U"] for k in (1, 2, 3)}
    c_e = np.mean([e[a] @ e[b] / np.linalg.norm(e[a]) / np.linalg.norm(e[b]) for a, b in pairs])
    pg = gr["per_group"]["A1-S0"]
    assert pg["n_grp"]["value"] == pytest.approx(n_grp) and pg["c_e"]["value"] == pytest.approx(c_e)
    assert gr["per_batch"]["A1-S0-1"]["rho"] == pytest.approx(np.linalg.norm(e[1]) / n_grp)


def test_bootstrap_interval_is_a_point_for_equal_micro_batches():
    r = pstats.ratio([2.0] * 5, [4.0] * 5, 200)
    assert r["value"] == 0.5 and r["ci"] == [0.5, 0.5]
    p = pstats.percentile([np.asarray([1.0, 3.0])] * 4, 50.0, 200)
    assert p["value"] == 2.0 and p["ci"] == [2.0, 2.0]


def test_pilot_select_takes_the_most_rows_and_the_longest_row():
    micro = {("A1-S0-1", 0, 0, 0): [{"total_tokens": 100}] * 3, ("A1-S0-1", 0, 0, 1): [{"total_tokens": 900}],
             ("A1-S0-2", 0, 0, 0): [{"total_tokens": 200}] * 2}
    assert pilot.select(micro) == [["A1-S0-1", 0, 0, 0], ["A1-S0-1", 0, 0, 1]]
