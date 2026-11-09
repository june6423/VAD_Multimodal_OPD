"""Tests of grads.py and update_path.py (HANDOFF-VLM-VAD-GRAD-003-R2 Part A items 3 to 5 and 8, gate G1).

The toy forward replaces DataParallelPPOActor._forward_micro_batch with a linear model over a small vocabulary (the
semantics of the PACK-001 test_pack.py ToyActor). When one forward holds more than one row, row j gets a state term
from rows 0 to j-1, so the gradient depends on the micro-batch membership and the row order. The O4 loss is the real
compute_self_distillation_loss with the A1 actor config (sources/A1_config.yaml, rank mini-batch 128 as in
fsdp_workers.py 244-245). The attention masks have different lengths, so the A1 split and prepare_dynamic_batch make
more than one micro-batch on each rank.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "scripts" / "vad_grad3"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIR))


def load(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


batches = load("vad_grad3_batches", DIR / "batches.py")
grads = load("vad_grad3_grads", DIR / "grads.py")
update_path = load("vad_grad3_update_path", DIR / "update_path.py")
obj = load("vad_grad3_objectives", DIR / "objectives.py")
g2 = load("vad_grad3_g2", DIR / "g2.py")

V, L, H = 16, 6, 8
W = 8000  # attention-mask width (below the A1 max_token_len 9216)
N_ROWS = 34  # 6 ranks: 2 dummy rows (DIRECTIVE-VLM-VAD-GRAD-003-001)


def toy_forward(self, inputs, temperature, calculate_entropy=False, return_all_logps=False, distill_topk=None,
                topk_indices=None, return_response_logits=False, module=None):
    model = module or self.actor_module
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


def a1_actor_config():
    from omegaconf import OmegaConf, open_dict

    from verl.utils.config import omega_conf_to_dataclass

    cfg = OmegaConf.load(DIR / "sources" / "A1_config.yaml")
    with open_dict(cfg):
        cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 128  # 96 x n 8 / 6 ranks
    OmegaConf.resolve(cfg)
    return omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)


def make_actor(config):
    """A DataParallelPPOActor without __init__ (no process group), with the toy forward."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    torch.manual_seed(0)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = config
    actor.actor_module = torch.nn.Linear(H, V, bias=False)
    actor.teacher_module = torch.nn.Linear(H, V, bias=False).requires_grad_(False)
    actor.actor_optimizer = None
    actor.use_prefix_grouper = False
    actor.ulysses_sequence_parallel_size = 1
    actor.use_dynamic_bsz = True
    actor.use_fused_kernels = False
    actor.use_remove_padding = True
    actor.scaler = None
    actor._forward_micro_batch = types.MethodType(toy_forward, actor)
    return actor


def toy_rows(seed: int = 1, n: int = N_ROWS):
    """Input rows (prompt_len, response_ids as batches.py reads them) and the model inputs of each row."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows, data = [], {}
    for t in range(n):
        n_tok = int(rng.integers(1, L + 1))
        s = int(rng.integers(1500, W + 1))
        r = {"responses": torch.randint(0, V, (1, L), generator=gen),
             "response_mask": (torch.arange(L) < n_tok).float().unsqueeze(0),
             "old_log_probs": torch.zeros(1, L), "self_distillation_mask": torch.ones(1)}
        for prefix in ("", "teacher_", "teacher_minus_"):
            r[prefix + "input_ids"] = torch.randn(1, L * H, generator=gen)
            r[prefix + "attention_mask"] = torch.ones(1, L)
            r[prefix + "position_ids"] = torch.zeros(1, L, dtype=torch.long)
            r[prefix + "response_start_idx"] = torch.zeros(1, dtype=torch.long)
            r[prefix + "multi_modal_inputs"] = None
        r["attention_mask"] = (torch.arange(W) >= W - s).long().unsqueeze(0)  # s tokens: the A1 seqlen
        data[t] = r
        rows.append({"traj": t, "prompt_len": s - n_tok, "response_ids": [0] * n_tok})
    return rows, data


def len_bin(n_tok: int) -> str:
    """Test bins (the real bins need more than 64 response tokens; the toy has at most 6)."""
    return "L1" if n_tok <= 2 else ("L2" if n_tok <= 4 else "L3")


def row_file(rows: list[dict], key: str, partition=None) -> list[dict]:
    recs, _ = batches.split_batch(rows, 8, partition)
    for rec in recs:
        i = rec.pop("input_index")
        rec.update(batch=key, n=8, traj=rows[i]["traj"], label=i % 2, len_bin=len_bin(len(rows[i]["response_ids"])))
    return recs


class ToyCtx:
    def __init__(self, actor, data):
        self.actor = actor
        self.params = list(actor.actor_module.parameters())
        self.device = "cpu"
        self.classes = grads.make_classes(obj)
        self.data = data

    def inputs(self, ros):
        rows = [self.data[ro["traj"]] for ro in ros]
        mi = {k: (torch.cat([r[k] for r in rows]) if torch.is_tensor(rows[0][k]) else None) for k in rows[0]}
        return mi, [dict(r) for r in rows]

    def batch(self, ros):
        from verl import DataProto

        rows = [self.data[ro["traj"]] for ro in ros]
        return DataProto.from_dict(tensors={k: torch.cat([r[k] for r in rows]) for k in rows[0] if torch.is_tensor(rows[0][k])})


def run(ctx, out: Path, key: str, recs: list[dict], **opts):
    out.mkdir(parents=True, exist_ok=True)
    o = {"stop_after": 0, "commit_seconds": 1e9, **opts}
    rollouts = {r["traj"]: {"traj": r["traj"]} for r in recs}
    return grads.run_batch(ctx, "O", key, recs, rollouts, out, o, {"micro": 0})


def vec(out: Path, key: str, part: str) -> torch.Tensor:
    return torch.load(out / "vectors" / f"O__{key}__{part}.pt")


def rank_data(ctx, recs: list[dict]):
    """g2.rank_batches: the DataProto of each rank in the order of its balance_batch part (rank_place)."""
    rollouts = {r["traj"]: {"traj": r["traj"]} for r in recs}
    return g2.rank_batches(ctx, recs, rollouts, {"temperature": 1.0, "pad_token_id": 0})


@pytest.fixture(scope="module")
def setup():
    return a1_actor_config()


@pytest.fixture
def cpu_device(monkeypatch):
    import verl.workers.actor.dp_actor as dpa

    monkeypatch.setattr(dpa, "get_device_id", lambda: "cpu")


def test_g1_analysis_gradient_equals_the_update_path(setup, cpu_device, tmp_path):
    """Gate G1: the total vector of grads.run_batch equals update_policy for each rank (rows in rank_place order),
    mean over 6 ranks; the update_policy micro-batches equal the micro-batches of the row file."""
    rows, data = toy_rows()
    recs = row_file(rows, "B8-f1")
    count = max(x["micro"] for x in recs) + 1
    assert count > 1 and max(x["micro_rows"] for x in recs) > 1  # the split is not trivial
    assert {x["rank_rows"] for x in recs} == {5, 6}  # 2 dummy rows
    actor = make_actor(setup)
    ctx = ToyCtx(actor, data)
    run(ctx, tmp_path, "B8-f1", recs)
    params = list(actor.actor_module.parameters())
    acc, parts = update_path.update_path_gradient(actor, params, rank_data(ctx, recs), count)
    assert parts == g2.expected_partitions(recs, count)
    assert sorted(i for p in parts for m in p for i in m) != list(range(N_ROWS))  # rank-local indices
    up = torch.cat([a.reshape(-1) for a in acc])
    cmp = update_path.compare(vec(tmp_path, "B8-f1", "total"), up)
    assert cmp["rel_diff"] < 1e-5 and cmp["cos"] > 1 - 1e-6
    assert all(p.grad is None for p in params)


def test_g1_detects_a_different_row_order(setup, cpu_device, tmp_path):
    """The comparison is sensitive: rows in input order instead of rank_place order give other micro-batches."""
    rows, data = toy_rows()
    recs = row_file(rows, "B8-f1")
    count = max(x["micro"] for x in recs) + 1
    actor = make_actor(setup)
    ctx = ToyCtx(actor, data)
    run(ctx, tmp_path, "B8-f1", recs)
    params = list(actor.actor_module.parameters())
    shuffled = [{**x, "rank_place": -x["rank_place"]} for x in recs]
    acc, parts = update_path.update_path_gradient(actor, params, rank_data(ctx, shuffled), count)
    assert parts != g2.expected_partitions(recs, count)
    cmp = update_path.compare(vec(tmp_path, "B8-f1", "total"), torch.cat([a.reshape(-1) for a in acc]))
    assert cmp["rel_diff"] > 1e-3


def test_parts_sum_to_the_total_and_equal_masked_losses(setup, tmp_path):
    """C + I = total and L1 + L2 + L3 = total. Part X equals the gradient of the packed micro-batch loss with the
    response mask of the other rows set to 0, times T_X / T_m (token-mean, core_algos.py 1057-1060), times
    B_m / 128 / 6, summed over the micro-batches."""
    rows, data = toy_rows()
    recs = row_file(rows, "B8-f1")
    actor = make_actor(setup)
    res = run(ToyCtx(actor, data), tmp_path, "B8-f1", recs)
    assert res["part_sum"]["C+I"] < 1e-5 and res["part_sum"]["L1+L2+L3"] < 1e-5
    total = vec(tmp_path, "B8-f1", "total")
    assert torch.allclose(vec(tmp_path, "B8-f1", "C") + vec(tmp_path, "B8-f1", "I"), total, rtol=1e-5, atol=1e-8)
    ob = obj.Objectives(actor, list(actor.actor_module.parameters()))
    params = list(actor.actor_module.parameters())
    ctx = ToyCtx(actor, data)
    micro = {}
    for x in recs:
        micro.setdefault((x["rank"], x["micro"]), []).append(x)
    for part in ("C", "I", "L1", "L2", "L3"):
        want = torch.zeros_like(total)
        for mrows in micro.values():
            mrows = sorted(mrows, key=lambda x: x["position"])
            keep = [i for i in grads.members(part, mrows)]
            if not keep:
                continue
            mi, _ = ctx.inputs(mrows)
            t_m = float(mi["response_mask"].sum())
            mask = mi["response_mask"].clone()
            for i in range(len(mrows)):
                if i not in keep:
                    mask[i] = 0
            loss = ob.losses({**mi, "response_mask": mask}, ["O4"])["O4"] * (float(mask.sum()) / t_m)
            g = torch.autograd.grad(loss * len(mrows) / 128 / 6, params)
            want += torch.cat([x.reshape(-1) for x in g])
        assert torch.allclose(vec(tmp_path, "B8-f1", part), want, rtol=1e-5, atol=1e-8), part
    lines = [json.loads(x) for x in open(tmp_path / "GRADROWS_O__B8-f1.jsonl")]
    assert len(lines) == N_ROWS and all(x["l_row"] is not None for x in lines)


def test_b1_and_other_keys_have_their_parts(setup, tmp_path):
    assert grads.parts_of("O", "B8-f2") == ["total", "C", "I", "L1", "L2", "L3"]
    assert grads.parts_of("O", "B1-f1-j3") == ["total", "C", "I"]
    for mode, key in (("O", "R8-f1-s1"), ("Fx", "B8-f1"), ("O", "B8-f1@rep"), ("Fx", "B1-f1-j0")):
        assert grads.parts_of(mode, key) == ["total"]


def test_resume_gives_the_same_outputs(setup, tmp_path):
    """A forced stop after one micro-batch and a restart give the files of a run without a stop."""
    rows, data = toy_rows()
    recs = row_file(rows, "B8-f1")
    actor = make_actor(setup)
    ctx = ToyCtx(actor, data)
    run(ctx, tmp_path / "a", "B8-f1", recs)
    with pytest.raises(grads.Stop):
        run(ctx, tmp_path / "b", "B8-f1", recs, stop_after=1)
    prog = json.loads((tmp_path / "b" / "progress_O__B8-f1.json").read_text())
    assert len(prog["committed"]) == 1
    assert (tmp_path / "b" / "vectors" / "partial_O__B8-f1__C.pt").exists()
    run(ctx, tmp_path / "b", "B8-f1", recs)
    ra = [json.loads(x) for x in open(tmp_path / "a" / "GRADROWS_O__B8-f1.jsonl")]
    rb = [json.loads(x) for x in open(tmp_path / "b" / "GRADROWS_O__B8-f1.jsonl")]
    assert ra == rb
    for part in grads.parts_of("O", "B8-f1"):
        assert torch.equal(vec(tmp_path / "a", "B8-f1", part), vec(tmp_path / "b", "B8-f1", part))
    assert not list((tmp_path / "b" / "vectors").glob("partial_*"))
    assert run(ctx, tmp_path / "b", "B8-f1", recs)["skipped"] == "GRADROWS exists"


def test_a_loss_token_mismatch_stops(setup, tmp_path):
    rows, data = toy_rows()
    recs = row_file(rows, "B8-f1")
    recs[0] = {**recs[0], "loss_tokens": recs[0]["loss_tokens"] + 1}
    with pytest.raises(SystemExit):
        run(ToyCtx(make_actor(setup), data), tmp_path, "B8-f1", recs)
    assert "loss tokens differ" in (tmp_path / "STOP.txt").read_text()


def test_random_partition_rows_give_their_own_gradient(setup, cpu_device, tmp_path):
    """R8 rows (a given partition) run with the total part only; the result differs from the A1 split."""
    rows, data = toy_rows()
    a1, _, _ = batches.a1_micro(rows)
    recs_r = row_file(rows, "R8-f1-s1", batches.random_micro(a1, rows, 1))
    actor = make_actor(setup)
    ctx = ToyCtx(actor, data)
    res = run(ctx, tmp_path, "R8-f1-s1", recs_r)
    assert res["part_sum"] is None and list((tmp_path / "vectors").glob("O__R8-f1-s1__*.pt")) == [
        tmp_path / "vectors" / "O__R8-f1-s1__total.pt"]
    run(ctx, tmp_path, "B8-f1", row_file(rows, "B8-f1"))
    cmp = update_path.compare(vec(tmp_path, "R8-f1-s1", "total"), vec(tmp_path, "B8-f1", "total"))
    assert cmp["rel_diff"] > 1e-4


# ---------- displacement and gram ----------

NAMES = [f"model.layers.{i // 4}.p{i % 4}.weight" for i in range(248)]
SHAPE = (6, 2)


def write_g3(path: Path):
    path.write_text("\n".join(f"{n} ({SHAPE[0]}, {SHAPE[1]}) {SHAPE[0] * SHAPE[1]}" for n in NAMES) + "\n")


class FakeDTensor:
    """The two DTensor members that ckpt.model_tensor uses: the global shape and the local Shard(0) part."""

    def __init__(self, full: torch.Tensor, r: int):
        self.shape = full.shape
        self.local = full[r:r + 1].clone()

    def to_local(self):
        return self.local


def test_displacement_vectors(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import ckpt

    g = torch.Generator().manual_seed(2)
    theta = {k: {n: torch.randn(SHAPE, generator=g) for n in NAMES} for k in ("base", "a1_80", "a1_130", "a2_80", "a2_130")}
    (tmp_path / "base").mkdir()
    save_file({n: t.to(torch.bfloat16) for n, t in theta["base"].items()}, str(tmp_path / "base" / "model.safetensors"))
    shards = {str(tmp_path / k): [{n: FakeDTensor(t, r) for n, t in theta[k].items()} for r in range(6)]
              for k in ("a1_80", "a1_130", "a2_80", "a2_130")}
    monkeypatch.setattr(ckpt, "open_shards", lambda d, kind: shards[d])
    write_g3(tmp_path / "G3_PARAMS.txt")
    out = tmp_path / "disp"
    grads.displacement([str(tmp_path / "base"), str(tmp_path / "G3_PARAMS.txt"), str(out)] +
                       [str(tmp_path / k) for k in ("a1_80", "a1_130", "a2_80", "a2_130")])
    base = {n: t.to(torch.bfloat16).float() for n, t in theta["base"].items()}
    flat = lambda d: torch.cat([d[n].reshape(-1) for n in NAMES])  # noqa: E731
    want = {"D__A1_0_80": flat(theta["a1_80"]) - flat(base), "D__A1_80_130": flat(theta["a1_130"]) - flat(theta["a1_80"]),
            "D__A2_0_80": flat(theta["a2_80"]) - flat(base), "D__A2_80_130": flat(theta["a2_130"]) - flat(theta["a2_80"])}
    info = json.loads((out / "DISPLACEMENT_VECTORS.json").read_text())
    for name, w in want.items():
        assert torch.equal(torch.load(out / f"{name}.pt"), w)
        assert info[name]["elements"] == 248 * 12


@pytest.mark.parametrize("block", [None, "1000"])
def test_gram_matrices(tmp_path, monkeypatch, block):
    from safetensors.torch import save_file

    if block:
        monkeypatch.setenv("BLOCK", block)
    n = 248 * 12
    g = torch.Generator().manual_seed(3)
    write_g3(tmp_path / "G3_PARAMS.txt")
    out = tmp_path / "out"
    (out / "vectors").mkdir(parents=True)
    disp = tmp_path / "disp"
    disp.mkdir()
    names = ["O__B8-f1__C", "O__B8-f1__total", "Fx__B8-f1__total"]
    vecs = {k: torch.randn(n, generator=g) for k in names + list(grads.DISPLACEMENTS)}
    for k in names:
        torch.save(vecs[k], out / "vectors" / f"{k}.pt")
    view = torch.randn(n + 5, generator=g)[5:]  # a view with storage offset 5 (the reader must skip it)
    vecs["O__B8-f1__C"] = view.clone()
    torch.save(view, out / "vectors" / "O__B8-f1__C.pt")
    for k in grads.DISPLACEMENTS:
        torch.save(vecs[k], disp / f"{k}.pt")
    (out / "progress_O__B8-f1.json").write_text(json.dumps(
        {"sq_norm": {p: float(vecs[f"O__B8-f1__{p}"].double().pow(2).sum()) for p in ("C", "total")}}))
    v = {nm: torch.rand(SHAPE, generator=g) * 1e-6 for nm in NAMES}
    save_file(v, str(tmp_path / "V.safetensors"), metadata={"step": "80"})
    grads.gram([str(out), str(disp), str(tmp_path / "V.safetensors"), str(tmp_path / "G3_PARAMS.txt"), "A1_80"])
    res = json.loads((out / "GRAM_A1_80.json").read_text())
    order = sorted(names) + list(grads.DISPLACEMENTS)
    assert res["names"] == order and res["elements"] == n and res["metric"]["step"] == 80
    x = torch.stack([vecs[k] for k in order]).double()
    corr = math.sqrt(1 - 0.999 ** 80)
    d = torch.cat([(1.0 / (v[nm].sqrt() / corr + 1e-8)).reshape(-1) for nm in NAMES]).double()
    for m, want in (("raw", x @ x.T), ("M1", (x * d) @ x.T), ("M2", (x * d) @ (x * d).T)):
        assert np.allclose(np.asarray(res[m]), want.numpy(), rtol=1e-6), m
    for i, k in enumerate(order):
        if k in res["sq_norm_separate"]:
            assert res["raw"][i][i] == pytest.approx(res["sq_norm_separate"][k], rel=1e-6)
    assert set(res["sq_norm_separate"]) == {"O__B8-f1__C", "O__B8-f1__total"}
    assert not list((out / "vectors").glob("*.pt"))
    assert all((disp / f"{k}.pt").exists() for k in grads.DISPLACEMENTS)
    removed = json.loads((out / "VECTORS_REMOVED.json").read_text())
    assert sorted(removed["removed"]) == sorted(f"{k}.pt" for k in names)
    assert (out / "GRAM_A1_80.sha256").read_text().split()[0] == grads.sha256_file(out / "GRAM_A1_80.json")


def test_diag_partsum_on_the_toy(setup):
    """DIRECTIVE-003 diagnostic: on the toy forward, the replay and the masked parts sum to the total."""
    diag = load("vad_grad3_diag_partsum", DIR / "diag_partsum.py")
    rows, data = toy_rows()
    recs = row_file(rows, "B8-f1")
    diag.MIN_ROWS = 2  # the toy micro-batches hold at most 2 rows
    mrows = diag.pick_micro(recs)
    assert len(mrows) == 2 and {r["label"] for r in mrows} == {0, 1}
    ctx = ToyCtx(make_actor(setup), data)
    res = diag.diagnose(ctx, mrows, {r["traj"]: {"traj": r["traj"]} for r in recs})
    assert abs(res["replay_loss_sum"] - res["loss"]) < 1e-6 * abs(res["loss"])
    for key in ("C+I", "L1+L2+L3"):
        assert res["coverage"][key]["rows_in_no_class"] == 0 and res["coverage"][key]["rows_in_more_than_one"] == 0
        for method in ("replay", "masked"):
            assert res["partitions"][key][method]["loss_rel_diff"] < 1e-6, (key, method)
            assert res["partitions"][key][method]["grad_rel_diff"] < 1e-6, (key, method)
