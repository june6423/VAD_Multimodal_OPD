"""Tests of batches.py (HANDOFF-VLM-VAD-GRAD-003-R1 Part A item 8.2, DIRECTIVE-VLM-VAD-GRAD-003-001 action 4)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "scripts" / "vad_grad3"
sys.path.insert(0, str(ROOT))


def load(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


batches = load("vad_grad3_batches", DIR / "batches.py")


def a1_balance(seq: list[int]) -> list[int]:
    """The batch order after RayPPOTrainer._balance_batch with 6 ranks."""
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    width = max(seq)
    mask = torch.zeros(len(seq), width, dtype=torch.long)
    for i, s in enumerate(seq):
        mask[i, width - s:] = 1
    batch = DataProto.from_dict(tensors={"attention_mask": mask, "idx": torch.arange(len(seq))})
    stub = SimpleNamespace(_get_dp_size=lambda wg, role: 6, actor_rollout_wg=None, use_prefix_grouper=False,
                           config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={})))
    RayPPOTrainer._balance_batch(stub, batch, metrics={})
    return batch.batch["idx"].tolist()


def seqlens(n: int, seed: int) -> list[int]:
    return [int(x) for x in np.random.default_rng(seed).integers(2284, 7775, size=n)]


def test_divisible_batch_gives_the_a1_split():
    """Action 4: a batch divisible by 6 gets no dummy and the split of the A1 code."""
    from tensordict import TensorDict

    from verl.utils.seqlen_balancing import rearrange_micro_batches

    seq = seqlens(48, 3)
    ours, dummies = batches.balance_order(seq)
    assert dummies == 0
    assert a1_balance(seq) == [i for part in ours for i in part]
    width = max(seq)
    for part in ours:
        s = [seq[i] for i in part]
        mask = torch.zeros(len(part), width, dtype=torch.long)
        for k, v in enumerate(s):
            mask[k, width - v:] = 1
        td = TensorDict({"attention_mask": mask, "input_ids": mask}, batch_size=[len(part)])
        _, verl_parts = rearrange_micro_batches(td, max_token_len=9216)
        assert batches.micro_partition(s, batches.micro_count(s)) == verl_parts


@pytest.mark.parametrize("n", [752, 94, 93, 50])
def test_dummies_fill_to_a_multiple_of_6_and_are_removed(n):
    seq = seqlens(n, n)
    ours, dummies = batches.balance_order(seq)
    assert dummies == (-n) % 6
    flat = [i for part in ours for i in part]
    assert sorted(flat) == list(range(n))  # each real row once, no dummy
    full = -(-n // 6)
    assert all(full - 4 <= len(p) <= full for p in ours) and sum(full - len(p) for p in ours) == dummies


def test_dummies_equal_the_a1_split_of_the_padded_batch():
    """The rank parts are those of the A1 code on the batch with the zero-token rows, without those rows."""
    seq = seqlens(94, 7)
    ours, dummies = batches.balance_order(seq)
    padded = a1_balance(seq + [0] * dummies)
    full = len(padded) // 6
    a1_parts = [[i for i in padded[r * full:(r + 1) * full] if i < len(seq)] for r in range(6)]
    assert ours == a1_parts


def rows_of(n_prompts: int, seed: int, siblings: int = 8) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, t = [], 0
    for p in range(n_prompts):
        plen = int(rng.integers(1500, 3000))
        for s in range(siblings):
            out.append({"traj": t, "prompt_pos": p, "sibling": s, "dataset_index": p, "prompt_len": plen,
                        "response_ids": [1] * int(rng.integers(1, 300))})
            t += 1
    return out


def test_dummies_carry_no_loss_and_weights_follow_the_a1_formula():
    """Action 4: the row files hold only real rows; the weights sum to rows / (6 mini_cfg)."""
    rows = rows_of(94, 1)
    recs, info = batches.split_batch(rows, 8)
    assert info["dummies"] == 4 and len(recs) == 752
    assert sorted(r["input_index"] for r in recs) == list(range(752))
    assert info["weight_sum"] == pytest.approx(752 / (6 * 128), rel=1e-12)
    one = [r for r in rows if r["sibling"] == 0]
    recs1, info1 = batches.split_batch(one, 1)
    assert info1["dummies"] == 2 and info1["weight_sum"] == pytest.approx(94 / (6 * 16), rel=1e-12)
    for r in recs:
        assert r["weight"] == pytest.approx(r["micro_rows"] / 128 / 6 * r["loss_tokens"] / r["micro_loss_tokens"])


def test_micro_count_is_the_largest_of_the_ranks():
    rows = [{"prompt_len": p, "response_ids": [0] * 10} for p in [9000] + [100] * 95]
    recs, info = batches.split_batch(rows, 1)
    per_rank = [math.ceil(sum(r["total_tokens"] for r in recs if r["rank"] == k) / 9216) for k in range(6)]
    assert min(per_rank) < max(per_rank)
    assert info["micro_batches"] == max(per_rank)
    assert {len({r["micro"] for r in recs if r["rank"] == k}) for k in range(6)} == {max(per_rank)}


def test_build_makes_each_batch_type():
    rollouts = rows_of(28, 2)
    for r in rollouts:
        r["prompt_pos"] += 100
    labels = {r["traj"]: r["traj"] % 2 for r in rollouts}
    out, summary = batches.build(rollouts, labels)
    folds = batches.fold_of(sorted({r["prompt_pos"] for r in rollouts}))
    assert set(out) == ({f"B8-f{f}" for f in range(1, 5)} | {f"B1-f{f}-j{j}" for f in range(1, 5) for j in range(8)}
                        | {f"R8-f1-s{s}" for s in (1, 2, 3)})
    for f in range(1, 5):
        prompts = {p for p, k in folds.items() if k == f - 1}
        assert {r["prompt"] for r in out[f"B8-f{f}"]} == prompts and len(out[f"B8-f{f}"]) == 8 * len(prompts)
        assert {r["sibling"] for r in out[f"B1-f{f}-j3"]} == {3}
    assert all("rows_new_neighbors" in summary["batches"][f"R8-f1-s{s}"] for s in (1, 2, 3))
    assert {r["len_bin"] for b in out.values() for r in b} <= {"L1", "L2", "L3"}


def test_random_partition_keeps_the_a1_counts_and_the_token_limit():
    """R2 Definition 9.3: same rows, A1 rows per rank, A1 micro-batch count, at most 9216 tokens per micro-batch."""
    rows = rows_of(94, 5)
    base, _, _ = batches.a1_micro(rows)
    sizes = [sum(len(m) for m in rk) for rk in base]
    parts = [batches.random_micro(base, rows, s, sweeps=50) for s in (1, 2, 3)]
    for part in parts:
        assert [sum(len(m) for m in rk) for rk in part] == sizes
        assert all(len(rk) == len(base[0]) and all(rk) for rk in part)
        assert sorted(i for rk in part for m in rk for i in m) == list(range(len(rows)))
        limit = max(9216, max(sum(batches.seqlen(rows[i]) for i in m) for rk in base for m in rk))
        assert max(sum(batches.seqlen(rows[i]) for i in m) for rk in part for m in rk) <= limit
        assert [[len(m) for m in rk] for rk in part] == [[len(m) for m in rk] for rk in base]
    assert parts[0] != parts[1] and parts[0] == batches.random_micro(base, rows, 1, sweeps=50)
    recs, info = batches.split_batch(rows, 8, parts[0])
    assert info["weight_sum"] == pytest.approx(752 / (6 * 128), rel=1e-12)
    # most rows get new micro-batch neighbors
    a1 = batches.neighbors([{"traj": i, "rank": r, "micro": u} for r, rk in enumerate(base) for u, m in enumerate(rk) for i in m])
    rnd = batches.neighbors([{"traj": i, "rank": r, "micro": u} for r, rk in enumerate(parts[0]) for u, m in enumerate(rk) for i in m])
    assert sum(a1[i] != rnd[i] for i in a1) > 0.9 * len(rows)
