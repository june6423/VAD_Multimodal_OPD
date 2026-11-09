"""Batches of Definition 6 and their A1 row split (HANDOFF-VLM-VAD-GRAD-003-R1 Part A item 2, DIRECTIVE-VLM-VAD-GRAD-003-001).

Folds: the 4 prompt folds of VLM-VAD-GRAD-001 (scripts/vad_grad/accum.py fold_of: random.Random(20261103) shuffles the
sorted prompts, fold index = place mod 4). Fold f = 1 to 4 is fold index f - 1.
Batches of one checkpoint (input order before balance_batch: the prompts in ascending order, the siblings in sibling
order, as the agent loop repeats each prompt rollout.n times):
  B8-f<f>: all 8 siblings of each prompt of fold f (n=8, mini_cfg 128).
  B1-f<f>-j<j>: sibling j of each prompt of fold f (n=1, mini_cfg 16), j = 0 to 7 in the TASK-053 sibling order.
  R8-f1-s<s>: the rows of B8-f1 in a random partition (HANDOFF-VLM-VAD-GRAD-003-R2 Definition 9.3), s = 1, 2, 3. It
    keeps the row count of each rank and of each micro-batch of the A1 split of B8-f1, and the token limit of a
    micro-batch: max(9216, the largest micro-batch of the A1 split), because the A1 partition balances the micro-batches
    but does not cap them at 9216 (seqlen_balancing.py 387-402). random_micro: a swap chain from the A1 split with
    numpy default_rng(s). The A1 split fills the micro-batches close to the limit, so a direct random assignment
    almost never meets it.
A1 row split (verl code of 2627743a, the functions are called, not copied):
  1. balance_batch: workload = calculate_workload(s), s = prompt tokens + response tokens (the attention-mask sum).
     DIRECTIVE-VLM-VAD-GRAD-003-001: zero-workload dummy rows are added up to a multiple of 6 ranks. Then the equal-size
     get_seqlen_balanced_partitions, the sort by (workload, index) and the even/odd reorder of ray_trainer.py 2483-2490
     run with the dummies. Then the dummies are removed.
  2. Rank r gets partition r. The rank mini-batch is the whole rank part (rows <= mini_cfg, dp_actor.py 1381).
  3. Micro-batch count = max over the ranks of min(rows, ceil(sum s / 9216)) (all_reduce MAX, seqlen_balancing.py 387-393).
  4. Micro-batch partition of each rank: rearrange_micro_batches order (seqlen_balancing.py 401-414).
Row weight (dp_actor.py 1444, 1829, core_algos.py 1057-1060, FSDP mean over the ranks): the batch loss is
  sum_i w_i l_i, with l_i the token-mean O4 loss of row i and w_i = (rows_mb / mini_cfg) / ranks * T_i / T_mb,
  T = loss-mask tokens = response tokens (self_distillation_mask 1 for each A1 row).

Usage: python batches.py <rollouts.jsonl> <LABELS.jsonl> <out dir> [<PILOT_PROMPTS.json>]
Writes ROWS_<batch>.jsonl (execution order: rank, micro-batch, position) and BATCHES.json.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

RANKS = 6
MAX_TOKEN_LEN = 9216
MINI_CFG = {8: 96 * 8 // RANKS, 1: 96 * 1 // RANKS}  # fsdp_workers.py 244-245: 128 and 16
SIBLINGS = 8
K = 4
FOLD_SEED = 20261103
PERM_SEEDS = (1, 2, 3)
SWEEPS = 200


def fold_of(prompts: list[int], k: int = K, seed: int = FOLD_SEED) -> dict[int, int]:
    """Fold index of each prompt (scripts/vad_grad/accum.py fold_of of exp/VLM-VAD-GRAD-001)."""
    order = sorted(prompts)
    random.Random(seed).shuffle(order)
    return {p: i % k for i, p in enumerate(order)}


def length_bin(response_tokens: int) -> str:
    return "L1" if response_tokens <= 5 else ("L2" if response_tokens <= 64 else "L3")


def workload(seqlens: list[int]) -> list[int]:
    import torch

    from verl.utils.seqlen_balancing import calculate_workload

    return calculate_workload(torch.tensor(seqlens, dtype=torch.long)).tolist()


def balanced(work: list[int], k: int, equal_size: bool) -> list[list[int]]:
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

    return get_seqlen_balanced_partitions(work, k_partitions=k, equal_size=equal_size)


def balance_order(seqlens: list[int], ranks: int = RANKS) -> tuple[list[list[int]], int]:
    """Rank parts of balance_batch (ray_trainer.py 2483-2490) with dummy rows (DIRECTIVE-001). Returns the input indices
    of each rank in order and the dummy count."""
    n = len(seqlens)
    dummies = (-n) % ranks
    work = workload(seqlens) + [0] * dummies
    out = []
    for p in balanced(work, ranks, equal_size=True):
        p = sorted(p, key=lambda x: (work[x], x))
        out.append([x for x in p[::2] + p[1::2][::-1] if x < n])
    return out, dummies


def micro_count(seqlens: list[int], max_token_len: int = MAX_TOKEN_LEN) -> int:
    """seqlen_balancing.py 387 for one rank, before the all_reduce MAX."""
    if max(seqlens) > max_token_len:
        raise SystemExit(f"STOP: a row has {max(seqlens)} tokens, above max_token_len {max_token_len}")
    return min(len(seqlens), math.ceil(sum(seqlens) / max_token_len))


def micro_partition(seqlens: list[int], count: int) -> list[list[int]]:
    """Micro-batches of one rank mini-batch (seqlen_balancing.py 401-414): indices into it, in execution order."""
    if count > len(seqlens):
        raise SystemExit(f"STOP: micro-batch count {count} above the {len(seqlens)} rows of a rank")
    work = workload(seqlens)
    parts = balanced(work, count, equal_size=False)
    parts.sort(key=lambda p: (sum(work[i] for i in p), p[0] if p else 0), reverse=True)
    return parts[::2][::-1] + parts[1::2]


def seqlen(row: dict) -> int:
    return row["prompt_len"] + len(row["response_ids"])


def a1_micro(rows: list[dict]) -> tuple[list[list[list[int]]], int, list[list[int]]]:
    """A1 split: for each rank, its micro-batches as lists of input indices in execution order; the dummy count; the
    input indices of each rank in the order of the rank part (the data that update_policy of the rank receives)."""
    seq = [seqlen(r) for r in rows]
    rank_rows, dummies = balance_order(seq)
    count = max(micro_count([seq[i] for i in rr]) for rr in rank_rows)  # all_reduce MAX over the ranks
    split = [[[rr[j] for j in part] for part in micro_partition([seq[i] for i in rr], count)] for rr in rank_rows]
    return split, dummies, rank_rows


def random_micro(start: list[list[list[int]]], rows: list[dict], seed: int, sweeps: int = SWEEPS) -> list[list[list[int]]]:
    """Random partition of R2 Definition 9.3: a random swap chain from the A1 split `start`.

    Each step draws two rows at random from the whole batch and swaps their micro-batches (any ranks) when both
    micro-batches stay within the token limit. A swap keeps the row count of each rank and of each micro-batch, so the
    rank row counts and the micro-batch count stay those of A1. sweeps x rows steps run. Then the rows of each
    micro-batch get a random order (their place in the packed sequence)."""
    rng = np.random.default_rng(seed)
    seq = [seqlen(r) for r in rows]
    slots = [(r, u) for r, rk in enumerate(start) for u in range(len(rk))]
    members = [list(m) for rk in start for m in rk]
    where = {i: s for s, m in enumerate(members) for i in m}
    load = [sum(seq[i] for i in m) for m in members]
    limit = max(MAX_TOKEN_LEN, max(load))
    n = len(rows)
    accepted = 0
    for a, b in rng.integers(0, n, size=(sweeps * n, 2)):
        a, b = int(a), int(b)
        sa, sb = where[a], where[b]
        if sa == sb:
            continue
        d = seq[b] - seq[a]
        if load[sa] + d > limit or load[sb] - d > limit:
            continue
        members[sa][members[sa].index(a)] = b
        members[sb][members[sb].index(b)] = a
        where[a], where[b] = sb, sa
        load[sa] += d
        load[sb] -= d
        accepted += 1
    out = [[None] * len(rk) for rk in start]
    for s, (r, u) in enumerate(slots):
        out[r][u] = [members[s][int(k)] for k in rng.permutation(len(members[s]))]
    random_micro.accepted, random_micro.limit = accepted, limit
    return out


def split_batch(rows: list[dict], n: int, partition: list[list[list[int]]] | None = None, dummies: int = 0
                ) -> tuple[list[dict], dict]:
    """Rank, micro-batch, position and weight of each row (rows in the input order of the step batch). partition:
    a given split (random partition); default the A1 split."""
    mini = MINI_CFG[n]
    seq = [seqlen(r) for r in rows]
    place = {}
    if partition is None:
        partition, dummies, rank_rows = a1_micro(rows)
        place = {i: k for rr in rank_rows for k, i in enumerate(rr)}
    if max(sum(len(m) for m in rk) for rk in partition) > mini:
        raise SystemExit(f"STOP: a rank has more than {mini} rows; the batch is not one mini-batch")
    count = len(partition[0])
    out = []
    for r, rk in enumerate(partition):
        rank_rows = sum(len(m) for m in rk)
        for u, idx in enumerate(rk):
            t_mb = sum(len(rows[i]["response_ids"]) for i in idx)
            for pos, i in enumerate(idx, 1):
                t_i = len(rows[i]["response_ids"])
                out.append({"rank": r, "mini": 0, "micro": u, "position": pos, "micro_rows": len(idx),
                            "rank_rows": rank_rows, "total_tokens": seq[i], "response_tokens": t_i, "loss_tokens": t_i,
                            "micro_loss_tokens": t_mb,
                            "weight": (len(idx) / mini) / RANKS * (t_i / t_mb if t_mb else 0.0), "input_index": i,
                            "rank_place": place.get(i)})
    info = {"rows": len(rows), "dummies": dummies, "rank_rows": [sum(len(m) for m in rk) for rk in partition],
            "micro_batches": count, "mini_cfg": mini, "weight_sum": sum(o["weight"] for o in out)}
    return out, info


def batch_inputs(by_prompt: dict[int, list[dict]], folds: dict[int, int], prompts: list[int]) -> dict[str, tuple[list[dict], int]]:
    """Input rows and n of each A1-split batch, in input order."""
    out = {}
    for f in range(1, K + 1):
        fp = sorted(p for p in prompts if folds[p] == f - 1)
        b8 = [r for p in fp for r in sorted(by_prompt[p], key=lambda r: r["sibling"])]
        out[f"B8-f{f}"] = (b8, 8)
        for j in range(SIBLINGS):
            out[f"B1-f{f}-j{j}"] = ([r for r in b8 if r["sibling"] == j], 1)
    return out


def neighbors(recs: list[dict]) -> dict[int, frozenset]:
    """traj -> the trajectories of its micro-batch."""
    groups: dict[tuple, set] = {}
    for r in recs:
        groups.setdefault((r["rank"], r["micro"]), set()).add(r["traj"])
    return {r["traj"]: frozenset(groups[(r["rank"], r["micro"])]) for r in recs}


def build(rollouts: list[dict], labels: dict[int, int], pilot_prompts: list[int] | None = None) -> tuple[dict, dict]:
    by_prompt: dict[int, list[dict]] = {}
    for r in rollouts:
        by_prompt.setdefault(r["prompt_pos"], []).append(r)
    if any(len(v) != SIBLINGS for v in by_prompt.values()):
        raise SystemExit("STOP: a prompt does not have 8 siblings")
    folds = fold_of(sorted(by_prompt))
    prompts = sorted(by_prompt) if pilot_prompts is None else sorted(pilot_prompts)
    batches, summary = {}, {}
    inputs = batch_inputs(by_prompt, folds, prompts)
    jobs = [(name, rows, n, None, 0) for name, (rows, n) in inputs.items() if rows]
    base_rows = inputs["B8-f1"][0]
    accepted = {}
    if base_rows:
        base_split, _, _ = a1_micro(base_rows)
        for s in PERM_SEEDS:
            jobs.append((f"R8-f1-s{s}", base_rows, 8, random_micro(base_split, base_rows, s), 0))
            accepted[f"R8-f1-s{s}"] = {"swaps_accepted": random_micro.accepted, "token_limit": random_micro.limit}
    for name, rows, n, partition, dummies in jobs:
        recs, info = split_batch(rows, n, partition, dummies)
        info.update(accepted.get(name, {}))
        for rec in recs:
            r = rows[rec.pop("input_index")]
            rec.update(batch=name, n=n, traj=r["traj"], prompt=r["prompt_pos"], sibling=r["sibling"],
                       fold=folds[r["prompt_pos"]] + 1, dataset_index=r["dataset_index"], label=labels[r["traj"]],
                       len_bin=length_bin(len(r["response_ids"])), dummies=info["dummies"])
        batches[name] = recs
        summary[name] = info
    if base_rows:
        base = neighbors(batches["B8-f1"])
        for s in PERM_SEEDS:
            name = f"R8-f1-s{s}"
            nb = neighbors(batches[name])
            summary[name]["rows_new_neighbors"] = sum(nb[t] != base[t] for t in base)
    return batches, {"fold_seed": FOLD_SEED, "perm_seeds": list(PERM_SEEDS), "ranks": RANKS,
                     "max_token_len": MAX_TOKEN_LEN, "batches": summary}


def main():
    rollouts_path, labels_path, out = sys.argv[1:4]
    pilot = json.loads(Path(sys.argv[4]).read_text())["prompts"] if len(sys.argv) > 4 else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda p: [json.loads(line) for line in open(p)]  # noqa: E731
    labels = {r["traj"]: int(r["label"]) for r in load(labels_path)}
    batches, summary = build(load(rollouts_path), labels, pilot)
    for name, recs in batches.items():
        with open(out / f"ROWS_{name}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
    (out / "BATCHES.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {"rows": v["rows"], "dummies": v["dummies"], "micro": v["micro_batches"]}
                      for k, v in summary["batches"].items()}))


if __name__ == "__main__":
    main()
