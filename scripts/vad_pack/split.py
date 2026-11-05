"""Batches of Definition 5 and the A1 row split of Definition 6 (HANDOFF-VLM-VAD-PACK-001-R1 Part A item 1).

Batches: numpy default_rng(20261106) permutes the 375 prompt positions. The first 288 make G1, G2 and G3 (96 each, in
permutation order). The same generator then draws one sibling (0 to 7) for each of the 288 prompts (A2 batches).
  A1-S0-k: the 8 S0 siblings of each prompt of Gk (model M0). A1-S80-k: the 8 S80 siblings (model A1_80).
  A2-S0-k: the drawn S0 sibling of each prompt of Gk (model M0).
Input order of a batch (the step batch before balance_batch): the prompts in group order, the siblings in sibling order
(agent loop: each prompt repeated rollout.n times, interleaved).

Row split (the verl code of the A1 run, trainer_values.py gives the source lines):
  1. balance_batch: workload = 24576 x s + s^2 with s = prompt_len + response tokens (the attention-mask sum).
     Equal-size get_seqlen_balanced_partitions over the ranks, each partition sorted by (workload, index), then the
     even places and the odd places reversed (ray_trainer.py 2483-2494).
  2. Dispatch: rank r gets partition r (contiguous equal chunks of the reordered batch).
  3. Mini-batches of rank_mini_batch_size rows (fsdp_workers.py 244-245, dp_actor.py 1381).
  4. Micro-batches of each mini-batch: count = max over the ranks of min(rows, ceil(tokens / 9216)) (all_reduce MAX),
     unequal-size get_seqlen_balanced_partitions, sorted by (workload sum, first index) descending, then the even places
     reversed and the odd places (seqlen_balancing.py 387-414). The rows of a micro-batch keep their sorted indices.
Position: the 1-based place of a row in its micro-batch (the order of the packed sequence).

Usage: python split.py <S0 rollouts.jsonl> <S80 rollouts.jsonl> <S0 LABELS.jsonl> <S80 LABELS.jsonl> <out dir>
Writes ROWS_<batch>.jsonl (one row per line, in execution order) and SPLIT.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository (verl)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trainer_values as tv  # noqa: E402

SEED = 20261106
GROUP_SIZE = 96
GROUPS = 3
SIBLINGS = 8
BATCHES = [f"{arm}-{s}-{k}" for arm, s in (("A1", "S0"), ("A1", "S80"), ("A2", "S0")) for k in (1, 2, 3)]


def workload(seqlens: list[int]) -> list[int]:
    from verl.utils.seqlen_balancing import calculate_workload
    import torch

    return calculate_workload(torch.tensor(seqlens, dtype=torch.long)).tolist()


def balanced(work: list[int], k: int, equal_size: bool) -> list[list[int]]:
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

    return get_seqlen_balanced_partitions(work, k_partitions=k, equal_size=equal_size)


def balance_order(seqlens: list[int], ranks: int) -> list[list[int]]:
    """Rank partitions of balance_batch (ray_trainer.py 2436-2494): batch indices of each rank, in order."""
    work = workload(seqlens)
    parts = balanced(work, ranks, equal_size=True)
    out = []
    for p in parts:
        p = sorted(p, key=lambda x: (work[x], x))
        out.append(p[::2] + p[1::2][::-1])
    return out


def micro_partition(seqlens: list[int], count: int) -> list[list[int]]:
    """Micro-batches of one mini-batch (seqlen_balancing.py 401-414): indices into the mini-batch, in execution order."""
    work = workload(seqlens)
    parts = balanced(work, count, equal_size=False)
    parts.sort(key=lambda p: (sum(work[i] for i in p), p[0] if p else 0), reverse=True)
    return parts[::2][::-1] + parts[1::2]


def micro_count(seqlens: list[int], max_token_len: int) -> int:
    """seqlen_balancing.py 387 for one rank, before the all_reduce MAX."""
    if max(seqlens) > max_token_len:
        raise SystemExit(f"STOP: a row has {max(seqlens)} tokens, above max_token_len {max_token_len}")
    return min(len(seqlens), math.ceil(sum(seqlens) / max_token_len))


def split_batch(rows: list[dict], arm: str) -> list[dict]:
    """Rank, micro-batch and position of each row (rows in the input order of the step batch)."""
    v = tv.VALUES[arm]
    ranks = v["ranks"] // v["ulysses"]
    max_tok = v["max_token_len"] * v["ulysses"]
    mini = tv.rank_mini_batch_size(arm)
    seq = [r["prompt_len"] + len(r["response_ids"]) for r in rows]
    if len(rows) % ranks:
        raise SystemExit(f"STOP: {len(rows)} rows do not split into {ranks} ranks")
    rank_rows = balance_order(seq, ranks)
    out = []
    n_mini = math.ceil(len(rank_rows[0]) / mini)
    for m in range(n_mini):
        minis = [rr[m * mini:(m + 1) * mini] for rr in rank_rows]
        count = max(micro_count([seq[i] for i in mb], max_tok) for mb in minis)  # all_reduce MAX over the ranks
        for r, mb in enumerate(minis):
            for u, part in enumerate(micro_partition([seq[i] for i in mb], count)):
                for pos, j in enumerate(part, 1):
                    i = mb[j]
                    out.append({"rank": r, "mini": m, "micro": u, "position": pos, "micro_rows": len(part),
                                "mini_rows": len(mb), "total_tokens": seq[i], "response_tokens": len(rows[i]["response_ids"]),
                                "input_index": i})
    return out


def select(prompts: list[int]) -> tuple[list[list[int]], dict[int, int]]:
    """Groups G1 to G3 and the A2 sibling of each selected prompt (Definition 5)."""
    rng = np.random.default_rng(SEED)
    perm = [int(p) for p in rng.permutation(sorted(prompts))[:GROUP_SIZE * GROUPS]]
    sib = [int(s) for s in rng.integers(0, SIBLINGS, size=len(perm))]
    groups = [perm[g * GROUP_SIZE:(g + 1) * GROUP_SIZE] for g in range(GROUPS)]
    return groups, dict(zip(perm, sib))


def build(s0: list[dict], s80: list[dict], labels: dict[str, dict[int, int]]) -> dict[str, list[dict]]:
    by = {"S0": {}, "S80": {}}
    for name, rows in (("S0", s0), ("S80", s80)):
        for r in rows:
            by[name].setdefault(r["prompt_pos"], []).append(r)
    groups, a2_sib = select(sorted(by["S0"]))
    batches = {}
    for name in BATCHES:
        arm, s, k = name.split("-")
        rows = []
        for p in groups[int(k) - 1]:
            sibs = sorted(by[s][p], key=lambda r: r["sibling"])
            rows += sibs if arm == "A1" else [r for r in sibs if r["sibling"] == a2_sib[p]]
        recs = split_batch(rows, arm)
        for rec in recs:
            r = rows[rec.pop("input_index")]
            rec.update(batch=name, arm=arm, set=s, traj=r["traj"], prompt=r["prompt_pos"], sibling=r["sibling"],
                       dataset_index=r["dataset_index"], label=labels[s][r["traj"]])
        batches[name] = recs
    return batches


def main():
    s0, s80, l0, l80, out = sys.argv[1:6]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda p: [json.loads(line) for line in open(p)]  # noqa: E731
    labels = {"S0": {r["traj"]: int(r["label"]) for r in load(l0)}, "S80": {r["traj"]: int(r["label"]) for r in load(l80)}}
    batches = build(load(s0), load(s80), labels)
    summary = {}
    for name, recs in batches.items():
        with open(out / f"ROWS_{name}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        micro = {(r["rank"], r["mini"], r["micro"]) for r in recs}
        summary[name] = {"rows": len(recs), "micro_batches": len(micro), "rank_mini_batch_size": tv.rank_mini_batch_size(name[:2]),
                         "max_micro_rows": max(r["micro_rows"] for r in recs),
                         "rows_position_2plus": sum(r["position"] >= 2 for r in recs)}
    (out / "SPLIT.json").write_text(json.dumps({"seed": SEED, "batches": summary}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
