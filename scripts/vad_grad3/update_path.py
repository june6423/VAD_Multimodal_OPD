"""The dp_actor update path of 2627743a for one step, without an optimizer step (gates G1 and G2 of
HANDOFF-VLM-VAD-GRAD-003-R2).

update_path_gradient runs DataParallelPPOActor.update_policy once for each rank on the rows of that rank, in the order
of the rank part of balance_batch. Changes to the actor, for the call only:
  1. _optimizer_step returns 0 and does not step; _update_teacher does nothing.
  2. The actor optimizer only clears the gradients.
  3. prepare_dynamic_batch gets min_num_micro_batch = the largest micro-batch count of the 6 ranks. This gives the
     all_reduce MAX of seqlen_balancing.py 391-394 in one process.
  4. A post-accumulate hook adds each bf16 parameter gradient of a backward to an fp32 sum and clears it (FSDP keeps
     fp32 gradients).
The step gradient is the mean of the rank sums (FSDP gradient averaging over the 6 ranks).
"""

from __future__ import annotations

import functools

import torch


class _ClearOnly:
    def __init__(self, module):
        self.module = module

    def zero_grad(self, set_to_none: bool = True):
        for p in self.module.parameters():
            p.grad = None


def update_path_gradient(actor, params: list, rank_data: list, count: int, ranks: int = 6):
    """(fp32 step gradient as a list over params, the micro-batch index lists of each rank)."""
    import verl.workers.actor.dp_actor as dpa

    acc = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    rank_acc = [torch.zeros_like(a) for a in acc]

    def hook_for(k):
        def hook(p):
            rank_acc[k].add_(p.grad.float())
            p.grad = None
        return hook

    handles = [p.register_post_accumulate_grad_hook(hook_for(k)) for k, p in enumerate(params)]
    partitions = []
    orig = dpa.prepare_dynamic_batch

    def prepare(data, max_token_len, **kwargs):
        micro, idx = orig(data, max_token_len=max_token_len, min_num_micro_batch=count, **kwargs)
        partitions[-1].append([list(map(int, p)) for p in idx])
        return micro, idx

    missing = object()
    saved = {k: actor.__dict__.get(k, missing) for k in ("_optimizer_step", "_update_teacher", "actor_optimizer")}
    dpa.prepare_dynamic_batch = prepare
    actor._optimizer_step = lambda: torch.tensor(0.0)
    actor._update_teacher = lambda: None
    actor.actor_optimizer = _ClearOnly(actor.actor_module)
    try:
        for data in rank_data:
            for a in rank_acc:
                a.zero_()
            partitions.append([])
            actor.update_policy(data)
            for a, r in zip(acc, rank_acc):
                a.add_(r, alpha=1.0 / ranks)
    finally:
        dpa.prepare_dynamic_batch = orig
        for h in handles:
            h.remove()
        for k, v in saved.items():
            if v is missing:
                actor.__dict__.pop(k, None)
            else:
                setattr(actor, k, v)
    return acc, [p[0] for p in partitions]


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Relative difference ||a - b|| / ||b|| and cosine, fp64 (a: analysis, b: update path)."""
    a64, b64 = a.double(), b.double()
    nb = float(b64.norm())
    return {"rel_diff": float((a64 - b64).norm()) / nb if nb > 0 else float("nan"),
            "cos": float(a64 @ b64) / (float(a64.norm()) * nb) if nb > 0 else float("nan"),
            "norm_analysis": float(a64.norm()), "norm_update": nb}
