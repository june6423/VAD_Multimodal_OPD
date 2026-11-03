"""Part A item 9: the O4 gradient of moments.trajectory_step equals the A8 Objectives.gradients on a CPU toy actor.

The toy actor replaces DataParallelPPOActor._forward_micro_batch: a linear model over a small vocabulary that returns
log_probs, topk_logps and topk_indices with the semantics of dp_actor (log-probabilities over the full vocabulary,
top-k by the student logits, teacher values at given indices). The loss is the real compute_self_distillation_loss with
the vopd self-distillation config.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vad_grad"))
moments = pytest.importorskip("moments")
obj = moments.obj

V, L, H = 16, 6, 8


class ToyActor:
    def __init__(self, policy_config):
        torch.manual_seed(0)
        self.student = torch.nn.Linear(H, V, bias=False)
        self.teacher_module = torch.nn.Linear(H, V, bias=False).requires_grad_(False)
        self.config = policy_config  # holds self_distillation, loss_agg_mode

    def _forward_micro_batch(self, inputs, temperature, calculate_entropy=False, return_all_logps=False,
                             distill_topk=None, topk_indices=None, return_response_logits=False, module=None):
        model = module or self.student
        feats = inputs["input_ids"].float().reshape(1, L, H)
        logits = model(feats) / temperature
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        out = {"log_probs": (logits.gather(-1, inputs["responses"].unsqueeze(-1)) - lse).squeeze(-1)}
        if distill_topk is not None or topk_indices is not None:
            k = min(distill_topk or 0, V)  # dp_actor.py 589
            idx = torch.topk(logits, k, dim=-1).indices if topk_indices is None else topk_indices
            out["topk_logps"] = logits.gather(-1, idx) - lse
            if topk_indices is None:
                out["topk_indices"] = idx
        return out


@pytest.fixture(scope="module")
def setup():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from verl.utils.config import omega_conf_to_dataclass

    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="vopd")
    OmegaConf.resolve(cfg)
    actor = ToyActor(omega_conf_to_dataclass(cfg.actor_rollout_ref.actor))
    params = list(actor.student.parameters())
    gen = torch.Generator().manual_seed(1)
    mi = {"responses": torch.randint(0, V, (1, L), generator=gen), "response_mask": torch.ones(1, L)}
    for prefix in ("", "teacher_", "teacher_minus_", "hint_"):
        mi[prefix + "input_ids"] = torch.randn(1, L * H, generator=gen)
        mi[prefix + "attention_mask"] = torch.ones(1, L)
        mi[prefix + "position_ids"] = torch.zeros(1, L, dtype=torch.long)
        mi[prefix + "response_start_idx"] = torch.zeros(1, dtype=torch.long)
        mi[prefix + "multi_modal_inputs"] = None
    return SimpleNamespace(actor=actor, params=params, objectives=moments.Capture(actor, params)), mi


def test_o4_gradient_equals_a8(setup):
    ctx, mi = setup
    a8 = obj.Objectives(ctx.actor, ctx.params).gradients(mi, ["O4"])["O4"][1]
    new0, rec = moments.trajectory_step(ctx, mi, 0.0, None)
    a_buf = [torch.zeros_like(p, dtype=torch.float32) for p in ctx.params]
    new1, _ = moments.trajectory_step(ctx, mi, 0.75, a_buf)
    for a, b, c in zip(a8, new0, new1):
        assert torch.equal(a, b) and torch.equal(a, c)
    assert rec["sq_norm"] == pytest.approx(float(sum(x.pow(2).sum() for x in a8)))
    logp = (ctx.actor._forward_micro_batch(mi, 1.0)["log_probs"] * mi["response_mask"]).sum()
    expected = torch.autograd.grad(logp, ctx.params)
    for a, e in zip(a_buf, expected):
        assert torch.allclose(a, 0.75 * e, atol=1e-6)
