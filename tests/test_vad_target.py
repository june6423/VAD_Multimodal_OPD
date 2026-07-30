"""Unit tests for the public VAD budgeted asymmetric target."""

import torch

from verl.trainer.ppo.core_algos import compute_self_distillation_loss
from verl.workers.config.actor import SelfDistillationConfig


def make_cfg(**overrides):
    defaults = dict(
        full_logit_distillation=True,
        alpha=0.5,
        distillation_topk=4,
        distillation_add_tail=False,
        target_mode="vc",
        counterfactual_teacher_image_key="degraded_bbox_images",
        teacher_always_on=True,
        teacher_image_key="bbox_images",
        vc_target_variant="budgeted_asvc",
        vc_positive_budget_cap=0.8,
        vc_nonvisual_anchor_eta=0.0,
        is_clip=None,
    )
    defaults.update(overrides)
    return SelfDistillationConfig(**defaults)


def test_vad_budgeted_target_loss_and_metrics():
    student = torch.log_softmax(torch.tensor([[[0.0, 1.0, -1.0, 0.5]]]), dim=-1).requires_grad_(True)
    teacher_plus = torch.log_softmax(torch.tensor([[[1.2, 0.2, -0.5, 0.0]]]), dim=-1)
    teacher_minus = torch.log_softmax(torch.tensor([[[-0.5, 1.2, 0.1, 0.0]]]), dim=-1)
    response_mask = torch.ones(1, 1)

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=torch.zeros(1, 1),
        teacher_log_probs=torch.zeros(1, 1),
        response_mask=response_mask,
        self_distillation_config=make_cfg(),
        student_topk_log_probs=student,
        teacher_topk_log_probs=teacher_plus,
        teacher_minus_topk_log_probs=teacher_minus,
        self_distillation_mask=torch.ones(1),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert metrics["self_distillation/vc_target_variant_code"] == 1.0
    assert metrics["self_distillation/basvc_tau_pos"] == 0.8
    assert metrics["self_distillation/basvc_alpha_pos.mean"] <= 0.8 + 1e-6
    assert metrics["self_distillation/basvc_shift_norm_over_single.mean"] <= 1.0 + 1e-5
    assert "self_distillation/vc_rho_single.mean" in metrics
