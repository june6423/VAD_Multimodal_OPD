"""A8 objectives O1 to O4 on one trajectory (HANDOFF-VLM-VAD-ROLLOUT-001-R9, Part A items 3 and 7).

All forwards are DataParallelPPOActor._forward_micro_batch (the A1 code path) at temperature 1.0.
- O1, OPSD paper form: hint teacher, sum_v min(q_v log(q_v / p_v), tau) over the full vocabulary (q teacher,
  p student, tau 0.05, C-OPSD section 4.1), token mean over the response.
- O2, OPSD matched form: hint teacher, compute_self_distillation_loss with target_mode full_teacher.
- O3, Vision-OPD form: x-plus teacher, compute_self_distillation_loss with target_mode full_teacher.
- O4, VAD form: x-plus and x-minus teachers, compute_self_distillation_loss with the A1 config (target_mode vc).
O2 to O4 use the student top-100 indices, as in update_policy (dp_actor.py 1464-1739). The importance ratio is 1:
old_log_probs is log_prob.detach() (on_policy, 1485-1486) and no rollout_is_weights are given.
The gradient of a trajectory is the gradient of its own token-mean loss, without the rows/128 factor of training.
"""

from __future__ import annotations

import copy
import dataclasses
import re

import torch
from omegaconf import open_dict

TAU = 0.05
TOPK_OBJECTIVES = ("O2", "O3", "O4")
VIEW_PREFIX = {"hint": "hint_", "plus": "teacher_", "minus": "teacher_minus_"}


def with_target_mode(sd, mode: str):
    if dataclasses.is_dataclass(sd):
        return dataclasses.replace(sd, target_mode=mode)
    sd = copy.deepcopy(sd)
    with open_dict(sd):
        sd.target_mode = mode
    return sd


def view_inputs(mi: dict, view: str) -> dict:
    """Teacher inputs as update_policy builds them (dp_actor.py 1497-1505, 1548-1558)."""
    p = VIEW_PREFIX[view]
    return {"responses": mi["responses"], "input_ids": mi[p + "input_ids"], "attention_mask": mi[p + "attention_mask"],
            "position_ids": mi[p + "position_ids"], "response_start_idx": mi[p + "response_start_idx"],
            "multi_modal_inputs": mi[p + "multi_modal_inputs"]}


class Objectives:
    def __init__(self, actor, params: list[torch.nn.Parameter]):
        self.actor = actor
        self.params = params
        sd = actor.config.self_distillation
        self.topk = sd.distillation_topk
        self.configs = {"O2": with_target_mode(sd, "full_teacher"), "O3": with_target_mode(sd, "full_teacher"), "O4": sd}
        self.teacher_of = {"O2": "hint", "O3": "plus", "O4": "plus"}

    def _forward(self, inputs: dict, grad: bool, **kwargs) -> dict:
        module = None if grad else self.actor.teacher_module
        with torch.set_grad_enabled(grad):
            return self.actor._forward_micro_batch(inputs, temperature=1.0, calculate_entropy=False, module=module, **kwargs)

    def losses(self, mi: dict, names: list[str]) -> dict[str, torch.Tensor]:
        """Losses of the named objectives for one trajectory (mi: model inputs of a one-row micro-batch)."""
        from verl.trainer.ppo.core_algos import compute_self_distillation_loss

        mask = mi["response_mask"]
        out = {}
        topk_names = [n for n in names if n in TOPK_OBJECTIVES]
        if topk_names:
            student = self._forward(mi, grad=True, distill_topk=self.topk)
            idx = student["topk_indices"]
            views = {self.teacher_of[n] for n in topk_names} | ({"minus"} if "O4" in topk_names else set())
            teacher = {v: self._forward(view_inputs(mi, v), grad=False, topk_indices=idx) for v in sorted(views)}
            for n in topk_names:
                t = teacher[self.teacher_of[n]]
                out[n], _ = compute_self_distillation_loss(
                    student_log_probs=student["log_probs"], teacher_log_probs=t["log_probs"], response_mask=mask,
                    self_distillation_config=self.configs[n], old_log_probs=student["log_probs"].detach(),
                    student_topk_log_probs=student["topk_logps"], teacher_topk_log_probs=t["topk_logps"],
                    teacher_minus_topk_log_probs=teacher["minus"]["topk_logps"] if n == "O4" else None,
                    distill_support_mask=None, self_distillation_mask=mi.get("self_distillation_mask"),
                    loss_agg_mode=self.actor.config.loss_agg_mode, rollout_is_weights=None,
                    policy_config=self.actor.config, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None)
        if "O1" in names:
            s = self._forward(mi, grad=True, return_all_logps=True)["all_logps"]
            q = self._forward(view_inputs(mi, "hint"), grad=False, return_all_logps=True)["all_logps"]
            per_token = torch.clamp(q.exp() * (q - s), max=TAU).sum(-1)
            out["O1"] = (per_token * mask).sum() / mask.sum().clamp(min=1)
        return out

    def gradients(self, mi: dict, names: list[str]) -> dict[str, tuple[float, list[torch.Tensor]]]:
        """(loss, gradient as a list of tensors over self.params) for each named objective."""
        losses = self.losses(mi, names)
        order = [n for n in names if n in losses]
        out = {}
        for i, n in enumerate(order):
            # O2 to O4 share one student graph; keep it until the last of them.
            keep = n in TOPK_OBJECTIVES and any(m in TOPK_OBJECTIVES for m in order[i + 1:])
            grads = torch.autograd.grad(losses[n], self.params, retain_graph=keep)
            out[n] = (losses[n].item(), list(grads))
        return out

    def teacher_check(self, mi: dict, position: int, gt_token: int, sampled_token: int) -> dict:
        """x-plus teacher at the answer-letter response position: argmax == ground-truth letter, and the log-odds gap."""
        lp = self._forward(view_inputs(mi, "plus"), grad=False, return_all_logps=True)["all_logps"][0, position]
        return {"flag": "f1" if int(lp.argmax()) == gt_token else "f0",
                "gap": float(lp[gt_token] - lp[sampled_token]), "argmax_token": int(lp.argmax())}


def answer_letter_char(text: str):
    """(char offset, letter) of the letter that the A1 rule grader extracts, or None.

    Mirrors extract_answer and extract_first_option (opd_benchmark_judge.py 82-132): the last match of the first
    pattern that matches, inside the extracted answer text.
    """
    from verl.utils.reward_score.opd_benchmark_judge import extract_answer, extract_first_option

    if "<answer>" in text and text.find("</answer>") != -1:
        start = text.find("<answer>") + len("<answer>")
        segment = text[start:text.find("</answer>")]
    elif "Answer:" in text:
        start = text.rfind("Answer:")
        segment = text[start:]
    else:
        start, segment = 0, text
    lead = len(segment) - len(segment.lstrip())
    start, segment = start + lead, segment.strip()
    letter = extract_first_option(extract_answer(text))
    if not letter:
        return None
    patterns = [
        r"(?:final\s+answer|correct\s+answer|answer|答案)\s*[:：]\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"(?:final\s+answer|correct\s+answer|answer|答案)\s+is\s+[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"(?:option|choice)\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"(?:choose|select|selected|pick|picked)\s+(?:option|choice)?\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
        r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\]]?\s*[*_`]*\s*$",
        r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\].:：]",
        r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\]]?\s+(?:because|is|should|seems|appears)\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, segment, flags=re.IGNORECASE))
        if matches:
            m = matches[-1]
            if m.group(1).upper() != letter:
                return None
            return start + m.start(1), letter
    return None


def letter_token(tokenizer, response_ids: list[int], letter_char: int) -> int | None:
    """Index of the response token that holds the character at letter_char of the decoded response."""
    for i in range(len(response_ids)):
        if len(tokenizer.decode(response_ids[: i + 1], skip_special_tokens=True)) > letter_char:
            return i
    return None


def gt_token_like(tokenizer, sampled_token: int, sampled_letter: str, gt_letter: str) -> int | None:
    """The token that equals the sampled letter token with the ground-truth letter in its place, if it is one token."""
    piece = tokenizer.decode([sampled_token])
    if sampled_letter.upper() not in piece.upper():
        return None
    pos = piece.upper().rfind(sampled_letter.upper())
    replaced = piece[:pos] + (gt_letter if piece[pos].isupper() else gt_letter.lower()) + piece[pos + 1:]
    ids = tokenizer.encode(replaced, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None
