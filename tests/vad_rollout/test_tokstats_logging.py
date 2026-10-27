"""VLM-VAD-ROLLOUT-001 gates G1 and G2: the token-statistics logging is logging only and joins the dump."""

import json
import os

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_self_distillation_loss
from verl.workers.config.actor import SelfDistillationConfig

VOCAB = 300
TOPK = 100
VECTORS = ("r_norm", "rho", "jsd", "anchor_jsd")


def a1_config():
    # The A1 values of scripts/train_vad.sh and configs/vad_4b.env.
    return SelfDistillationConfig(
        full_logit_distillation=True,
        alpha=0.5,
        distillation_topk=TOPK,
        distillation_add_tail=True,
        target_mode="vc",
        counterfactual_teacher_image_key="degraded_bbox_images",
        teacher_always_on=True,
        teacher_image_key="bbox_images",
        vc_target_variant="budgeted_asvc",
        vc_positive_budget_cap=0.8,
        vc_projection_ridge=1e-3,
        vc_step_alpha=1.0,
        vc_max_logit_shift=20.0,
        vc_nonvisual_anchor_eta=0.1,
        is_clip=2.0,
    )


def make_inputs(response_mask, seed):
    """Random student, teacher-plus and teacher-minus logits, gathered at the student top-k as dp_actor does."""
    generator = torch.Generator().manual_seed(seed)
    batch, length = response_mask.shape
    shape = (batch, length, VOCAB)
    student_logits = torch.randn(shape, generator=generator)
    plus = torch.log_softmax(torch.randn(shape, generator=generator), dim=-1)
    minus = torch.log_softmax(torch.randn(shape, generator=generator), dim=-1)
    sampled = torch.randint(0, VOCAB, (batch, length, 1), generator=generator)
    old_shift = 0.05 * torch.randn((batch, length), generator=generator)
    indices = torch.log_softmax(student_logits, dim=-1).topk(TOPK, dim=-1).indices
    return {"student_logits": student_logits, "plus": plus, "minus": minus, "sampled": sampled,
            "old_shift": old_shift, "indices": indices, "response_mask": response_mask}


def run_loss(inputs, tokstats_out):
    logits = inputs["student_logits"].clone().requires_grad_(True)
    student = torch.log_softmax(logits, dim=-1)
    student_log_probs = student.gather(-1, inputs["sampled"]).squeeze(-1)
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=inputs["plus"].gather(-1, inputs["sampled"]).squeeze(-1),
        response_mask=inputs["response_mask"],
        self_distillation_config=a1_config(),
        old_log_probs=(student_log_probs + inputs["old_shift"]).detach(),
        student_topk_log_probs=student.gather(-1, inputs["indices"]),
        teacher_topk_log_probs=inputs["plus"].gather(-1, inputs["indices"]),
        teacher_minus_topk_log_probs=inputs["minus"].gather(-1, inputs["indices"]),
        self_distillation_mask=torch.ones(inputs["response_mask"].shape[0]),
        tokstats_out=tokstats_out,
    )
    loss.backward()
    return loss.detach(), metrics, logits.grad


def actor_tokstats_arg():
    # dp_actor.update_policy passes a dict only when VAD_TOKSTATS_DIR is set.
    return {} if os.environ.get("VAD_TOKSTATS_DIR") else None


def test_g1_loss_and_metrics_bitwise_equal(tmp_path, monkeypatch):
    mask = torch.ones(4, 9)
    mask[1, 6:] = 0
    mask[3, 2:] = 0
    inputs = make_inputs(mask, seed=0)

    monkeypatch.delenv("VAD_TOKSTATS_DIR", raising=False)
    unset_arg = actor_tokstats_arg()
    loss_unset, metrics_unset, grad_unset = run_loss(inputs, unset_arg)
    monkeypatch.setenv("VAD_TOKSTATS_DIR", str(tmp_path))
    set_arg = actor_tokstats_arg()
    loss_set, metrics_set, grad_set = run_loss(inputs, set_arg)

    assert unset_arg is None and set_arg is not None
    assert torch.equal(loss_unset, loss_set)
    assert sorted(metrics_unset) == sorted(metrics_set)
    differing = [k for k in metrics_unset if not (metrics_unset[k] == metrics_set[k])]
    assert differing == []
    assert torch.equal(grad_unset, grad_set)
    assert sorted(set_arg) == sorted(VECTORS)
    assert all(set_arg[name].shape == mask.shape and not set_arg[name].requires_grad for name in VECTORS)
    print(f"G1 loss unset {loss_unset.item().hex()} set {loss_set.item().hex()} equal True; "
          f"metrics compared {len(metrics_unset)} differing 0; grad equal True; "
          f"anchor metric present {'self_distillation/vc_nonvisual_anchor_eta' in metrics_set}")


class FakeTokenizer:
    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(f"t{int(t)}" for t in row if int(t) != 0) for row in ids]


def test_g2_dump_rows_join_token_statistics(tmp_path, monkeypatch):
    # G2 alone needs the trainer and the actor modules. G1 needs only core_algos.
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    monkeypatch.setenv("VAD_TOKSTATS_DIR", str(tmp_path / "tokstats"))
    generator = torch.Generator().manual_seed(1)
    prompts, siblings, length = 2, 8, 7
    rows = prompts * siblings
    lengths = torch.randint(1, length + 1, (rows,), generator=generator)
    response_mask = (torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)).float()
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.randint(1, 50, (rows, 5), generator=generator),
            "responses": torch.randint(1, 50, (rows, length), generator=generator) * response_mask.long(),
            "response_mask": response_mask,
        },
        non_tensors={
            "uid": np.array([f"uid-{p}" for p in range(prompts) for _ in range(siblings)], dtype=object),
            "reward_model": np.array([{"ground_truth": "A"} for _ in range(rows)], dtype=object),
        },
    )
    # _balance_batch reorders the rows before the actor update.
    batch.reorder(torch.randperm(rows, generator=generator))
    # The line of RayPPOTrainer.fit before the actor update.
    if os.environ.get("VAD_TOKSTATS_DIR"):
        batch.batch["row_index"] = torch.arange(len(batch), dtype=torch.long)

    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = FakeTokenizer()
    trainer.global_steps = 3
    trainer._log_rollout_data(batch, {}, {}, str(tmp_path / "rollouts"))

    # Two data-parallel ranks, two micro-batches each, as the actor update splits the step batch.
    for rank, shard in enumerate(batch.chunk(2)):
        records = []
        for micro_index, micro in enumerate(shard.split(4)):
            tokstats = actor_tokstats_arg()
            run_loss(make_inputs(micro.batch["response_mask"], seed=10 * rank + micro_index), tokstats)
            records += DataParallelPPOActor._collect_tokstats_rows(
                tokstats=tokstats,
                response_mask=micro.batch["response_mask"],
                row_index=micro.batch["row_index"],
                uids=micro.non_tensor_batch["uid"],
            )
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        DataParallelPPOActor._write_tokstats_rows(
            tokstats_dir=os.environ["VAD_TOKSTATS_DIR"], meta_info={"global_steps": 3}, rows=records
        )

    dump = [json.loads(line) for line in (tmp_path / "rollouts" / "3.jsonl").read_text().splitlines()]
    stat_files = sorted((tmp_path / "tokstats").glob("step_3_rank_*.jsonl"))
    stats = [json.loads(line) for path in stat_files for line in path.read_text().splitlines()]
    valid_length = {int(i): int(n) for i, n in zip(batch.batch["row_index"], batch.batch["response_mask"].sum(-1))}

    assert len(dump) == rows and all("uid" in row and "row_index" in row for row in dump)
    assert sorted(row["row_index"] for row in dump) == list(range(rows))
    assert [p.name for p in stat_files] == ["step_3_rank_0.jsonl", "step_3_rank_1.jsonl"]
    assert len(stats) == rows
    by_index = {}
    for record in stats:
        by_index.setdefault(record["row_index"], []).append(record)
    for row in dump:
        matches = by_index.get(row["row_index"], [])
        assert len(matches) == 1
        record = matches[0]
        assert record["uid"] == row["uid"]
        assert record["response_length"] == valid_length[row["row_index"]]
        assert all(len(record[name]) == valid_length[row["row_index"]] for name in VECTORS)
    print(f"G2 dump rows {len(dump)} with uid and row_index; token-statistics records {len(stats)} in "
          f"{len(stat_files)} files; one record per dump row with equal uid and row_index; "
          f"vector lengths equal valid lengths {sorted(valid_length.values())}")
