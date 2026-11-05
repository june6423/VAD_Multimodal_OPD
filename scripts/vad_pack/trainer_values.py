"""A1 and A2 trainer values of Definition 6 (HANDOFF-VLM-VAD-PACK-001-R3, gate G1), each with its source file and line,
and the loss path of the old and rollout log-probabilities (R3 Part A items 4 and 5).

Sources:
  sources/A1_config.yaml: .hydra/config.yaml of the A1 run (vast backup harness/vad_train/outputs/2026-10-29/07-22-21,
    sha256 1ef79f45...; the A1 restarts 2026-10-30/01-39-47 and 06-09-04 have the same file).
  sources/A2_config.yaml: .hydra/config.yaml of the A2 run (2026-10-30/21-33-31, sha256 19b11133...; the A2 restarts
    2026-10-31/03-39-23 and 07-42-32 have the same file).
  verl/: the verl code of the repository. verl/ is the same at the A1 and A2 commit 2627743a and at this branch.
check() reads each quoted line and stops when the line does not contain the expected text.
VALUES gives the numbers that split.py uses.

Usage: python trainer_values.py <repo root> <out .md>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = {"A1": "scripts/vad_pack/sources/A1_config.yaml", "A2": "scripts/vad_pack/sources/A2_config.yaml"}

# (name, line, expected text) in the config file of each arm. The line numbers are equal in the two files.
CONFIG_LINES = [
    ("trainer.nnodes", 675, "nnodes: 1"),
    ("trainer.n_gpus_per_node (data-parallel ranks = nnodes x n_gpus_per_node / ulysses size)", 676, "n_gpus_per_node: 6"),
    ("actor.ulysses_sequence_parallel_size", 195, "ulysses_sequence_parallel_size: 1"),
    ("data.train_batch_size (prompts per step)", 407, "train_batch_size: 96"),
    ("actor.ppo_mini_batch_size (prompts)", 45, "ppo_mini_batch_size: 96"),
    ("rollout.n", 297, {"A1": "'n': 8", "A2": "'n': 1"}),
    ("trainer.balance_batch", 664, "balance_batch: true"),
    ("model.use_remove_padding", 379, "use_remove_padding: true"),
    ("actor.use_dynamic_bsz", 48, "use_dynamic_bsz: true"),
    ("actor.ppo_max_token_len_per_gpu", 49, "ppo_max_token_len_per_gpu: 9216"),
    ("rollout.log_prob_use_dynamic_bsz (= actor.use_dynamic_bsz)", 293,
     "log_prob_use_dynamic_bsz: ${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}"),
    ("rollout.log_prob_max_token_len_per_gpu, the log-probability token limit (= actor.ppo_max_token_len_per_gpu)", 294,
     "log_prob_max_token_len_per_gpu: ${oc.select:actor_rollout_ref.actor.ppo_max_token_len_per_gpu,16384}"),
    ("actor.ppo_epochs", 152, "ppo_epochs: 1"),
    ("actor.shuffle", 153, "shuffle: false"),
    ("actor.loss_agg_mode", 143, "loss_agg_mode: token-mean"),
    ("actor.use_prefix_grouper", 148, "use_prefix_grouper: false"),
    # R3 Part A items 4 and 5: the loss path of the old and rollout log-probabilities
    ("actor.policy_loss.loss_mode", 58, "loss_mode: vopd"),
    ("actor.self_distillation.target_mode (not vzero: no sampled-token PPO surrogate)", 85, "target_mode: vc"),
    ("actor.self_distillation.is_clip (the only use of old_log_probs in the vopd loss)", 125, "is_clip: 2.0"),
    ("algorithm.rollout_correction.rollout_is", 626, "rollout_is: token"),
    ("algorithm.rollout_correction.rollout_is_threshold", 627, "rollout_is_threshold: 2.0"),
    ("algorithm.rollout_correction.bypass_mode", 630, "bypass_mode: false"),
]

# (meaning, file, line, expected text) in the verl code.
CODE_LINES = [
    ("balance_batch is applied to the step batch", "verl/trainer/ppo/ray_trainer.py", 2778, "if self.config.trainer.balance_batch:"),
    ("balance_batch call", "verl/trainer/ppo/ray_trainer.py", 2779, "self._balance_batch(batch, metrics=metrics)"),
    ("rank partition: equal-size Karmarkar-Karp over the dp ranks", "verl/trainer/ppo/ray_trainer.py", 2483,
     "get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)"),
    ("order inside a rank partition: sort by (workload, index)", "verl/trainer/ppo/ray_trainer.py", 2488,
     "partition.sort(key=lambda x: (workload_lst[x], x))"),
    ("order inside a rank partition: even places, then odd places reversed", "verl/trainer/ppo/ray_trainer.py", 2489,
     "ordered_partition = partition[::2] + partition[1::2][::-1]"),
    ("the batch is reordered rank after rank", "verl/trainer/ppo/ray_trainer.py", 2493,
     "global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])"),
    ("dispatch splits the batch into dp_size parts", "verl/single_controller/base/decorator.py", 280,
     "_split_args_kwargs_data_proto(dp_size, *args, **kwargs)"),
    ("each part is a contiguous chunk along dim 0", "verl/protocol.py", 891, "self.batch.chunk(chunks=chunks, dim=0)"),
    ("ppo_mini_batch_size per rank: x rollout.n", "verl/workers/fsdp_workers.py", 244,
     "self.config.actor.ppo_mini_batch_size *= self.config.rollout.n"),
    ("ppo_mini_batch_size per rank: // (ranks / ulysses size)", "verl/workers/fsdp_workers.py", 245,
     "self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size"),
    ("mini-batches of a rank", "verl/workers/actor/dp_actor.py", 1381, "mini_batches = data.split(self.config.ppo_mini_batch_size)"),
    ("on_policy: one mini-batch and one epoch", "verl/workers/actor/dp_actor.py", 1383,
     "on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1"),
    ("micro-batch token limit", "verl/workers/actor/dp_actor.py", 1413,
     "max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size"),
    ("micro-batches of a mini-batch", "verl/workers/actor/dp_actor.py", 1414,
     "micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)"),
    ("micro-batch loss scale B_m / B_mini", "verl/workers/actor/dp_actor.py", 1444,
     "loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size"),
    ("old log-probabilities from the training forward (on_policy)", "verl/workers/actor/dp_actor.py", 1486,
     "old_log_prob = log_prob.detach()"),
    ("micro-batch loss for backward", "verl/workers/actor/dp_actor.py", 1829, "loss = policy_loss * loss_scale_factor"),
    ("token-mean over the micro-batch tokens (dp_size 1, batch_num_tokens None in dp_actor)",
     "verl/trainer/ppo/core_algos.py", 1060, "loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size"),
    ("workload of a row", "verl/utils/seqlen_balancing.py", 46, "return 24576 * seqlen_list + seqlen_list**2"),
    ("indices inside a partition are sorted", "verl/utils/seqlen_balancing.py", 249, "sorted_partitions[i] = sorted(partition)"),
    ("number of micro-batches", "verl/utils/seqlen_balancing.py", 387,
     "num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))"),
    ("the same number of micro-batches on each rank (maximum over ranks)", "verl/utils/seqlen_balancing.py", 393,
     "dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)"),
    ("micro-batch partition: unequal-size Karmarkar-Karp", "verl/utils/seqlen_balancing.py", 402,
     "micro_bsz_idx = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)"),
    ("micro-batch order: by workload, descending", "verl/utils/seqlen_balancing.py", 411, "reverse=True,"),
    ("micro-batch order: even places reversed, then odd places", "verl/utils/seqlen_balancing.py", 414,
     "micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]"),
    ("prepare_dynamic_batch default (dp_actor passes no value)", "verl/utils/seqlen_balancing.py", 450,
     "use_dynamic_bsz_balance=True,"),
    # R3 Part A item 4: the trainer reuses the rollout log-probabilities as old log-probabilities
    ("reuse path: the batch has rollout_log_probs", "verl/trainer/ppo/ray_trainer.py", 205,
     'if "rollout_log_probs" not in batch.batch:'),
    ("reuse path: one PPO epoch", "verl/trainer/ppo/ray_trainer.py", 209, "if actor_cfg.ppo_epochs != 1:"),
    ("reuse path: batch rows = ppo_mini_batch_size x rollout.n (A1 768 = 96 x 8, A2 96 = 96 x 1)",
     "verl/trainer/ppo/ray_trainer.py", 213,
     "expected_on_policy_batch_size = actor_cfg.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n"),
    ("reuse path: condition", "verl/trainer/ppo/ray_trainer.py", 214,
     "return effective_batch_size == expected_on_policy_batch_size"),
    ("reuse path: used when bypass_mode is false", "verl/trainer/ppo/ray_trainer.py", 2823,
     "and should_reuse_rollout_log_probs_as_old_log_probs(self.config, batch)"),
    ("reuse path: no old-log-probability forward", "verl/trainer/ppo/ray_trainer.py", 2833,
     "elif reuse_rollout_log_probs:"),
    ("reuse path: old_log_probs = rollout_log_probs", "verl/trainer/ppo/ray_trainer.py", 2834,
     'batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]'),
    ("rollout IS weights from old_log_probs and rollout_log_probs: exp(0) = 1 for each token",
     "verl/trainer/ppo/ray_trainer.py", 2913,
     "batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)"),
    # R3 Part A item 5: the vopd loss and old_log_probs
    ("use_rollout_log_probs default (FSDPActorConfig; the A1 and A2 configs do not set it)",
     "verl/workers/config/actor.py", 620, "use_rollout_log_probs: bool = False"),
    ("use_rollout_log_probs is false: old_log_prob comes from the on_policy branch", "verl/workers/actor/dp_actor.py",
     1482, 'if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:'),
    ("the vopd loss receives old_log_prob (= log_prob.detach(), line 1486)", "verl/workers/actor/dp_actor.py", 1721,
     "old_log_probs=old_log_prob,"),
    ("only use of old_log_probs in the vc loss: the is_clip ratio, detached", "verl/trainer/ppo/core_algos.py", 1625,
     "negative_approx_kl = (student_log_probs - old_log_probs).detach()"),
    ("is_clip ratio = exp(0) = 1 when old_log_probs = log_prob.detach()", "verl/trainer/ppo/core_algos.py", 1627,
     "ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)"),
    ("the vzero branch (PPO surrogate on old_log_probs) needs target_mode vzero", "verl/trainer/ppo/core_algos.py", 1552,
     'if self_distillation_config.get("target_mode", "full_teacher") == "vzero":'),
    ("rollout IS weights multiply the per-token loss (weights 1 in A1 and A2)", "verl/trainer/ppo/core_algos.py", 1632,
     "weighted_per_token_loss = weighted_per_token_loss * rollout_is_weights"),
    ("GRPO fallback rows: the vanilla loss with the same old_log_prob (ratio 1)", "verl/workers/actor/dp_actor.py", 1760,
     'policy_loss_fn = get_policy_loss_fn("vanilla")'),
    # R3 Part A items 3 and 6.7: rollout_corr/kl
    ("rollout_corr/kl input: the rollout log-probabilities of the micro-batch", "verl/workers/actor/dp_actor.py", 1795,
     'rollout_log_prob = model_inputs.get("rollout_log_probs", None)'),
    ("rollout_corr/kl call in update_policy (metric only)", "verl/workers/actor/dp_actor.py", 1801,
     "rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs("),
    ("rollout_corr/kl uses the log_prob of the packed training forward", "verl/trainer/ppo/rollout_corr_helper.py", 1023,
     "old_log_prob=log_prob,"),
    ("rollout_corr/kl = token mean over the micro-batch of rollout_log_prob - log_prob",
     "verl/trainer/ppo/rollout_corr_helper.py", 889,
     'metrics["kl"] = verl_F.masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()'),
]

# R3 Part A item 4: rollout_corr records of the A1 and A2 wandb runs. Source: files/output.log of each run in the vast
# backup (harness-data vast_backup/VLM-VAD-ROLLOUT-001/harness/vad_train/wandb/wandb/), read by the ET script
# runtime/drafts/et/pack001/is_metrics.py on 2026-11-05. b200 has no copy of these logs, so the text is fixed here.
WANDB = """\
| arm | run | steps logged (last step) | output.log sha256 | rollout_is min / max / mean / std | timing_s/old_log_prob entries | rollout_corr/kl range |
|---|---|---|---|---|---|---|
| A1 | run-20261029_072504-6382rhkd | 69 (69) | adcd16d099dcd54d81da90493525d054c8b47c86e5b43077d9c376bfcd1fd0e4 | 1.0 / 1.0 / 1.0 / 0.0 | 0 | 0.0473 .. 0.1382 |
| A1 | run-20261030_014236-vag8862r | 19 (79) | 8857368b6ee13aa18b8a18568456ec0a65db38e66677623888f8cc3eab2eb65d | 1.0 / 1.0 / 1.0 / 0.0 | 0 | 0.0592 .. 0.0900 |
| A1 | run-20261030_061147-i51ii2ty | 60 (130) | 7a6a11a135aed883ae08bd22a0cf2202d02b499f14f4ca34e747b99cd81a2dc4 | 1.0 / 1.0 / 1.0 / 0.0 | 0 | 0.0520 .. 0.0933 |
| A2 | run-20261030_213733-5xtgpa7k | 60 (60) | 4277ceb4a69033216aad58ff3affd2af8691c5256b563e365d954b8dc6858726 | 1.0 / 1.0 / 1.0 / 0.0 | 0 | 0.0304 .. 0.1761 |
| A2 | run-20261031_034214-o68d3gv1 | 40 (100) | 4fd6414b7d712345e8909fe41173d76d54f5a0c27cc1fe9de8a79cfccf1db901 | 1.0 / 1.0 / 1.0 / 0.0 | 0 | 0.0443 .. 0.1000 |
| A2 | run-20261031_074519-991h4fbx | 30 (130) | 9cb102b46741c96a684cd9f6030774eb6484bb913bc98adb3456c033627e869d | 1.0 / 1.0 / 1.0 / 0.0 | 0 | 0.0501 .. 0.1209 |

In each logged step of these runs, every rollout_corr/rollout_is_* value is 1.0 (min, max, mean, seq_min, seq_max,
seq_mean, eff_sample_size) or 0.0 (std, seq_std, seq_max_deviation, fraction_high, fraction_low). No step has a
timing_s/old_log_prob entry. Thus the reuse path ran in each step and the rollout IS weights were 1.
"""

# The numbers of the lines above.
VALUES = {arm: {"ranks": 6, "ulysses": 1, "train_batch_size": 96, "ppo_mini_batch_size": 96, "n": n,
                "balance_batch": True, "use_remove_padding": True, "use_dynamic_bsz": True, "max_token_len": 9216,
                "log_prob_max_token_len": 9216, "ppo_epochs": 1}
          for arm, n in (("A1", 8), ("A2", 1))}


def rank_mini_batch_size(arm: str) -> int:
    """Rows of one rank mini-batch (fsdp_workers.py 244-245)."""
    v = VALUES[arm]
    return v["ppo_mini_batch_size"] * v["n"] // (v["ranks"] // v["ulysses"])


def check(repo: Path) -> list[str]:
    """Rows of the markdown table. Raises SystemExit when a quoted line does not hold its expected text."""
    rows, files = [], {}

    def line_of(rel: str, n: int, want: str) -> str:
        text = files.setdefault(rel, (repo / rel).read_text().splitlines())[n - 1]
        if want not in text:
            raise SystemExit(f"STOP: {rel}:{n} does not hold {want!r}: {text!r}")
        return text.strip()

    for arm, rel in CONFIG.items():
        for name, n, want in CONFIG_LINES:
            w = want[arm] if isinstance(want, dict) else want
            rows.append(f"| {arm} | {name} | `{rel}:{n}` | `{line_of(rel, n, w)}` |")
    for meaning, rel, n, want in CODE_LINES:
        rows.append(f"| A1, A2 | {meaning} | `{rel}:{n}` | `{line_of(rel, n, want)}` |")
    sums = [f"- `{rel}` sha256 {hashlib.sha256((repo / rel).read_bytes()).hexdigest()}" for rel in sorted(files)]
    return rows, sums


def main():
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, sums = check(repo)
    derived = [f"- {arm}: data-parallel ranks {v['ranks'] // v['ulysses']}, rows per step {v['train_batch_size'] * v['n']}, "
               f"rows per rank mini-batch {rank_mini_batch_size(arm)}, micro-batch token limit "
               f"{v['max_token_len'] * v['ulysses']}" for arm, v in VALUES.items()]
    out.write_text("# A1 and A2 trainer values (HANDOFF-VLM-VAD-PACK-001-R3 Definitions 6 and 7)\n\n"
                   "| arm | value | source | line |\n|---|---|---|---|\n" + "\n".join(rows) +
                   "\n\n## Derived values used by split.py\n\n" + "\n".join(derived) +
                   "\n\n## Rollout correction records of the A1 and A2 wandb runs\n\n" + WANDB +
                   "\n## Source files\n\n" + "\n".join(sums) + "\n")
    print(f"trainer values: {len(rows)} quoted lines, all checked")


if __name__ == "__main__":
    main()
