"""A1 and A2 trainer values of Definition 6 (HANDOFF-VLM-VAD-PACK-001-R1, gate G1), each with its source file and line.

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
]

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
    out.write_text("# A1 and A2 trainer values (HANDOFF-VLM-VAD-PACK-001-R1 Definition 6)\n\n"
                   "| arm | value | source | line |\n|---|---|---|---|\n" + "\n".join(rows) +
                   "\n\n## Derived values used by split.py\n\n" + "\n".join(derived) +
                   "\n\n## Source files\n\n" + "\n".join(sums) + "\n")
    print(f"trainer values: {len(rows)} quoted lines, all checked")


if __name__ == "__main__":
    main()
