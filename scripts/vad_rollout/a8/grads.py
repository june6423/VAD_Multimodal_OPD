"""A8 gradient passes on one checkpoint (HANDOFF-VLM-VAD-ROLLOUT-001-R9, Part A items 5 to 10, G5).

One process uses one GPU. No parameter changes: gradients come from torch.autograd.grad and no optimizer exists.
Sums are fp32 on the GPU. Earlier sums of a Gram matrix wait in host memory. No per-trajectory gradient is stored.

Usage: python grads.py <repo root> <A1 COMMAND.txt> <model dir> <rollouts.jsonl> <out dir> <pass> [options]
  teacher                      teacher check of every trajectory -> teacher.jsonl
  within <O> <shard> <shards>  within-prompt unit-vector sums and norms -> within_<O>_<shard>.jsonl
  trials <O> <pool> <shard> <shards>  across-prompt trials (pool: pooled, f1, f0) -> trials_<O>_<pool>_<shard>.jsonl
  cross <shard> <shards>       cross-objective cosines -> cross_<shard>.jsonl
  summarize                    all files of <out dir> -> STATS.json
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import objectives as obj  # noqa: E402
import params as gparams  # noqa: E402
import stats  # noqa: E402
import trainer_io  # noqa: E402

OBJECTIVES = ("O1", "O2", "O3", "O4")
TRIAL_SEED = 42


def gpu_dot(a, b) -> float:
    return float(sum(torch.dot(x.reshape(-1).double(), y.reshape(-1).to(x.device).double()) for x, y in zip(a, b)))


class HostPool:
    """Reusable pinned host vectors with the shapes of the gradient set."""

    def __init__(self, shapes):
        self.shapes, self.free = shapes, []

    def take(self, src):
        vec = self.free.pop() if self.free else [torch.empty(s, dtype=torch.float32, pin_memory=True) for s in self.shapes]
        for h, g in zip(vec, src):
            h.copy_(g)
        return vec

    def give(self, vec):
        self.free.append(vec)


class Gram:
    """Squared norms and inner products of named vectors. Each new vector is on the GPU; earlier ones are on the host."""

    def __init__(self, pool: HostPool):
        self.pool, self.host, self.values = pool, {}, {}

    def add(self, name: str, vec, keep: bool = True):
        self.values[f"{name}|{name}"] = gpu_dot(vec, vec)
        for other, hv in self.host.items():
            a, b = sorted((other, name))
            self.values[f"{a}|{b}"] = gpu_dot(vec, hv)
        if keep:
            self.host[name] = self.pool.take(vec)

    def release(self):
        for vec in self.host.values():
            self.pool.give(vec)
        self.host = {}


class Context:
    def __init__(self, repo, command_txt, model_dir, rollouts_path):
        self.cfg = trainer_io.compose_a1_config(repo, command_txt)
        self.tokenizer, self.processor = trainer_io.tokenizer_processor(self.cfg)
        self.dataset = trainer_io.load_dataset(self.cfg, self.tokenizer, self.processor)
        self.rows = [json.loads(line) for line in open(rollouts_path)]
        self.by_traj = {r["traj"]: r for r in self.rows}
        self._prompts = {}
        trainer_io.init_single_rank()
        student = trainer_io.load_model(model_dir, self.cfg, train=True)
        teacher = trainer_io.load_model(self.cfg.actor_rollout_ref.model.path, self.cfg, train=False)
        self.actor = trainer_io.make_actor(self.cfg, student, teacher)
        self.params = [p for _, p in gparams.grad_params(student)]
        self.objectives = obj.Objectives(self.actor, self.params)
        self.pool = HostPool([p.shape for p in self.params])

    def prompt(self, index: int) -> dict:
        # Keep only the last prompt: the images are large, and siblings of one prompt come together.
        if index not in self._prompts:
            self._prompts = {index: trainer_io.prompt_inputs(self.dataset, index, self.processor, self.cfg)}
        return self._prompts[index]

    def model_inputs(self, trajs: list[int]) -> list[dict]:
        """One-row model inputs for each trajectory, with the student, x-plus, x-minus and hint views."""
        items = []
        for t in trajs:
            r = self.by_traj[t]
            item = self.prompt(r["dataset_index"])
            if hashlib.sha256(json.dumps(item["prompt_ids"]).encode()).hexdigest() != r["prompt_sha256"]:
                raise RuntimeError(f"prompt of trajectory {t} differs from the rollout prompt")
            items.append({**item, "response_ids": r["response_ids"], "uid": str(r["prompt_pos"])})
        batch = trainer_io.assemble_batch(items, self.tokenizer, self.processor, self.cfg)
        batch = batch.union(trainer_io.teacher_views(batch, self.cfg, self.tokenizer, self.processor))
        batch = batch.union(trainer_io.hint_views(batch, self.cfg, self.tokenizer, self.processor))
        batch = trainer_io.to_device(batch, "cuda")
        out = []
        for i in range(len(trajs)):
            sub = batch[[i]]
            out.append({**sub.batch, **sub.non_tensor_batch, "pad_token_id": self.tokenizer.pad_token_id})
        return out


def norm_of(vec) -> float:
    return math.sqrt(gpu_dot(vec, vec))


def check_finite(value: float, what: str):
    if not math.isfinite(value):
        raise SystemExit(f"STOP R9 stop condition 3: non-finite gradient norm ({what})")


def run_teacher(ctx: Context, out: Path):
    with open(out / "teacher.jsonl", "w") as f:
        for r in ctx.rows:
            rec = {"traj": r["traj"], "correct": r["correct"], "letter": r["letter"], "gt": r["gt"], "flag": "fx", "gap": None}
            found = obj.answer_letter_char(r["response_text"])
            pos = obj.letter_token(ctx.tokenizer, r["response_ids"], found[0]) if found else None
            if pos is not None:
                sampled = r["response_ids"][pos]
                gt_tok = obj.gt_token_like(ctx.tokenizer, sampled, found[1], r["gt"])
                if gt_tok is not None:
                    mi = ctx.model_inputs([r["traj"]])[0]
                    rec.update(ctx.objectives.teacher_check(mi, pos, gt_tok, sampled), position=pos, gt_token=gt_tok,
                               sampled_token=sampled)
            f.write(json.dumps(rec) + "\n")


def load_flags(out: Path) -> dict[int, str]:
    return {r["traj"]: r["flag"] for r in map(json.loads, open(out / "teacher.jsonl"))}


def run_within(ctx: Context, out: Path, name: str, shard: int, shards: int):
    flags = load_flags(out)
    by_prompt = defaultdict(list)
    for r in ctx.rows:
        by_prompt[r["prompt_pos"]].append(r)
    with open(out / f"within_{name}_{shard}.jsonl", "w") as f:
        for pos in sorted(by_prompt)[shard::shards]:
            rows = by_prompt[pos]
            classes = defaultdict(list)
            for r in rows:
                classes[("C_" if r["correct"] else "I_") + flags[r["traj"]]].append(r["traj"])
            gram, counts, norms, losses = Gram(ctx.pool), {}, {}, {}
            keys = sorted(classes)
            for k_i, key in enumerate(keys):
                acc = [torch.zeros(p.shape, dtype=torch.float32, device="cuda") for p in ctx.params]
                for t, mi in zip(classes[key], ctx.model_inputs(classes[key])):
                    loss, grads = ctx.objectives.gradients(mi, [name])[name]
                    n = norm_of(grads)
                    check_finite(n, f"{name} trajectory {t}")
                    norms[t], losses[t] = n, loss
                    for a, g in zip(acc, grads):
                        a.add_(g.float(), alpha=1.0 / n)
                    del grads
                counts[key] = len(classes[key])
                gram.add(key, acc, keep=k_i < len(keys) - 1)
                del acc
            gram.release()
            f.write(json.dumps({"prompt": pos, "counts": counts, "gram": gram.values,
                                "norms": {str(t): v for t, v in norms.items()},
                                "losses": {str(t): v for t, v in losses.items()}}) + "\n")
            f.flush()


def pool_rows(ctx: Context, flags: dict[int, str], which: str) -> list[dict]:
    return [{"id": r["traj"], "prompt": r["prompt_pos"], "correct": r["correct"]} for r in ctx.rows
            if which == "pooled" or flags[r["traj"]] == which]


def run_trials(ctx: Context, out: Path, name: str, which: str, shard: int, shards: int):
    flags = load_flags(out)
    plan = stats.trial_plan(pool_rows(ctx, flags, which), TRIAL_SEED)
    jobs = [(e["n"], i, trial) for e in plan if e["trials"] for i, trial in enumerate(e["trials"])]
    with open(out / f"trials_{name}_{which}_{shard}.jsonl", "w") as f:
        f.write(json.dumps({"plan": [{k: e[k] for k in ("n", "count", "largest_n")} | {"possible": e["trials"] is not None}
                                     for e in plan]}) + "\n")
        for n, i, trial in jobs[shard::shards]:
            start = time.time()
            gram = Gram(ctx.pool)
            for g_i, group in enumerate(stats.GROUPS):
                acc = [torch.zeros(p.shape, dtype=torch.float32, device="cuda") for p in ctx.params]
                for t, mi in zip(trial[group], ctx.model_inputs(trial[group])):
                    _, grads = ctx.objectives.gradients(mi, [name])[name]
                    check_finite(norm_of(grads), f"{name} trajectory {t}")
                    for a, g in zip(acc, grads):
                        a.add_(g.float())
                    del grads
                gram.add(group, acc, keep=g_i < len(stats.GROUPS) - 1)
                del acc
            gram.release()
            v = gram.values
            rec = {"n": n, "trial": i, "seconds": round(time.time() - start, 1)}
            for g in stats.GROUPS:
                rec[f"sq_{g}"] = v[f"{g}|{g}"]
            for a, b in (("C1", "C2"), ("I1", "I2")) + stats.CROSS_PAIRS:
                rec[f"dot_{a}_{b}"] = v["|".join(sorted((a, b)))]
            f.write(json.dumps(rec) + "\n")
            f.flush()


def run_cross(ctx: Context, out: Path, shard: int, shards: int):
    with open(out / f"cross_{shard}.jsonl", "w") as f:
        for r in ctx.rows[shard::shards]:
            mi = ctx.model_inputs([r["traj"]])[0]
            grads = {k: g for k, (_, g) in ctx.objectives.gradients(mi, list(OBJECTIVES)).items()}
            norms = {k: norm_of(g) for k, g in grads.items()}
            for k, n in norms.items():
                check_finite(n, f"{k} trajectory {r['traj']}")
            cos = {f"{a}|{b}": gpu_dot(grads[a], grads[b]) / (norms[a] * norms[b])
                   for i, a in enumerate(OBJECTIVES) for b in OBJECTIVES[i + 1:]}
            f.write(json.dumps({"traj": r["traj"], "norms": norms, "cos": cos}) + "\n")
            f.flush()
            del grads


def summarize(out: Path, rollouts_path: str):
    rows = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    teacher = {r["traj"]: r for r in map(json.loads, open(out / "teacher.jsonl"))}
    by_prompt = defaultdict(list)
    for r in rows.values():
        by_prompt[r["prompt_pos"]].append(r["correct"])
    pclass = {p: stats.prompt_class(sum(v), len(v) - sum(v)) for p, v in by_prompt.items()}
    result = {"trajectories": len(rows), "correct": sum(r["correct"] for r in rows.values()),
              "prompt_classes": {k: sum(1 for v in pclass.values() if v == k) for k in ("mixed", "all_correct", "all_wrong", "other")},
              "fx": {"count": sum(1 for t in teacher.values() if t["flag"] == "fx"),
                     "correct": sum(1 for t in teacher.values() if t["flag"] == "fx" and t["correct"])},
              "teacher": stats.summarize_teacher([{"correct": t["correct"], "flag": t["flag"], "gap": t["gap"]} for t in teacher.values()]),
              "objectives": {}}
    for name in OBJECTIVES:
        records, norms = [], {}
        for path in sorted(glob.glob(str(out / f"within_{name}_*.jsonl"))):
            for rec in map(json.loads, open(path)):
                records.append({"counts": rec["counts"], "gram": rec["gram"]})
                norms.update({int(t): v for t, v in rec["norms"].items()})
        if not records:
            continue
        entry = {"within": stats.summarize_prompts(records), "trials": {}}
        for which in ("pooled",) + stats.FLAGS:
            trials, plan = [], None
            for path in sorted(glob.glob(str(out / f"trials_{name}_{which}_*.jsonl"))):
                lines = list(map(json.loads, open(path)))
                plan = lines[0]["plan"]
                trials += lines[1:]
            if not trials:
                continue
            members = [t for t in rows if which == "pooled" or teacher[t]["flag"] == which]
            pool_norms = {"C": [norms[t] for t in members if rows[t]["correct"] and t in norms],
                          "I": [norms[t] for t in members if not rows[t]["correct"] and t in norms]}
            p = len(pool_norms["C"]) / max(1, len(pool_norms["C"]) + len(pool_norms["I"]))
            entry["trials"][which] = {"plan": plan, **stats.summarize_trials(trials, p, pool_norms)}
        if "pooled" in entry["trials"]:
            entry["reading_rule"] = stats.reading_rule_checks(entry["trials"]["pooled"], entry["within"])
        result["objectives"][name] = entry
    cross = []
    for path in sorted(glob.glob(str(out / "cross_*.jsonl"))):
        for rec in map(json.loads, open(path)):
            r = rows[rec["traj"]]
            cross.append({"correct": r["correct"], "prompt_class": pclass[r["prompt_pos"]],
                          "flag": teacher[rec["traj"]]["flag"], "cos": rec["cos"]})
    if cross:
        result["cross"] = stats.summarize_cross(cross)
    (out / "STATS.json").write_text(json.dumps(result, indent=1))


def main():
    repo, command_txt, model_dir, rollouts, out, which = sys.argv[1:7]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    args = sys.argv[7:]
    if which == "summarize":
        summarize(out, rollouts)
        return
    ctx = Context(repo, command_txt, model_dir, rollouts)
    if which == "teacher":
        run_teacher(ctx, out)
    elif which == "within":
        run_within(ctx, out, args[0], int(args[1]), int(args[2]))
    elif which == "trials":
        run_trials(ctx, out, args[0], args[1], int(args[2]), int(args[3]))
    elif which == "cross":
        run_cross(ctx, out, int(args[0]), int(args[1]))
    else:
        raise SystemExit(f"unknown pass {which}")


if __name__ == "__main__":
    main()
