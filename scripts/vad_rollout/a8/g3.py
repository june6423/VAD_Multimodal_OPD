"""R9 G3 (and the server parts of G1) on one GPU with 16 step-0 trajectories (HANDOFF-VLM-VAD-ROLLOUT-001-R9).

Usage: python g3.py <repo root> <A1 COMMAND.txt> <rollouts.jsonl of the Base model> <out dir> <generation seconds>
Writes G3.json, G3.txt, G3_PARAMS.txt, G3_HINT_PROMPT.txt in <out dir>.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grads as g  # noqa: E402
import params as gparams  # noqa: E402
import stats  # noqa: E402

R9_HINT = ("Here is a reference solution to this problem:\n{answer}\n\nAfter understanding the reference solution, "
           "please try to solve this problem using your own approach below:")
TRAJECTORIES_PER_CHECKPOINT = 3000
PROMPTS = 375
SIBLINGS = 8
CHECKPOINTS = 3
GPUS = 6
NMAX_GRID = (100, 375)
WITHIN_CLASSES = 4  # fine classes per prompt assumed for the within-pass store term (up to 6 exist)
RESPONSE_CASES = (85, 1024)


class MemWatch(threading.Thread):
    """Lowest MemAvailable of the host (GB, 10^9 bytes) during the run, sampled every 2 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.min_available = float("inf")
        self.stop = False

    def run(self):
        while not self.stop:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    self.min_available = min(self.min_available, int(line.split()[1]) * 1024 / 1e9)
            time.sleep(2)


def vm_hwm_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / 1e9
    return float("nan")


def timed(fn):
    torch.cuda.synchronize()
    start = time.time()
    value = fn()
    torch.cuda.synchronize()
    return value, time.time() - start


def cosine(a, b) -> float:
    return g.gpu_dot(a, b) / math.sqrt(g.gpu_dot(a, a) * g.gpu_dot(b, b))


def makespan(durations: list[float], gpus: int = GPUS) -> float:
    """schedule.py: each free GPU takes the next job of the jobs file, in file order."""
    free = [0.0] * gpus
    for d in durations:
        i = free.index(min(free))
        free[i] += d
    return max(free)


def plan_jobs(n_max: int) -> list[tuple]:
    """The (n, index, trial) jobs of stats.trial_plan for a pool with this largest n, without the trials."""
    sizes = [(n, c) for n, c in stats.TRIAL_SIZES if n <= n_max]
    if n_max >= 1 and n_max not in (n for n, _ in stats.TRIAL_SIZES):
        sizes.append((n_max, stats.LARGEST_N_TRIALS))
    return [(n, i, None) for n, c in sizes for i in range(c)]


def shard_sizes(total: int) -> list[int]:
    return [len(range(s, total, GPUS)) for s in range(GPUS)]


def gradient_hours(c: dict, n_max: int, split: bool, scale: float = 1.0) -> float:
    """Wall time of run_grads.sh for all checkpoints on GPU 0-5 by the schedule.py order (h).

    Each job is one process: one start (model loads and dataset) and one first-call extra per objective.
    scale multiplies the per-trajectory compute times, not the start, store and dot times.
    """
    t = {k: scale * (c["t"][k] + c["build"]) for k in g.OBJECTIVES}
    teacher, shared = scale * c["teacher"], scale * (c["shared"] + c["build"])
    trial_extra = 3 * c["store"] + 6 * c["dot"]
    within_extra = (WITHIN_CLASSES - 1) * c["store"] + WITHIN_CLASSES * (WITHIN_CLASSES - 1) / 2 * c["dot"]
    trajs, prompts = shard_sizes(TRAJECTORIES_PER_CHECKPOINT), shard_sizes(PROMPTS)
    pools = {"pooled": n_max} | ({"f1": n_max // 2, "f0": n_max // 2} if split else {})
    shards = {p: g.shard_jobs(plan_jobs(n), GPUS) for p, n in pools.items()}
    phase1, phase2 = [], []
    for _ in range(CHECKPOINTS):  # the order of run_grads.sh
        for s in range(GPUS):
            phase1.append(c["start"] + trajs[s] * teacher)
            phase2.append(c["start"] + max(c["warm"].values()) + trajs[s] * shared)
            for k in g.OBJECTIVES:
                phase2.append(c["start"] + c["warm"][k] + prompts[s] * (SIBLINGS * t[k] + within_extra))
                for p in pools:
                    phase2.append(c["start"] + c["warm"][k] + sum(4 * n * t[k] + trial_extra for n, _, _ in shards[p][s]))
    return (makespan(phase1) + makespan(phase2)) / 3600


def generation_hours(per_rollout: float, start: float) -> float:
    """Wall time of run_generation.sh: 6 shard jobs per checkpoint on GPU 0-5, one vLLM start per job (h)."""
    return makespan([start + n * per_rollout for _ in range(CHECKPOINTS)
                     for n in shard_sizes(TRAJECTORIES_PER_CHECKPOINT)]) / 3600


def a1_generation(command_txt: str, cfg) -> dict | None:
    """GPU-seconds per rollout and mean response length of A1 from STEP_METRICS.json beside the A1 COMMAND.txt."""
    path = Path(command_txt).parent / "STEP_METRICS.json"
    if not path.exists():
        return None
    steps = json.loads(path.read_text()).values()
    gen = [m["timing_s/gen"] for m in steps if "timing_s/gen" in m]
    resp = [m["response_length/mean"] for m in steps if "response_length/mean" in m]
    rollouts = cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n
    gpus = cfg.trainer.n_gpus_per_node * cfg.trainer.nnodes
    return {"steps": len(gen), "timing_s_gen_mean": sum(gen) / len(gen), "rollouts_per_step": rollouts, "gpus": gpus,
            "gpu_seconds_per_rollout": sum(gen) / len(gen) * gpus / rollouts,
            "response_length_mean": sum(resp) / len(resp)}


def trainer_path_o4(ctx, traj: int) -> dict:
    """O4 of one trajectory alone in its micro-batch through DataParallelPPOActor.update_policy at 2627743a code.

    The optimizer does nothing and _optimizer_step is replaced, so no parameter changes (checked).
    """
    from verl import DataProto

    class NoOpOptimizer:
        def zero_grad(self, *args, **kwargs):
            pass

        def step(self, *args, **kwargs):
            pass

    mi = ctx.model_inputs([traj])[0]
    tensors = {k: v for k, v in mi.items() if isinstance(v, torch.Tensor)}
    non_tensors = {k: v for k, v in mi.items() if not isinstance(v, torch.Tensor) and k != "pad_token_id"}
    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors,
                               meta_info={"temperature": 1.0, "pad_token_id": ctx.tokenizer.pad_token_id})
    probe = [p.detach().clone() for p in ctx.params[:3]]
    total_before = sum(float(p.detach().double().sum()) for p in ctx.params)
    ctx.actor.actor_optimizer = NoOpOptimizer()
    ctx.actor._optimizer_step = lambda: torch.tensor(0.0, device="cuda")
    metrics = ctx.actor.update_policy(data)
    for p in ctx.actor.actor_module.parameters():
        p.grad = None
    unchanged = all(torch.equal(a, b) for a, b in zip(probe, ctx.params[:3])) and \
        sum(float(p.detach().double().sum()) for p in ctx.params) == total_before
    value = metrics["self_distillation/weighted_jsd_token_mean"]
    value = sum(value) / len(value) if isinstance(value, list) else float(value)
    a8 = float(ctx.objectives.losses(mi, ["O4"])["O4"])
    return {"trainer_token_mean": value, "a8_o4": a8, "relative_difference": abs(a8 - value) / abs(value),
            "parameters_unchanged": unchanged, "actor_vopd_loss_scaled": metrics.get("actor/vopd_loss")}


def main():
    repo, command_txt, rollouts, out, gen_seconds = sys.argv[1:6]
    out = Path(out)
    watch = MemWatch()
    watch.start()
    cfg_path = g.trainer_io.compose_a1_config(repo, command_txt).actor_rollout_ref.model.path
    ctx, context_seconds = timed(lambda: g.Context(repo, command_txt, cfg_path, rollouts))
    report = {"model": cfg_path, "generation_seconds_16": float(gen_seconds)}

    # G1: the gradient set.
    selected = gparams.grad_params(ctx.actor.actor_module)
    lines = [f"{name} {tuple(p.shape)} {p.numel()}" for name, p in selected]
    total = sum(p.numel() for _, p in selected)
    excluded = [n for n, p in ctx.actor.actor_module.named_parameters()
                if p.dim() == 2 and gparams.DECODER_MARKER in n and not gparams.is_grad_param(n, p)]
    lines += [f"selected tensors {len(selected)} parameters {total}", f"excluded 2-D decoder-layer tensors {excluded}"]
    (out / "G3_PARAMS.txt").write_text("\n".join(lines) + "\n")
    report["gradient_set"] = {"tensors": len(selected), "parameters": total, "fp32_gb": total * 4 / 1e9}

    # G1: the rendered O1 teacher prompt.
    rows = ctx.rows
    mi0 = ctx.model_inputs([rows[0]["traj"]])[0]
    start = int(mi0["hint_response_start_idx"][0])
    ids = mi0["hint_input_ids"][0][:start].tolist()
    rendered = ctx.tokenizer.decode(ids, skip_special_tokens=False)
    (out / "G3_HINT_PROMPT.txt").write_text(rendered)
    body = R9_HINT.format(answer=rows[0]["gt"])
    report["hint"] = {"body_present": body in rendered, "leading_blank_lines": ("\n\n" + body) in rendered,
                      "gt": rows[0]["gt"]}

    # G3: per-trajectory gradients, one objective at a time and all four with shared forwards.
    per = {k: [] for k in g.OBJECTIVES}
    shared, build, norms = [], [], {k: [] for k in g.OBJECTIVES}
    torch.cuda.reset_peak_memory_stats()
    for r in rows:
        mi, t = timed(lambda: ctx.model_inputs([r["traj"]])[0])
        build.append(t)
        for k in g.OBJECTIVES:
            res, t = timed(lambda: ctx.objectives.gradients(mi, [k])[k])
            per[k].append(t)
            norms[k].append(g.norm_of(res[1]))
            del res
        res, t = timed(lambda: ctx.objectives.gradients(mi, list(g.OBJECTIVES)))
        shared.append(t)
        del res
    report["seconds"] = {"context_build": context_seconds, "build_inputs_mean": sum(build) / len(build),
                         **{f"{k}_mean": sum(v) / len(v) for k, v in per.items()},
                         **{f"{k}_max": max(v) for k, v in per.items()},
                         **{f"{k}_without_slowest": (sum(v) - max(v)) / (len(v) - 1) for k, v in per.items()},
                         "all4_shared_mean": sum(shared) / len(shared)}
    report["norms_finite"] = all(math.isfinite(x) for v in norms.values() for x in v)
    report["norms"] = {k: {"min": min(v), "max": max(v)} for k, v in norms.items()}

    # G3: repeat cosine, and shared against separate forwards.
    rep = {}
    for k in g.OBJECTIVES:
        a = ctx.objectives.gradients(mi0, [k])[k][1]
        b = ctx.objectives.gradients(mi0, [k])[k][1]
        rep[k] = cosine(a, b)
        del a, b
    both = ctx.objectives.gradients(mi0, list(g.OBJECTIVES))
    sep = {k: ctx.objectives.gradients(mi0, [k])[k][1] for k in ("O2", "O4")}
    report["repeat_cosine"] = rep
    report["shared_vs_separate_cosine"] = {k: cosine(both[k][1], sep[k]) for k in sep}
    del both, sep

    # Teacher check on the 16 trajectories.
    _, report["seconds"]["teacher_check_16"] = timed(lambda: g.run_teacher(ctx, out, 0, 1))
    report["teacher_flags"] = stats.summarize_teacher(
        [{"correct": t["correct"], "flag": t["flag"], "gap": t["gap"]} for t in g.load_teacher(out).values()])

    # Transfer cost of one host-stored Gram vector. The first store allocates the pinned vector; after release the
    # pool gives it back, as in the FULL passes. Then one GPU dot against a stored vector.
    vec = [torch.randn(p.shape, device="cuda") for p in ctx.params]
    gram = g.Gram(ctx.pool)
    stores = []
    for i in range(3):
        _, t = timed(lambda: gram.add(f"s{i}", vec))
        stores.append(t)
        gram.release()
    gram.add("a", vec)
    _, report["seconds"]["host_dot"] = timed(lambda: gram.add("b", vec, keep=False))  # one dot with the stored vector
    gram.release()
    report["seconds"]["host_store_first"] = stores[0]
    report["seconds"]["host_store_reuse"] = stores[1:]
    del vec

    # G1: O4 against the trainer path.
    report["o4_trainer_path"] = trainer_path_o4(ctx, rows[0]["traj"])
    report["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    report["peak_rss_gb"] = vm_hwm_gb()
    watch.stop = True
    report["min_mem_available_gb"] = watch.min_available

    # Projection of Part B items 3 and 4 on GPU 0-5: the jobs of run_generation.sh and run_grads.sh in schedule.py
    # order. Per-trajectory times without the slowest call; each job pays the slowest-call extra once.
    s = report["seconds"]
    c = {"start": s["context_build"], "build": s["build_inputs_mean"], "shared": s["all4_shared_mean"],
         "teacher": s["teacher_check_16"] / len(rows), "dot": s["host_dot"],
         "store": sum(s["host_store_reuse"]) / len(s["host_store_reuse"]),
         "t": {k: s[f"{k}_without_slowest"] for k in g.OBJECTIVES},
         "warm": {k: s[f"{k}_max"] - s[f"{k}_without_slowest"] for k in g.OBJECTIVES}}
    grads_h = {mode: {str(n): gradient_hours(c, n, mode == "split") for n in NMAX_GRID} for mode in ("nosplit", "split")}
    a1 = a1_generation(command_txt, ctx.cfg)
    gen_h = {"upper_bound": generation_hours(float(gen_seconds) / len(rows), 0.0)}
    if a1:
        gen_h["a1_rate"] = generation_hours(a1["gpu_seconds_per_rollout"], float(gen_seconds))
    prompt_len = sum(r["prompt_len"] for r in rows) / len(rows)
    response_len = sum(len(r["response_ids"]) for r in rows) / len(rows)
    cases = {str(n): n for n in RESPONSE_CASES} | ({"a1_mean": a1["response_length_mean"]} if a1 else {})
    scaling = {}
    for name, length in cases.items():
        factor = (prompt_len + length) / (prompt_len + response_len)
        scaling[name] = {"response_tokens": length, "factor": factor,
                         "gradients_h": {mode: {str(n): gradient_hours(c, n, mode == "split", factor) for n in NMAX_GRID}
                                         for mode in ("nosplit", "split")}}
    report["projection"] = {
        "inputs": c, "a1_generation": a1, "generation_h": gen_h, "gradients_h": grads_h,
        "items_3_4_h": {f"{mode}|{n}": {k: v + grads_h[mode][n] for k, v in gen_h.items()}
                        for mode in grads_h for n in grads_h[mode]},
        "response_scaling": {"prompt_tokens_mean": prompt_len, "response_tokens_mean_g3": response_len, "cases": scaling},
        "assumptions": [f"{WITHIN_CLASSES} fine classes per prompt in the within pass",
                        "flag pools f1 and f0 have the largest n // 2 each",
                        "the a1_rate generation uses the G3 generate time as the vLLM start of each job",
                        "response scaling multiplies the per-trajectory times by (prompt + response) tokens"]}
    (out / "G3.json").write_text(json.dumps(report, indent=1))
    (out / "G3.txt").write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in report.items()) + "\n")
    print((out / "G3.txt").read_text())


if __name__ == "__main__":
    main()
