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
GPUS = 6
NMAX_GRID = (100, 200, 300, 375)


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
    ctx = g.Context(repo, command_txt, cfg_path, rollouts)
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
    report["seconds"] = {"build_inputs_mean": sum(build) / len(build),
                         **{f"{k}_mean": sum(v) / len(v) for k, v in per.items()},
                         **{f"{k}_max": max(v) for k, v in per.items()},
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
    g.run_teacher(ctx, out, 0, 1)
    report["teacher_flags"] = stats.summarize_teacher(
        [{"correct": t["correct"], "flag": t["flag"], "gap": t["gap"]} for t in g.load_teacher(out).values()])

    # Transfer cost of one host-stored Gram vector (copy to pinned memory, then one GPU dot against it).
    vec = [torch.randn(p.shape, device="cuda") for p in ctx.params]
    gram = g.Gram(ctx.pool)
    _, report["seconds"]["host_store"] = timed(lambda: gram.add("a", vec))
    _, report["seconds"]["host_dot"] = timed(lambda: gram.add("b", vec, keep=False))  # one dot with the stored vector
    gram.release()
    del vec

    # G1: O4 against the trainer path.
    report["o4_trainer_path"] = trainer_path_o4(ctx, rows[0]["traj"])
    report["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    report["peak_rss_gb"] = vm_hwm_gb()
    watch.stop = True
    report["min_mem_available_gb"] = watch.min_available

    # Projection of Part B items 3 and 4 on GPU 0-5 (one worker per GPU).
    s = report["seconds"]
    t_obj = {k: s[f"{k}_mean"] + s["build_inputs_mean"] for k in g.OBJECTIVES}
    trial_overhead = 3 * s["host_store"] + 6 * s["host_dot"]
    proj = {}
    for n_max in NMAX_GRID:
        def trial_seconds(nmax):
            slots = 64 * 16 + 8 * 64 + 2 * 4 * nmax
            return sum(slots * t for t in t_obj.values()) + (64 + 8 + 2) * len(t_obj) * trial_overhead
        within = TRAJECTORIES_PER_CHECKPOINT * sum(t_obj.values())
        cross = TRAJECTORIES_PER_CHECKPOINT * (s["all4_shared_mean"] + s["build_inputs_mean"])
        teacher = TRAJECTORIES_PER_CHECKPOINT * s["build_inputs_mean"] * 1.2
        pooled = 3 * (within + trial_seconds(n_max) + cross + teacher) / GPUS / 3600
        split = 3 * 2 * trial_seconds(n_max // 2) / GPUS / 3600
        proj[n_max] = {"gradients_h_without_split": pooled, "gradients_h_with_split": pooled + split}
    gen_h = 3 * (float(gen_seconds) / 16) * TRAJECTORIES_PER_CHECKPOINT / GPUS / 3600
    report["projection"] = {"generation_h_upper": gen_h, "by_largest_n": proj,
                            "note": "generation from 16 rollouts at batch 16 (upper bound); flag pools assume n_max/2"}
    (out / "G3.json").write_text(json.dumps(report, indent=1))
    (out / "G3.txt").write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in report.items()) + "\n")
    print((out / "G3.txt").read_text())


if __name__ == "__main__":
    main()
