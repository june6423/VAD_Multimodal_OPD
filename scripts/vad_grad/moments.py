"""Moment pass of one checkpoint (HANDOFF-VLM-VAD-GRAD-001-R1 Part A items 5 and 7; DIRECTIVE-VLM-VAD-GRAD-001-002).

For each trajectory: the O4 gradient over the 248 tensors of the A8 gradient set (the A8 code of a5ace7be: model
loading, inputs, teacher views, Objectives.losses), and, when the prompt weight r_i - mean_p r is not 0, the gradient
of the summed response log-probability under the student from the same student forward. Answer-position
probabilities of the ground-truth and the sampled letter under the student and the Base teacher on the x-plus and
x-minus views, at the position of the TASK-056 teacher check. Per prompt: the 8 sibling gradients stay on the GPU
(bf16, as autograd returns them); accum.finish_prompt writes the sibling Gram (raw and with D), adds the group sums to
S|fold|class|flag and the prompt term to A|fold on the host. Prompts run in fold order. A save every SAVE_EVERY
prompts; a restart resumes. At the end: GRAM.json, ANSWER_POS.jsonl, TRAJ.jsonl, and the accumulators are deleted.

Usage: python moments.py <repo root> <A1 COMMAND.txt> <model dir> <rollouts.jsonl> <labels.jsonl> <teacher dir>
         <exp_avg_sq safetensors> <out dir> <checkpoint name> [prompt list json]
Env: SAVE_EVERY (25), STOP_AFTER_PROMPTS (0 = no forced stop; the pilot stops once to test the resume).
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vad_rollout" / "a8"))
import accum  # noqa: E402
import grads as a8  # noqa: E402
import objectives as obj  # noqa: E402
import params as gparams  # noqa: E402
import provenance  # noqa: E402

STOP_EXIT = 75  # exit code of the forced stop


class Capture(obj.Objectives):
    """A8 Objectives that also keeps the output of the student forward (the one with gradients)."""

    last_student = None

    def _forward(self, inputs: dict, grad: bool, **kwargs) -> dict:
        out = super()._forward(inputs, grad, **kwargs)
        if grad:
            self.last_student = out
        return out


def trajectory_step(ctx, mi: dict, weight: float, a_buf: list[torch.Tensor] | None) -> tuple[list[torch.Tensor], dict]:
    """O4 gradient of one trajectory; adds weight x grad log p(y) to a_buf when weight != 0."""
    loss = ctx.objectives.losses(mi, ["O4"])["O4"]
    student = ctx.objectives.last_student
    ctx.objectives.last_student = None
    logp = (student["log_probs"].float() * mi["response_mask"]).sum()
    g = torch.autograd.grad(loss, ctx.params, retain_graph=weight != 0.0)
    if weight != 0.0:
        h = torch.autograd.grad(logp, ctx.params)
        for a, x in zip(a_buf, h):
            a.add_(x.float(), alpha=weight)
        del h
    sq = float(sum(x.float().pow(2).sum() for x in g))
    return list(g), {"o4_loss": float(loss), "logp_sum": float(logp), "sq_norm": sq,
                     "response_tokens": int(mi["response_mask"].sum())}


def answer_probs(ctx, mi: dict, teacher_row: dict) -> dict:
    """Probabilities of the ground-truth and the sampled letter token at the answer position (stage 3b)."""
    pos, gt, sampled = teacher_row["position"], teacher_row["gt_token"], teacher_row["sampled_token"]
    length = mi["responses"].shape[1]
    idx = torch.tensor([gt, sampled], device=mi["responses"].device).view(1, 1, 2).expand(1, length, 2).contiguous()
    out = {"position": pos, "gt_token": gt, "sampled_token": sampled}
    with torch.no_grad():
        views = {"S": ctx.actor._forward_micro_batch(mi, temperature=1.0, topk_indices=idx),
                 "T+": ctx.objectives._forward(obj.view_inputs(mi, "plus"), grad=False, topk_indices=idx),
                 "T-": ctx.objectives._forward(obj.view_inputs(mi, "minus"), grad=False, topk_indices=idx)}
    for name, res in views.items():
        lp = res["topk_logps"][0, pos].float()
        out[f"p_{name}_gt"], out[f"p_{name}_sampled"] = float(lp[0].exp()), float(lp[1].exp())
    return out


def load_d(path: str, names: list[str], device: str) -> tuple[list[torch.Tensor], dict]:
    """D = 1 / (sqrt(v) / sqrt(1 - 0.999^t) + 1e-8) (torch AdamW denominator, DIRECTIVE-002 D6)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        step = int(meta["step"])
        corr = math.sqrt(1.0 - 0.999 ** step)
        d = [1.0 / (f.get_tensor(n).to(device=device, dtype=torch.float32).sqrt() / corr + 1e-8) for n in names]
    return d, {"adam_file": path, "step": step, "beta2": 0.999, "eps": 1e-8}


def peak_rss_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / 1e9
    return float("nan")


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    return float("nan")


def main():
    repo, command_txt, model_dir, rollouts, labels_path, teacher_dir, adam_v, out, name = sys.argv[1:10]
    prompt_list = json.loads(Path(sys.argv[10]).read_text()) if len(sys.argv) > 10 else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_every = int(os.environ.get("SAVE_EVERY", "25"))
    stop_after = int(os.environ.get("STOP_AFTER_PROMPTS", "0"))
    run = {"checkpoint": name, "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "save_every": save_every,
           "stop_after": stop_after, "saves": [], "prompt_seconds": [], "finish_seconds": []}
    if (out / "GRAM.json").exists():
        print(f"{name}: GRAM.json exists, nothing to do")
        return
    run["verl"] = provenance.verl_origin(repo)
    print(f"{name}: verl {run['verl']}")

    t0 = time.time()
    ctx = a8.Context(repo, command_txt, model_dir, rollouts)
    ctx.objectives = Capture(ctx.actor, ctx.params)
    names = [n for n, _ in gparams.grad_params(ctx.actor.actor_module)]
    d, run["metric"] = load_d(adam_v, names, "cuda")
    run["context_seconds"] = time.time() - t0
    labels = {r["traj"]: r for r in map(json.loads, open(labels_path))}
    teacher = {r["traj"]: r for p in sorted(glob.glob(str(Path(teacher_dir) / "teacher_*.jsonl"))) for r in map(json.loads, open(p))}
    by_prompt = defaultdict(list)
    for r in ctx.rows:
        by_prompt[r["prompt_pos"]].append(r)
    missing = [r["traj"] for r in ctx.rows if r["traj"] not in labels or r["traj"] not in teacher]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} trajectories have no label or no teacher record, first {missing[:5]}")
    folds = accum.fold_of(sorted(by_prompt))
    prompts = sorted(by_prompt) if prompt_list is None else sorted(prompt_list)
    order = sorted(prompts, key=lambda p: (folds[p], p))

    store = accum.Store([p.shape for p in ctx.params])
    progress = accum.Progress(out)
    progress.restore(store)
    records = list(progress.state["prompts"])
    todo = [p for p in order if p not in progress.done]
    run["resumed_prompts"] = len(records)
    buf = [torch.zeros(p.shape, dtype=torch.float32, device="cuda") for p in ctx.params]
    a_buf = [torch.zeros(p.shape, dtype=torch.float32, device="cuda") for p in ctx.params]
    torch.cuda.reset_peak_memory_stats()
    for n, p in enumerate(todo):
        start = time.time()
        rows = sorted(by_prompt[p], key=lambda r: r["sibling"])
        labs = [int(labels[r["traj"]]["label"]) for r in rows]
        flags = [teacher[r["traj"]]["flag"] for r in rows]
        weights = accum.a_weights(labs)
        for a in a_buf:
            a.zero_()
        grads, trajs = [], []
        for r, w in zip(rows, weights):
            ts = time.time()
            mi = ctx.model_inputs([r["traj"]])[0]
            g, rec = trajectory_step(ctx, mi, w, a_buf)
            if not math.isfinite(rec["sq_norm"]):
                (out / "STOP.txt").write_text(f"non-finite gradient norm, trajectory {r['traj']}\n")
                raise SystemExit(f"STOP (stop condition 4): non-finite gradient norm, trajectory {r['traj']}")
            tr = teacher[r["traj"]]
            if tr.get("position") is not None and tr.get("gt_token") is not None:
                rec.update(answer_probs(ctx, mi, tr))
            rec.update(traj=r["traj"], label=labs[len(grads)], flag=tr["flag"], label_source=labels[r["traj"]]["source"],
                       weighted=w != 0.0,
                       total_tokens=r["prompt_len"] + len(r["response_ids"]), seconds=round(time.time() - ts, 3))
            grads.append(g)
            trajs.append(rec)
            del mi
        tf = time.time()
        record = accum.finish_prompt(store, p, folds[p], [r["traj"] for r in rows], labs, flags, grads, d,
                                     a_buf if any(weights) else None, buf, extra={"trajectories": trajs})
        del grads
        records.append(record)
        run["finish_seconds"].append(round(time.time() - tf, 3))
        run["prompt_seconds"].append(round(time.time() - start, 3))
        if (n + 1) % save_every == 0:
            ts = time.time()
            folds_saved = sorted(store.dirty)
            progress.save(store, records)
            run["saves"].append({"after_prompts": len(records), "folds": folds_saved, "seconds": round(time.time() - ts, 1),
                                 "bytes": 7 * len(folds_saved) * sum(4 * x.numel() for x in ctx.params),
                                 "mem_available_gb": mem_available_gb(), "peak_rss_gb": peak_rss_gb()})
        if stop_after and n + 1 == stop_after:
            run.update(stopped_after=stop_after, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_rss_gb=peak_rss_gb())
            with open(out / "runs.jsonl", "a") as f:
                f.write(json.dumps(run) + "\n")
            print(f"{name}: forced stop after {stop_after} prompts")
            sys.exit(STOP_EXIT)

    ts = time.time()
    data = accum.gram_file(store, d, "cuda", records, name)
    tmp = out / "GRAM.json.tmp"
    tmp.write_text(json.dumps(data))
    run["gram_seconds"] = round(time.time() - ts, 1)
    run["vectors_allocated"] = len(store.vec)
    with open(out / "ANSWER_POS.jsonl", "w") as fa, open(out / "TRAJ.jsonl", "w") as ft:
        for rec in sorted(records, key=lambda r: r["prompt"]):
            for t in rec["trajectories"]:
                row = {"prompt": rec["prompt"], "fold": rec["fold"], **t}
                ft.write(json.dumps(row) + "\n")
                if "position" in t:
                    fa.write(json.dumps({k: row[k] for k in ("traj", "prompt", "label", "flag", "position", "gt_token",
                                                            "sampled_token") + tuple(k for k in row if k.startswith("p_"))}) + "\n")
    os.replace(tmp, out / "GRAM.json")
    progress.remove_vectors()
    run.update(end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), prompts_done=len(records),
               trajectories=sum(len(r["trajs"]) for r in records), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
               peak_rss_gb=peak_rss_gb(), mem_available_gb_end=mem_available_gb())
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(run) + "\n")
    print(f"{name}: GRAM.json written, {len(records)} prompts")


if __name__ == "__main__":
    main()
