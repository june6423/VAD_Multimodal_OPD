"""fp32 part-sum diagnostic of DIRECTIVE-VLM-VAD-GRAD-003-005 (one micro-batch, the forward of this process: O or Fx).
The micro-batches are the ones of TASK-VLM-VAD-GRAD-003-004 (diag_partsum.pick_micro).

For each key, two phases on the same model inputs (each phase as TASK-004: fresh forward and one backward per vector):
  fp32: the student parameters are the exact fp32 copies of the bf16 parameters. g32 = the full-target gradient.
        The fp32 parts (C+I, L) are summed and compared with g32. scale3_vs_g32 = the rounding level of one fp32 backward
        (3 x the target, divided by 3).
  bf16: the parameters are cast back to bf16 (exact). g_ref and the bf16 part sums are compared with g32 (Action 3).
Settings of the fp32 phase: torch fp32 matmuls without TF32 (the values before the phase are recorded). Flash attention 2
has no fp32 kernel: transformers casts fp32 q, k, v to the dtype of get_target_dtype; this script makes it bf16 (the
transformers choice under bf16 autocast). Thus the attention kernel stays bf16 in the fp32 phase.
Env MAX_ROWS (0 = all rows): keep the first rows of the micro-batch, with both labels (Action 4, a smaller micro-batch).

Usage: python diag_fp32.py <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt>
         <out json> <mode O|Fx> <key>...
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag_order as order  # noqa: E402
import grads  # noqa: E402

partsum = order.partsum


def set_dtype(model, dtype):
    """Cast the parameters (not the fp32 buffers); bf16 -> fp32 -> bf16 is exact."""
    for p in model.parameters():
        if p.dtype != dtype:
            p.data = p.data.to(dtype)


def first_rows(mrows: list[dict], n: int) -> list[dict]:
    """The first n rows; the last of them is replaced by the first row of the missing label."""
    if not n or len(mrows) <= n:
        return mrows
    out = mrows[:n]
    labels = {r["label"] for r in out}
    if labels != {0, 1}:
        other = next((r for r in mrows[n:] if r["label"] not in labels), None)
        if other is not None:
            out = out[:-1] + [other]
    return out


def phase(ctx, mrows, mi, t_m, scale, ref, names, with_scale3: bool) -> dict:
    """Full gradient and fresh-forward part sums; ref None: the full gradient is the reference (returned in 'g')."""
    import torch

    Capture, _ = ctx.classes
    cap = Capture(ctx.actor, ctx.params)

    def grad(m, factor):
        lx = cap.run(m)[0]
        out = torch.autograd.grad(lx * factor, ctx.params)
        return float(lx.detach()), [g.float() for g in out], sorted({str(g.dtype) for g in out})

    loss, g, dtypes = grad(mi, scale)
    res = {"loss": loss, "grad_dtypes": dtypes}
    if ref is None:
        ref = g
        res["g"] = g
        res["ref_sq_norm"] = order.sq_norm(g)
    else:
        res["full_vs_g32"] = order.rel(g, ref)
        del g
    res["parts"] = {}
    for group in partsum.PARTITIONS:
        acc = [torch.zeros_like(t) for t in ref]
        loss_sum = 0.0
        for p in group:
            idx = grads.members(p, mrows)
            if not idx:
                continue
            mask = mi["response_mask"].clone()
            for i in range(len(mrows)):
                if i not in idx:
                    mask[i] = 0
            mmi = {**mi, "response_mask": mask}
            frac = float(grads.loss_tokens(mmi).sum()) / t_m
            lx, gp, _ = grad(mmi, frac * scale)
            loss_sum += lx * frac
            for a, x in zip(acc, gp):
                a.add_(x)
            del gp
        d = order.sq_diffs(acc, ref)
        top = sorted(range(len(d)), key=lambda i: -d[i])[:order.TOP]
        n2 = order.sq_norm(ref)
        res["parts"]["+".join(group)] = {
            "vs_g32": math.sqrt(sum(d) / n2), "loss_rel_diff": abs(loss_sum - loss) / max(abs(loss), 1e-30),
            "top_tensors": [{"name": names[i], "share_of_sq_error": d[i] / sum(d) if sum(d) > 0 else 0.0}
                            for i in top]}
        del acc
    if with_scale3:
        _, g3, _ = grad(mi, 3 * scale)
        res["scale3_vs_g32"] = order.rel([x / 3 for x in g3], ref)
        del g3
    return res


def diagnose(ctx, mrows, rollouts, names, student) -> dict:
    import torch

    mini = {8: 128, 1: 16}[mrows[0]["n"]]
    scale = len(mrows) / mini
    mi, _ = ctx.inputs([rollouts[r["traj"]] for r in mrows])
    t_m = float(grads.loss_tokens(mi).sum())
    res = {"rows": len(mrows), "micro": list(grads.micro_key(mrows[0])), "trajs": [r["traj"] for r in mrows],
           "labels": [r["label"] for r in mrows], "len_bins": [r["len_bin"] for r in mrows], "scale": scale}
    cuda = torch.cuda.is_available()
    set_dtype(student, torch.float32)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    f32 = phase(ctx, mrows, mi, t_m, scale, None, names, True)
    g32 = f32.pop("g")
    res["fp32"] = f32
    if cuda:
        res["fp32_peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    set_dtype(student, torch.bfloat16)
    res["bf16"] = phase(ctx, mrows, mi, t_m, scale, g32, names, False)
    del g32
    return res


def main(argv: list[str]):
    import torch

    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, out, mode, *keys = argv
    grads.setup_paths(repo)
    import ckpt
    import transformers.integrations.flash_attention as fa

    if mode not in ("O", "Fx") or (mode == "Fx") != (os.environ.get("VAD_PACKFIX") == "1"):
        raise SystemExit(f"STOP: mode {mode} with VAD_PACKFIX={os.environ.get('VAD_PACKFIX')}")
    max_rows = int(os.environ.get("MAX_ROWS", "0"))
    t0 = time.time()
    ctx = grads.Context(repo, command_txt, model_dir, g3_params)
    names = ckpt.grad_set(g3_params)
    student = ctx.actor.actor_module
    log = {"mode": mode, "model_dir": model_dir, "load_seconds": round(time.time() - t0, 1), "max_rows": max_rows,
           "before": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                      "float32_matmul_precision": torch.get_float32_matmul_precision()},
           "results": {}}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    fa.get_target_dtype = lambda query, module: torch.bfloat16 if query.dtype == torch.float32 else None
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    for k in keys:
        rows = [json.loads(x) for x in open(Path(rows_dir) / f"ROWS_{k}.jsonl")]
        t = time.time()
        r = diagnose(ctx, first_rows(partsum.pick_micro(rows), max_rows), rollouts, names, student)
        r.update(seconds=round(time.time() - t, 1), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        log["results"][k] = r
        print(json.dumps({k: {"rows": r["rows"], "fp32": {g: v["vs_g32"] for g, v in r["fp32"]["parts"].items()},
                              "scale3_vs_g32": r["fp32"]["scale3_vs_g32"], "bf16_full_vs_g32": r["bf16"]["full_vs_g32"],
                              "bf16": {g: v["vs_g32"] for g, v in r["bf16"]["parts"].items()}}}), flush=True)
    Path(out).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
