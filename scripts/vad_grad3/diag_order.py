"""Backward-order and fresh-forward part-sum diagnostic of DIRECTIVE-VLM-VAD-GRAD-003-004 (one micro-batch, the
forward of this process: O or Fx). The micro-batch is the one of diag_partsum.pick_micro (TASK-VLM-VAD-GRAD-003-003).

The model inputs are built once. A full target is the O4 loss of the packed micro-batch x B_m / mini. A part target
(masked) is the O4 loss with the response mask of the rows outside the part set to 0, x T_X / T_m x B_m / mini.
  g_ref        the first backward of the full target on a fresh forward (Action 2).
  order.full_x3      one fresh forward, the full-target backward 3 times (retain_graph): b2 and b3 against b1 = g_ref.
  order.C_then_full  one fresh forward (Capture outputs); the masked part-C target on these outputs (Replay), its
                     backward (retain_graph), then the full-target backward: full against g_ref, C against fresh C.
  fresh_parts  each part with its own fresh forward and one backward; the sum of the parts against g_ref, the part loss
               sum against the full loss, and the 5 tensors with the largest share of the squared error.
  fresh_full_vs_ref  a second fresh forward and backward of the full target against g_ref.
  scale3_vs_ref      (ET addition) a fresh forward, the backward of 3 x the full target, divided by 3, against g_ref:
                     the rounding level of one backward (a non-power-of-2 scale changes each bf16 rounding).
Before each backward each parameter .grad must be None or zero (buffer_checks). Relative differences are
||a - b|| / ||b||, fp64 per tensor. The rank factor 1/6 of grads.py is left out (it cancels in each ratio).

Usage: python diag_order.py <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt>
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

import diag_partsum as partsum  # noqa: E402
import grads  # noqa: E402

TOP = 5


def sq_diffs(a, b) -> list[float]:
    return [float((x.double() - y.double()).pow(2).sum()) for x, y in zip(a, b)]


def sq_norm(a) -> float:
    return sum(float(x.double().pow(2).sum()) for x in a)


def rel(a, b) -> float:
    n = sq_norm(b)
    return math.sqrt(sum(sq_diffs(a, b)) / n) if n > 0 else float("nan")


def model_report(model) -> dict:
    """Gradient checkpointing and the hooks of each module and parameter (Action 4)."""
    import torch

    mods, prms = [], []
    for name, m in model.named_modules():
        counts = {k: len(getattr(m, f"_{k}")) for k in
                  ("forward_hooks", "forward_pre_hooks", "backward_hooks", "backward_pre_hooks")
                  if getattr(m, f"_{k}", None)}
        if counts:
            mods.append({"module": name, **counts})
    for name, p in model.named_parameters():
        counts = {k: len(v) for k, v in (("tensor_hooks", p._backward_hooks),
                                         ("post_accumulate_grad_hooks", getattr(p, "_post_accumulate_grad_hooks", None)))
                  if v}
        if counts:
            prms.append({"param": name, **counts})
    gc = [m for _, m in model.named_modules() if getattr(m, "gradient_checkpointing", False)]
    funcs = sorted({repr(getattr(m, "_gradient_checkpointing_func", None)) for m in gc})
    g = torch.nn.modules.module
    return {"module_hooks": mods, "param_hooks": prms, "checkpointing_modules": len(gc), "checkpointing_funcs": funcs,
            "is_gradient_checkpointing": getattr(model, "is_gradient_checkpointing", None),
            "global_module_hooks": {k: len(getattr(g, k, {})) for k in
                                    ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                                     "_global_backward_pre_hooks")}}


def diagnose(ctx, mrows: list[dict], rollouts: dict, names: list[str] | None = None) -> dict:
    import torch

    names = names or [str(i) for i in range(len(ctx.params))]
    mini = {8: 128, 1: 16}[mrows[0]["n"]]
    scale = len(mrows) / mini
    Capture, Replay = ctx.classes
    cap, rep = Capture(ctx.actor, ctx.params), Replay(ctx.actor, ctx.params)
    mi, _ = ctx.inputs([rollouts[r["traj"]] for r in mrows])
    t_m = float(grads.loss_tokens(mi).sum())
    checks, dtypes = [], set()

    def masked(idx):
        mask = mi["response_mask"].clone()
        for i in range(len(mrows)):
            if i not in idx:
                mask[i] = 0
        mmi = {**mi, "response_mask": mask}
        return mmi, float(grads.loss_tokens(mmi).sum()) / t_m

    def backward(tag, target, keep):
        bad = [n for n, p in zip(names, ctx.params) if p.grad is not None and bool(p.grad.any())]
        checks.append({"before": tag, "grad_not_none": sum(p.grad is not None for p in ctx.params), "nonzero": bad[:5]})
        out = torch.autograd.grad(target, ctx.params, retain_graph=keep)
        dtypes.update(str(g.dtype) for g in out)
        return [g.float() for g in out]

    res = {"rows": len(mrows), "micro": list(grads.micro_key(mrows[0])), "batch": mrows[0]["batch"],
           "trajs": [r["traj"] for r in mrows], "labels": [r["label"] for r in mrows],
           "len_bins": [r["len_bin"] for r in mrows], "scale": scale}

    # Action 2.1: one fresh forward, the full-target backward 3 times. b1 is g_ref.
    loss = cap.run(mi)[0]
    res["loss"] = float(loss.detach())
    g_ref = backward("full_x3 b1", loss * scale, True)
    b2 = backward("full_x3 b2", loss * scale, True)
    o = {"b2_vs_b1": rel(b2, g_ref), "b2_equal_b1": all(torch.equal(x, y) for x, y in zip(b2, g_ref))}
    del b2
    b3 = backward("full_x3 b3", loss * scale, False)
    o.update(b3_vs_b1=rel(b3, g_ref), b3_equal_b1=all(torch.equal(x, y) for x, y in zip(b3, g_ref)))
    del b3, loss
    res["ref_sq_norm"] = sq_norm(g_ref)
    res["order"] = {"full_x3": o}

    # Action 2.2: one forward; masked part C on its outputs, then the full target.
    idx_c = grads.members("C", mrows)
    c_ret = None
    if idx_c:
        loss, student, minus, plus = cap.run(mi)
        mmi, frac = masked(idx_c)
        lx = rep.o4(mmi, student, {k: v.detach() for k, v in minus.items() if hasattr(v, "dim")},
                    {k: v.detach() for k, v in plus.items() if hasattr(v, "dim")})
        c_ret = backward("C_then_full C", lx * (frac * scale), True)
        full = backward("C_then_full full", loss * scale, False)
        res["order"]["C_then_full"] = {"full_vs_ref": rel(full, g_ref)}
        del full, loss, student, minus, plus, lx

    # Action 2.3: each part with its own fresh forward and one backward.
    res["fresh_parts"] = {}
    for group in partsum.PARTITIONS:
        acc = [torch.zeros_like(t) for t in g_ref]
        loss_sum, part_sq = 0.0, {}
        for p in group:
            idx = grads.members(p, mrows)
            if not idx:
                continue
            mmi, frac = masked(idx)
            lx = cap.run(mmi)[0]
            loss_sum += float(lx.detach()) * frac
            g = backward(f"fresh {p}", lx * (frac * scale), False)
            part_sq[p] = sq_norm(g) / res["ref_sq_norm"]
            if p == "C" and c_ret is not None:
                res["order"]["C_then_full"]["C_retained_vs_C_fresh"] = rel(c_ret, g)
            for a, x in zip(acc, g):
                a.add_(x)
            del g, lx
        d = sq_diffs(acc, g_ref)
        top = sorted(range(len(d)), key=lambda i: -d[i])[:TOP]
        res["fresh_parts"]["+".join(group)] = {
            "grad_rel_diff": math.sqrt(sum(d) / res["ref_sq_norm"]),
            "loss_sum": loss_sum, "loss_rel_diff": abs(loss_sum - res["loss"]) / max(abs(res["loss"]), 1e-30),
            "part_sq_norm_over_ref": part_sq,
            "top_tensors": [{"name": names[i], "share_of_sq_error": d[i] / sum(d) if sum(d) > 0 else 0.0,
                             "rel_diff": math.sqrt(d[i] / max(float(g_ref[i].double().pow(2).sum()), 1e-300))}
                            for i in top]}
        del acc
    del c_ret

    # Action 2.4: a second fresh forward and backward of the full target.
    loss = cap.run(mi)[0]
    g = backward("fresh full", loss * scale, False)
    res["fresh_full_vs_ref"] = rel(g, g_ref)
    res["fresh_full_equal_ref"] = all(torch.equal(x, y) for x, y in zip(g, g_ref))
    del g, loss

    # ET addition: the rounding level of one backward (3 x the target, divided by 3).
    loss = cap.run(mi)[0]
    g = [x / 3 for x in backward("fresh full x3 scale", loss * (3 * scale), False)]
    res["scale3_vs_ref"] = rel(g, g_ref)
    del g, loss

    res["grad_dtypes"] = sorted(dtypes)
    res["buffer_checks"] = len(checks)
    res["buffers_clean"] = all(not c["nonzero"] for c in checks)
    res["buffer_check_list"] = checks
    return res


def main(argv: list[str]):
    import torch

    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, out, mode, *keys = argv
    grads.setup_paths(repo)
    import ckpt

    if mode not in ("O", "Fx") or (mode == "Fx") != (os.environ.get("VAD_PACKFIX") == "1"):
        raise SystemExit(f"STOP: mode {mode} with VAD_PACKFIX={os.environ.get('VAD_PACKFIX')}")
    t0 = time.time()
    ctx = grads.Context(repo, command_txt, model_dir, g3_params)
    names = ckpt.grad_set(g3_params)
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    log = {"mode": mode, "model_dir": model_dir, "load_seconds": round(time.time() - t0, 1),
           "student": model_report(ctx.actor.actor_module), "teacher": model_report(ctx.actor.teacher_module),
           "param_dtypes": sorted({str(p.dtype) for p in ctx.params}),
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "FLASH_ATTENTION_DETERMINISTIC": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
           "matmul_allow_bf16_reduced_precision_reduction":
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
           "results": {}}
    for k in keys:
        rows = [json.loads(x) for x in open(Path(rows_dir) / f"ROWS_{k}.jsonl")]
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        r = diagnose(ctx, partsum.pick_micro(rows), rollouts, names)
        r.update(seconds=round(time.time() - t, 1), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        log["results"][k] = r
        print(json.dumps({k: {f: r[f] for f in ("micro", "order", "fresh_full_vs_ref", "scale3_vs_ref",
                                                "buffers_clean")}}), flush=True)
        print(json.dumps({k: {g: v["grad_rel_diff"] for g, v in r["fresh_parts"].items()}}), flush=True)
    Path(out).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
