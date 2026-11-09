"""Part-sum diagnostic of DIRECTIVE-VLM-VAD-GRAD-003-003 (one micro-batch, the forward of this process: O or Fx).

For one micro-batch: the O4 loss of one packed forward (Capture) and the total gradient (loss x B_m / mini). Then, for
each partition (C/I and L1/L2/L3):
  replay: the current grads.run_batch method (grads.py 229-253): per-row O4 losses on row slices of the forward
    outputs (Replay), part target = sum_{i in X} l_i (T_i / T_m) (B_m / mini).
  masked: Definition 9.1: the O4 loss of the same packed forward outputs with the response mask of the rows outside X
    set to 0, times T_X / T_m (token mean) and B_m / mini, retain_graph.
Values: loss scalars and their sums, rows in no class and in more than one class, ||sum of parts - total|| / ||total||
(fp64 per tensor). The rank factor 1/6 of grads.py is left out (it cancels in each ratio).

Usage: python diag_partsum.py <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt>
         <out json> <mode O|Fx> <key>...   (one micro-batch with >= 3 rows of each key)
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

import grads  # noqa: E402

PARTITIONS = (("C", "I"), ("L1", "L2", "L3"))
MIN_ROWS = 3


def pick_micro(rows: list[dict]) -> list[dict]:
    """The first micro-batch (row file order) with >= 3 rows and both labels; else the first with >= 3 rows."""
    micro = {}
    for r in rows:
        micro.setdefault(grads.micro_key(r), []).append(r)
    big = [sorted(m, key=lambda r: r["position"]) for m in micro.values() if len(m) >= MIN_ROWS]
    if not big:
        raise SystemExit(f"STOP: no micro-batch with {MIN_ROWS} or more rows")
    both = [m for m in big if {r["label"] for r in m} >= {0, 1}]
    return (both or big)[0]


def coverage(mrows: list[dict]) -> dict:
    out = {}
    for group in PARTITIONS:
        n_in = [sum(i in grads.members(p, mrows) for p in group) for i in range(len(mrows))]
        out["+".join(group)] = {"rows_in_no_class": n_in.count(0), "rows_in_more_than_one": sum(c > 1 for c in n_in),
                                "rows_per_part": {p: len(grads.members(p, mrows)) for p in group}}
    return out


def rel_diff(acc, total) -> float:
    d2 = sum(float((a.double() - t.double()).pow(2).sum()) for a, t in zip(acc, total))
    t2 = sum(float(t.double().pow(2).sum()) for t in total)
    return math.sqrt(d2 / t2) if t2 > 0 else float("nan")


def diagnose(ctx, mrows: list[dict], rollouts: dict) -> dict:
    import torch

    mini = {8: 128, 1: 16}[mrows[0]["n"]]
    scale = len(mrows) / mini
    Capture, Replay = ctx.classes
    cap, rep = Capture(ctx.actor, ctx.params), Replay(ctx.actor, ctx.params)
    mi, rows_mi = ctx.inputs([rollouts[r["traj"]] for r in mrows])
    t_rows = grads.loss_tokens(mi)
    t_m = float(t_rows.sum())
    loss, student, minus, plus = cap.run(mi)
    minus_d = {k: v.detach() for k, v in minus.items() if hasattr(v, "dim")}
    plus_d = {k: v.detach() for k, v in plus.items() if hasattr(v, "dim")}
    total = [g.float() for g in torch.autograd.grad(loss * scale, ctx.params, retain_graph=True)]
    w = [float(t_rows[i]) / t_m if t_m > 0 else 0.0 for i in range(len(mrows))]
    l_rows = [rep.o4(rows_mi[i], grads.row_slice(student, i), grads.row_slice(minus_d, i), grads.row_slice(plus_d, i))
              for i in range(len(mrows))]
    res = {"rows": len(mrows), "micro": list(grads.micro_key(mrows[0])), "batch": mrows[0]["batch"],
           "trajs": [r["traj"] for r in mrows], "prompts": [r.get("prompt") for r in mrows],
           "labels": [r["label"] for r in mrows], "len_bins": [r["len_bin"] for r in mrows],
           "loss_tokens": [float(x) for x in t_rows], "scale": scale, "loss": float(loss.detach()),
           "replay_loss_sum": sum(wi * float(li.detach()) for wi, li in zip(w, l_rows)),
           "coverage": coverage(mrows), "partitions": {}}
    jobs = [(m, g) for g in PARTITIONS for m in ("replay", "masked")]
    for j, (method, group) in enumerate(jobs):
        acc = [torch.zeros_like(t) for t in total]
        loss_sum = 0.0
        for p in group:
            idx = grads.members(p, mrows)
            if not idx:
                continue
            if method == "replay":
                target = sum(l_rows[i] * (w[i] * scale) for i in idx)
                loss_sum += sum(w[i] * float(l_rows[i].detach()) for i in idx)
            else:
                mask = mi["response_mask"].clone()
                for i in range(len(mrows)):
                    if i not in idx:
                        mask[i] = 0
                mmi = {**mi, "response_mask": mask}
                t_x = float(grads.loss_tokens(mmi).sum())
                lx = rep.o4(mmi, student, minus_d, plus_d)
                target = lx * (t_x / t_m * scale)
                loss_sum += float(lx.detach()) * t_x / t_m
            last = j == len(jobs) - 1 and p == group[-1]
            for a, g in zip(acc, torch.autograd.grad(target, ctx.params, retain_graph=not last)):
                a.add_(g.float())
            del target
        key = "+".join(group)
        res["partitions"].setdefault(key, {})[method] = {
            "loss_sum": loss_sum, "loss_rel_diff": abs(loss_sum - res["loss"]) / max(abs(res["loss"]), 1e-30),
            "grad_rel_diff": rel_diff(acc, total)}
        del acc
    res["total_sq_norm"] = sum(float(t.double().pow(2).sum()) for t in total)
    return res


def main(argv: list[str]):
    import torch

    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, out, mode, *keys = argv
    grads.setup_paths(repo)
    if mode not in ("O", "Fx") or (mode == "Fx") != (os.environ.get("VAD_PACKFIX") == "1"):
        raise SystemExit(f"STOP: mode {mode} with VAD_PACKFIX={os.environ.get('VAD_PACKFIX')}")
    t0 = time.time()
    ctx = grads.Context(repo, command_txt, model_dir, g3_params)
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    log = {"mode": mode, "model_dir": model_dir, "load_seconds": round(time.time() - t0, 1), "results": {}}
    for k in keys:
        rows = [json.loads(x) for x in open(Path(rows_dir) / f"ROWS_{k}.jsonl")]
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        r = diagnose(ctx, pick_micro(rows), rollouts)
        r.update(seconds=round(time.time() - t, 1), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        log["results"][k] = r
        print(json.dumps({k: {"rows": r["rows"], "partitions": r["partitions"]}}), flush=True)
    Path(out).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
