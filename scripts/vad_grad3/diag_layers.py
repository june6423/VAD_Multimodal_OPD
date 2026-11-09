"""Layer diagnostic of HANDOFF-VLM-VAD-GRAD-003-R3 Part B item 2 (forward O, one micro-batch per key).

The micro-batch is the one of diag_partsum.pick_micro (the TASK-VLM-VAD-GRAD-003-004 micro-batches). One student
forward (Capture), then on the retained graph, as grads.run_batch does:
  total   the backward of the O4 loss x B_m / mini;
  scale3  the backward of 3 x that target, divided by 3;
  parts   for each part of C/I and L1/L2/L3 with rows, the backward of the masked O4 loss x T_X / T_m x B_m / mini
          (Definition 9.1); the parts of a split are summed.
Tensor hooks record the gradient at these places: the output of each decoder layer of the language model
(language_model.layers.N, type linear_attn or self_attn), the student log-probabilities (log_probs) and the student
top-k log-probabilities (topk_logps). For each place: the relative difference ||x - total|| / ||total|| of scale3 and
of each part sum, fp64. The same values for the parameter gradients, for all 248 tensors and for each layer.
The backward goes from the log-probabilities to layer 31 and then to layer 0. The rank factor 1/6 of grads.py is
left out (it cancels in each ratio).

Usage: python diag_layers.py <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt>
         <out json> <key>...
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag_partsum as partsum  # noqa: E402
import grads  # noqa: E402

LAYER = re.compile(r"(?:^|\.)layers\.(\d+)$")


def layer_places(model) -> list[tuple[str, str, object]]:
    """(place, type, module) for each decoder layer of the language model, in layer order."""
    found = []
    for name, m in model.named_modules():
        hit = LAYER.search(name)
        if hit and "visual" not in name:
            kind = "linear_attn" if hasattr(m, "linear_attn") else "self_attn" if hasattr(m, "self_attn") else \
                type(m).__name__
            found.append((int(hit.group(1)), f"layer.{hit.group(1)}", kind, m))
    return [(p, k, m) for _, p, k, m in sorted(found, key=lambda x: x[0])]


class Probe:
    """Captures the place tensors of one forward and the gradient at each place in each backward."""

    def __init__(self, places):
        self.kinds = {}
        self.armed = False
        self.tensors = {}
        for name, kind, module in places:
            self.kinds[name] = kind
            module.register_forward_hook(self._capture(name))

    def _capture(self, name):
        def hook(module, inputs, output):
            # Only in a forward of forward(); the recomputation of gradient checkpointing runs in the backward.
            if self.armed:
                if name in self.tensors:
                    raise RuntimeError(f"place {name} captured twice in one forward")
                self.tensors[name] = output[0] if isinstance(output, tuple) else output
        return hook

    def forward(self, fn):
        self.tensors = {}
        self.armed = True
        try:
            return fn()
        finally:
            self.armed = False

    def watch(self, name: str, kind: str, tensor):
        self.kinds[name] = kind
        self.tensors[name] = tensor

    def backward(self, target, params, retain: bool):
        """Parameter gradients and {place: fp32 gradient at the place}."""
        import torch

        got = {}

        def sink(name):
            def hook(grad):
                got[name] = grad.detach().to(torch.float32, copy=True)
            return hook

        handles = [t.register_hook(sink(n)) for n, t in self.tensors.items()]
        try:
            out = torch.autograd.grad(target, params, retain_graph=retain)
        finally:
            for h in handles:
                h.remove()
        return out, got


def rel_places(x: dict, total: dict, div: float = 1.0) -> dict:
    """{place: ||x / div - total|| / ||total||}, fp64; null for a place without a total norm or without x."""
    out = {}
    for n, t in total.items():
        t2 = float(t.double().pow(2).sum())
        if n not in x or t2 == 0:
            out[n] = None
            continue
        out[n] = math.sqrt(float((x[n].double() / div - t.double()).pow(2).sum()) / t2)
    return out


def rel_params(x: list, total: list, names: list[str], div: float = 1.0) -> dict:
    """Parameter relative difference for all tensors ("all") and for each layer ("layer.N"), fp64."""
    d2, t2 = {}, {}
    for n, a, t in zip(names, x, total):
        hit = re.search(r"layers\.(\d+)\.", n)
        keys = ("all", f"layer.{hit.group(1)}" if hit else "other")
        dd = float((a.double() / div - t.double()).pow(2).sum())
        tt = float(t.double().pow(2).sum())
        for k in keys:
            d2[k] = d2.get(k, 0.0) + dd
            t2[k] = t2.get(k, 0.0) + tt
    return {k: (math.sqrt(d2[k] / t2[k]) if t2[k] > 0 else None) for k in d2}


def add_into(acc: dict, x: dict):
    for n, v in x.items():
        if n in acc:
            acc[n].add_(v)
        else:
            acc[n] = v.clone()


def diagnose(ctx, probe: Probe, mrows: list[dict], rollouts: dict, names: list[str]) -> dict:
    import torch

    mini = {8: 128, 1: 16}[mrows[0]["n"]]
    scale = len(mrows) / mini
    Capture, Replay = ctx.classes
    cap, rep = Capture(ctx.actor, ctx.params), Replay(ctx.actor, ctx.params)
    mi, _ = ctx.inputs([rollouts[r["traj"]] for r in mrows])
    t_m = float(grads.loss_tokens(mi).sum())
    res = {"rows": len(mrows), "micro": list(grads.micro_key(mrows[0])), "batch": mrows[0]["batch"],
           "trajs": [r["traj"] for r in mrows], "labels": [r["label"] for r in mrows],
           "len_bins": [r["len_bin"] for r in mrows], "scale": scale, "coverage": partsum.coverage(mrows)}

    loss, student, minus, plus = probe.forward(lambda: cap.run(mi))
    probe.watch("log_probs", "log_probs", student["log_probs"])
    probe.watch("topk_logps", "topk_logps", student["topk_logps"])
    res["loss"] = float(loss.detach())
    res["places"] = [{"place": n, "type": probe.kinds[n], "shape": list(t.shape)} for n, t in probe.tensors.items()]

    g, h_tot = probe.backward(loss * scale, ctx.params, True)
    g_tot = [x.float() for x in g]
    del g
    res["total_sq_norm"] = {"params": sum(float(x.double().pow(2).sum()) for x in g_tot),
                            **{n: float(v.double().pow(2).sum()) for n, v in h_tot.items()}}
    res["missing_places"] = [n for n in probe.tensors if n not in h_tot]

    g, h = probe.backward(loss * (3 * scale), ctx.params, True)
    res["scale3"] = {"params": rel_params(g, g_tot, names, 3.0), "places": rel_places(h, h_tot, 3.0)}
    del g, h

    minus_d = {k: v.detach() for k, v in minus.items() if hasattr(v, "dim")}
    plus_d = {k: v.detach() for k, v in plus.items() if hasattr(v, "dim")}
    jobs = [(group, p) for group in partsum.PARTITIONS for p in group if grads.members(p, mrows)]
    acc_g = {"+".join(gr): [torch.zeros_like(t) for t in g_tot] for gr in partsum.PARTITIONS}
    acc_h = {"+".join(gr): {} for gr in partsum.PARTITIONS}
    part_losses = {}
    for j, (group, p) in enumerate(jobs):
        idx = grads.members(p, mrows)
        mask = mi["response_mask"].clone()
        for i in range(len(mrows)):
            if i not in idx:
                mask[i] = 0
        mmi = {**mi, "response_mask": mask}
        lx = rep.o4(mmi, student, minus_d, plus_d) * (float(grads.loss_tokens(mmi).sum()) / t_m)
        part_losses[p] = float(lx.detach())
        g, h = probe.backward(lx * scale, ctx.params, j < len(jobs) - 1)
        split = "+".join(group)
        for a, x in zip(acc_g[split], g):
            a.add_(x.float())
        add_into(acc_h[split], h)
        del g, h, lx, mmi
    res["part_losses"] = part_losses
    res["parts"] = {}
    for group in partsum.PARTITIONS:
        split = "+".join(group)
        s = sum(part_losses.get(p, 0.0) for p in group)
        res["parts"][split] = {"loss_rel_diff": abs(s - res["loss"]) / max(abs(res["loss"]), 1e-30),
                               "params": rel_params(acc_g[split], g_tot, names),
                               "places": rel_places(acc_h[split], h_tot)}
    del loss, student, minus, plus, acc_g, acc_h, g_tot, h_tot
    return res


def table(r: dict) -> list[str]:
    """One line per place in backward order (log-probabilities first, layer 0 last), then the parameter lines."""
    order = [p["place"] for p in r["places"]][::-1]
    splits = list(r["parts"])
    lines = [f"{'place':<12} {'type':<12} {'scale3':>10} " + " ".join(f"{s:>10}" for s in splits)
             + f" {'params_scale3':>14} " + " ".join(f"{'params_' + s:>16}" for s in splits)]

    def f(x):
        return f"{x:.3e}" if isinstance(x, float) else "null"

    for n in order:
        kind = next(p["type"] for p in r["places"] if p["place"] == n)
        lines.append(f"{n:<12} {kind:<12} {f(r['scale3']['places'].get(n)):>10} "
                     + " ".join(f"{f(r['parts'][s]['places'].get(n)):>10}" for s in splits)
                     + f" {f(r['scale3']['params'].get(n)):>14} "
                     + " ".join(f"{f(r['parts'][s]['params'].get(n)):>16}" for s in splits))
    lines.append(f"{'params all':<25} {f(r['scale3']['params']['all']):>10} "
                 + " ".join(f"{f(r['parts'][s]['params']['all']):>10}" for s in splits))
    return lines


def main(argv: list[str]):
    import torch

    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, out, *keys = argv
    grads.setup_paths(repo)
    import ckpt

    if os.environ.get("VAD_PACKFIX") == "1":
        raise SystemExit("STOP: the layer diagnostic uses forward O (VAD_PACKFIX unset)")
    t0 = time.time()
    ctx = grads.Context(repo, command_txt, model_dir, g3_params)
    names = ckpt.grad_set(g3_params)
    places = layer_places(ctx.actor.actor_module)
    n_layers = len({re.search(r"layers\.(\d+)\.", n).group(1) for n in names if re.search(r"layers\.(\d+)\.", n)})
    if len(places) != n_layers:
        raise SystemExit(f"STOP: {len(places)} decoder-layer modules found, the gradient tensors have {n_layers} layers")
    probe = Probe(places)
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    log = {"mode": "O", "model_dir": model_dir, "load_seconds": round(time.time() - t0, 1),
           "layers": len(places), "layer_types": {p: k for p, k, _ in places},
           "param_dtypes": sorted({str(p.dtype) for p in ctx.params}),
           "FLASH_ATTENTION_DETERMINISTIC": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"), "results": {}}
    for k in keys:
        rows = [json.loads(x) for x in open(Path(rows_dir) / f"ROWS_{k}.jsonl")]
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        r = diagnose(ctx, probe, partsum.pick_micro(rows), rollouts, names)
        r.update(seconds=round(time.time() - t, 1), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        log["results"][k] = r
        print(f"== {k} micro {r['micro']} rows {r['rows']} labels {r['labels']} len_bins {r['len_bins']} "
              f"missing {r['missing_places']} seconds {r['seconds']} peak_gpu_gb {r['peak_gpu_gb']:.1f}", flush=True)
        print("\n".join(table(r)), flush=True)
    Path(out).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
