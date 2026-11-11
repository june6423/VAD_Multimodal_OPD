"""Stage 2 view gradients of HANDOFF-VLM-VAD-VIEW-001-R3 Definitions 9.1 and 9.5 (Part A item 5).

g(B8(f), v): the official step gradient of HANDOFF-VLM-VAD-GRAD-003-R3 (Definitions 4, 6.1, 7) with x-plus = v. The
code is scripts/vad_grad3/grads.run_batch at 50801aa6, unchanged: the A1 row split of batches.py (ROWS_B8-f<f>.jsonl),
the packed O4 loss of each micro-batch times B_m / mini_cfg, the mean over 6 ranks, fp32 accumulators, a progress
file after each micro-batch, partial vectors for resume. Key B8-f<f>@<view>: the '@' suffix makes run_batch compute the
total only (no parts) and read the rows of B8-f<f>. run_batch also computes its scale-3 rounding accumulator
(rounding_rel in the progress file); it writes no vector file for it.
x-plus = v: ViewContext replaces the bbox_images entry of each dataset row by the image of view v (views MANIFEST.jsonl).
For v0 the image is the released teacher image, so the inputs equal the GRAD-003 inputs.
g_geo(B8(f)) (Definition 9.5, key B8-f<f>@geo): the same run_batch with the geometric-pool target of Definition 13.
ViewContext.inputs builds the inputs once for each view of GEO_VIEWS (the student and x-minus inputs must be equal
across the views); GeoCapture (make_geo_classes) runs the student forward and the x-minus forward once and the x-plus
forward once for each view, all on the student top-k indices, and gives the x-plus top-k input geo_topk(views) to the
official O4 loss. The view keys do not change.
Vector files: vectors/O__B8-f<f>@<view>__total.pt and vectors/O__B8-f<f>@geo__total.pt (flat fp32, G3_PARAMS.txt
order).

Usage: python vgrads.py <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt> <views dir>
         <out dir> <key>...
Env: STOP_AFTER_MICRO (0; exit 75 once after this many micro-batches), COMMIT_SECONDS (1800), GEO_VIEWS (comma list
of view names; default the 8 views).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEO = "geo"
GEO_KEY = "geo_plus_inputs"
SAME_KEYS = ("input_ids", "responses", "response_mask", "teacher_minus_input_ids")


def setup(repo: str):
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(HERE.parent / "vad_grad3"))
    sys.path.insert(0, str(HERE))


def geo_views() -> list[str]:
    """The views of the geometric pool: env GEO_VIEWS (comma list), default the 8 views of views.VIEW_KEYS."""
    import views

    env = os.environ.get("GEO_VIEWS", "")
    return env.split(",") if env else [views.view_name(lab, z) for lab, z in views.VIEW_KEYS]


def add_tail64(x):
    """add_tail of core_algos.py 1247-1260 (support mask None) in fp64."""
    import torch

    x = x.double()
    log_s = torch.logsumexp(x, dim=-1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(log_s))], dim=-1)


def geo_topk(views_topk: list):
    """Definition 13: the x-plus top-k input whose add_tail and centering in the VAD loss give phi_geo.

    views_topk: the x-plus topk_logps of each view on the student top-k indices (same shape [..., K]). For each view
    phi_v = add_tail(x_v) - mean (K + 1 entries, core_algos.py 1305-1315 with support mask None); phi_geo = mean over
    the views. Returns log_softmax(phi_geo)[..., :K] in the input dtype: its add_tail is log_softmax(phi_geo) (the
    tail entry is 1 - the sum of the K probabilities), and its centering is phi_geo."""
    import torch

    phis = []
    for x in views_topk:
        y = add_tail64(x)
        phis.append(y - y.mean(dim=-1, keepdim=True))
    phi = torch.stack(phis).mean(0)
    k = views_topk[0].shape[-1]
    return torch.log_softmax(phi, dim=-1)[..., :k].to(views_topk[0].dtype)


def make_view_context(g3):
    class ViewContext(g3.Context):
        """grads.Context with x-plus = the image of self.view for every row. For self.view == GEO, inputs() builds
        the inputs for each view of self.geo_views and puts the x-plus inputs of each view under GEO_KEY."""

        def __init__(self, *args, manifest: dict, **kwargs):
            super().__init__(*args, **kwargs)
            self.manifest = manifest
            self.view = None
            self.geo_views = geo_views()

        def prompt(self, index: int) -> dict:
            item = super().prompt(index)
            if self.view is None or self.view == GEO:
                raise RuntimeError(f"ViewContext.view is {self.view}")
            row = {**item["row"], "bbox_images": [{"path": self.manifest[index]["views"][self.view]["path"]}]}
            return {**item, "row": row}

        def inputs(self, ros: list[dict]) -> tuple[dict, list[dict]]:
            if self.view != GEO:
                return super().inputs(ros)
            import objectives as obj
            import torch

            built = []
            try:
                for v in self.geo_views:
                    self.view = v
                    built.append(super().inputs(ros))
            finally:
                self.view = GEO
            mi, rows_mi = built[0]
            for v, (m, _) in zip(self.geo_views[1:], built[1:]):
                for key in SAME_KEYS:
                    if not torch.equal(m[key], mi[key]):
                        raise RuntimeError(f"view {v}: {key} differs from view {self.geo_views[0]}")
            return {**mi, GEO_KEY: [obj.view_inputs(m, "plus") for m, _ in built]}, rows_mi

    return ViewContext


def make_geo_classes(g3, obj):
    """(GeoCapture, Replay): GeoCapture.run gives the O4 loss with the geometric-pool x-plus target (Definition 13).
    One student forward (with gradient, top-k), one x-minus forward, one x-plus forward for each view on the student
    top-k indices; the x-plus topk_logps = geo_topk of the views. The loss is Objectives.losses O4 on these outputs
    (Replay), so every other part of the loss is the official code."""
    _, Replay = g3.make_classes(obj)

    class GeoCapture(obj.Objectives):
        def run(self, mi):
            student = self._forward(mi, grad=True, distill_topk=self.topk)
            idx = student["topk_indices"]
            minus = self._forward(obj.view_inputs(mi, "minus"), grad=False, topk_indices=idx)
            plus = [self._forward(p, grad=False, topk_indices=idx) for p in mi[GEO_KEY]]
            geo = {**plus[0], "topk_logps": geo_topk([p["topk_logps"] for p in plus])}
            del plus
            loss = Replay(self.actor, self.params).o4(mi, student, minus, geo)
            return loss, student, minus, geo

    return GeoCapture, Replay


def main():
    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, views_dir, out, *keys = sys.argv[1:]
    setup(repo)
    import grads as g3
    import objectives as obj
    import provenance
    import torch

    if os.environ.get("VAD_PACKFIX") == "1":
        raise SystemExit("STOP: the official forward needs VAD_PACKFIX unset")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {r["row"]: r for r in map(json.loads, open(Path(views_dir) / "MANIFEST.jsonl"))}
    removed = {r["row"] for r in json.loads((Path(views_dir) / "REMOVED.json").read_text())["removed"]}
    opts = {"stop_after": int(os.environ.get("STOP_AFTER_MICRO", "0")),
            "commit_seconds": float(os.environ.get("COMMIT_SECONDS", "1800"))}
    todo = [k for k in keys if not (out / f"GRADROWS_O__{k}.jsonl").exists()]
    log = {"model_dir": model_dir, "keys": keys, "todo": todo, "start": g3.now(), "verl": provenance.verl_origin(repo),
           **opts}
    if not todo:
        print(f"{keys}: all GRADROWS files exist")
        return
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    t0 = time.time()
    ctx = make_view_context(g3)(repo, command_txt, model_dir, g3_params, manifest=manifest)
    classes = {"view": ctx.classes, GEO: make_geo_classes(g3, obj)}
    log.update(load_seconds=round(time.time() - t0, 1), grad_tensors=len(ctx.params),
               grad_elements=sum(p.numel() for p in ctx.params), geo_views=ctx.geo_views)
    torch.cuda.reset_peak_memory_stats()
    counter = {"micro": 0}
    log["results"] = []
    try:
        for k in todo:
            batch, view = k.split("@")
            rows = [json.loads(line) for line in open(Path(rows_dir) / f"ROWS_{batch}.jsonl")]
            bad = sorted({r["dataset_index"] for r in rows} & removed)
            if bad:
                raise SystemExit(f"STOP: rows {bad} of {batch} are removed (REMOVED.json); the batch would change")
            ctx.view = view
            ctx.classes = classes[GEO if view == GEO else "view"]
            ctx._prompts = {}
            log["results"].append(g3.run_batch(ctx, "O", k, rows, rollouts, out, opts, counter))
            print(json.dumps(log["results"][-1]), flush=True)
    except g3.Stop:
        log.update(stopped_after_micro=opts["stop_after"], end=g3.now(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        g3.append_jsonl(out / "runs.jsonl", log)
        print(f"forced stop after {opts['stop_after']} micro-batches")
        sys.exit(g3.STOP_EXIT)
    log.update(end=g3.now(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    g3.append_jsonl(out / "runs.jsonl", log)


if __name__ == "__main__":
    main()
