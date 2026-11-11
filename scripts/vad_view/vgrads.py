"""Stage 2 view gradients of HANDOFF-VLM-VAD-VIEW-001-R2 Definition 9.1 (Part A item 5).

g(B8(f), v): the official step gradient of HANDOFF-VLM-VAD-GRAD-003-R3 (Definitions 4, 6.1, 7) with x-plus = v. The
code is scripts/vad_grad3/grads.run_batch at 50801aa6, unchanged: the A1 row split of batches.py (ROWS_B8-f<f>.jsonl),
the packed O4 loss of each micro-batch times B_m / mini_cfg, the mean over 6 ranks, fp32 accumulators, a progress
file after each micro-batch, partial vectors for resume. Key B8-f<f>@<view>: the '@' suffix makes run_batch compute the
total only (no parts) and read the rows of B8-f<f>. run_batch also computes its scale-3 rounding accumulator
(rounding_rel in the progress file); it writes no vector file for it.
x-plus = v: ViewContext replaces the bbox_images entry of each dataset row by the image of view v (views MANIFEST.jsonl).
For v0 the image is the released teacher image, so the inputs equal the GRAD-003 inputs.
Vector files: vectors/O__B8-f<f>@<view>__total.pt (flat fp32, G3_PARAMS.txt order).

Usage: python vgrads.py <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt> <views dir>
         <out dir> <key>...
Env: STOP_AFTER_MICRO (0; exit 75 once after this many micro-batches), COMMIT_SECONDS (1800).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def setup(repo: str):
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(HERE.parent / "vad_grad3"))
    sys.path.insert(0, str(HERE))


def make_view_context(g3):
    class ViewContext(g3.Context):
        """grads.Context with x-plus = the image of self.view for every row."""

        def __init__(self, *args, manifest: dict, **kwargs):
            super().__init__(*args, **kwargs)
            self.manifest = manifest
            self.view = None

        def prompt(self, index: int) -> dict:
            item = super().prompt(index)
            if self.view is None:
                raise RuntimeError("ViewContext.view is not set")
            row = {**item["row"], "bbox_images": [{"path": self.manifest[index]["views"][self.view]["path"]}]}
            return {**item, "row": row}

    return ViewContext


def main():
    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, views_dir, out, *keys = sys.argv[1:]
    setup(repo)
    import grads as g3
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
    log.update(load_seconds=round(time.time() - t0, 1), grad_tensors=len(ctx.params),
               grad_elements=sum(p.numel() for p in ctx.params))
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
