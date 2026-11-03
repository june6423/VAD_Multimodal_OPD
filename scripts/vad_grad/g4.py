"""G4 (HANDOFF-VLM-VAD-GRAD-001-R1, Part B item 4): O4 gradient of the new code against the A8 code at a5ace7be.

One fixed trajectory per checkpoint: the lowest traj of the lowest prompt that is mixed by the TASK-053 labels.
New code: moments.trajectory_step (weight 0, and weight 1 with the log-probability backward). A8 code:
objectives.Objectives.gradients(mi, ["O4"]). Pass: cosine(new, A8) >= 0.9999, |norm difference| / norm(A8) < 1e-3, and
cosine of two repeated new computations >= 0.9999 (run with FLASH_ATTENTION_DETERMINISTIC=1). A failure is stop
condition 2 (exit 1).
Usage: python g4.py <repo root> <mapped A1 COMMAND.txt> <model dir> <rollouts.jsonl> <out json>
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accum  # noqa: E402
import moments  # noqa: E402

a8, obj = moments.a8, moments.obj


def cos(a, b) -> float:
    return accum.dot(a, b) / math.sqrt(accum.dot(a, a) * accum.dot(b, b))


def main():
    repo, command_txt, model_dir, rollouts, out = sys.argv[1:6]
    start = time.time()
    ctx = a8.Context(repo, command_txt, model_dir, rollouts)
    ctx.objectives = moments.Capture(ctx.actor, ctx.params)
    by_prompt = defaultdict(list)
    for r in ctx.rows:
        by_prompt[r["prompt_pos"]].append(r)
    prompt = min(p for p, rows in by_prompt.items()
                 if sum(r["correct"] for r in rows) >= 2 and sum(not r["correct"] for r in rows) >= 2)
    traj = min(r["traj"] for r in by_prompt[prompt])
    mi = ctx.model_inputs([traj])[0]
    a_buf = [torch.zeros(p.shape, dtype=torch.float32, device="cuda") for p in ctx.params]
    new0, rec = moments.trajectory_step(ctx, mi, 0.0, None)
    old = obj.Objectives(ctx.actor, ctx.params).gradients(mi, ["O4"])["O4"][1]
    res = {"model_dir": model_dir, "traj": traj, "prompt": prompt, "o4_loss": rec["o4_loss"],
           "flash_attention_deterministic": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
           "cos_new_a8": cos(new0, old), "norm_new": math.sqrt(accum.dot(new0, new0)), "norm_a8": math.sqrt(accum.dot(old, old))}
    res["rel_norm_diff"] = abs(res["norm_new"] - res["norm_a8"]) / res["norm_a8"]
    del old
    rep, _ = moments.trajectory_step(ctx, mi, 0.0, None)
    res["cos_repeat"] = cos(new0, rep)
    del rep
    with_logp, _ = moments.trajectory_step(ctx, mi, 1.0, a_buf)
    res["cos_weight1_weight0"] = cos(with_logp, new0)
    res["logp_grad_norm"] = math.sqrt(accum.dot(a_buf, a_buf))
    res["pass"] = res["cos_new_a8"] >= 0.9999 and res["rel_norm_diff"] < 1e-3 and res["cos_repeat"] >= 0.9999
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["seconds"] = round(time.time() - start, 1)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
