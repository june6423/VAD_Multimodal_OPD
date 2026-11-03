"""Cross-check of the moment estimators (HANDOFF-VLM-VAD-GRAD-001-R1 Part A item 8, gate G9).

At one checkpoint: trials of two disjoint groups of 4 correct and two disjoint groups of 4 incorrect trajectories,
at most one sibling per prompt in a group (the A8 trial draw, stats.draw_trials of scripts/vad_rollout/a8). The O4
gradient of each member comes from moments.trajectory_step. Per trial: c_4 = cos(C1, C2) and cos(I1, I2), raw and with
D. Next to the trial means: the c_4 that the moment estimators of the Gram file predict, c_4 = m / (m + sigma^2 / 4),
and the difference in units of the trial standard error sd / sqrt(trials).
Usage: python crosscheck.py <repo> <mapped COMMAND.txt> <model dir> <rollouts> <labels.jsonl> <V safetensors>
         <GRAM.json> <out json> [trials]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accum  # noqa: E402
import moments  # noqa: E402
import params as gparams  # noqa: E402
import stats  # noqa: E402

a8_stats = moments.a8.stats
TRIAL_SEED = 20261103


def main():
    repo, command_txt, model_dir, rollouts, labels_path, adam_v, gram_path, out = sys.argv[1:9]
    n_trials = int(sys.argv[9]) if len(sys.argv) > 9 else 16
    start = time.time()
    ctx = moments.a8.Context(repo, command_txt, model_dir, rollouts)
    ctx.objectives = moments.Capture(ctx.actor, ctx.params)
    d, metric_info = moments.load_d(adam_v, [n for n, _ in gparams.grad_params(ctx.actor.actor_module)], "cuda")
    labels = {r["traj"]: r["label"] for r in map(json.loads, open(labels_path))}
    pool = [{"id": r["traj"], "prompt": r["prompt_pos"], "correct": bool(labels[r["traj"]])} for r in ctx.rows]
    trials = a8_stats.draw_trials(pool, 4, n_trials, TRIAL_SEED)
    rows = []
    for i, trial in enumerate(trials):
        ts = time.time()
        sums = []
        for group in a8_stats.GROUPS:
            acc = [torch.zeros(p.shape, dtype=torch.float32, device="cuda") for p in ctx.params]
            for t in trial[group]:
                g, _ = moments.trajectory_step(ctx, ctx.model_inputs([t])[0], 0.0, None)
                for a, x in zip(acc, g):
                    a.add_(x.float())
                del g
            sums.append(acc)
        raw, met = accum.sibling_gram(sums, d)  # order C1, C2, I1, I2
        rec = {"trial": i, "members": {g: trial[g] for g in a8_stats.GROUPS}, "seconds": round(time.time() - ts, 1)}
        for name, gm in (("raw", raw), ("D", met)):
            rec[f"c4_C_{name}"] = gm[0][1] / math.sqrt(gm[0][0] * gm[1][1])
            rec[f"c4_I_{name}"] = gm[2][3] / math.sqrt(gm[2][2] * gm[3][3])
        rows.append(rec)
        del sums
    pred = stats.compute(json.loads(Path(gram_path).read_text()))["metrics"]
    summary = {"trials": n_trials, "n": 4, "seed": TRIAL_SEED, "metric": metric_info, "rows": rows, "compare": {}}
    for name in ("raw", "D"):
        for cls in ("C", "I"):
            vals = [r[f"c4_{cls}_{name}"] for r in rows]
            mean = sum(vals) / len(vals)
            se = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) if len(vals) > 1 else float("nan")
            p = pred[name]["subsets"][cls]["c_4"]
            summary["compare"][f"{name}|{cls}"] = {"trial_mean_c4": mean, "trial_se": se, "predicted_c4": p["value"],
                                                   "predicted_se": p["se"], "difference_in_trial_se": (mean - p["value"]) / se if se else None}
    summary["seconds"] = round(time.time() - start, 1)
    Path(out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
