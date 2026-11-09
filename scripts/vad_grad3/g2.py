"""Gate G2 of HANDOFF-VLM-VAD-GRAD-003-R2: the official n=8 step gradient of fold 1 at one checkpoint, from the
analysis code (grads.run_batch, key B8-f1@g2, total only) and from the dp_actor update path of 2627743a without an
optimizer step (update_path.update_path_gradient, the 6 A1 ranks in one process).

Checks: the update_policy micro-batches of each rank equal the micro-batches of the row file (members and order), and
rel_diff = ||g_analysis - g_update|| / ||g_update|| <= 1e-2 and cos >= 0.999. Writes G2.json; removes the analysis
vector file after the comparison. Exit 3 when the gate fails.

Usage: python g2.py <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt> <out dir>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = "B8-f1@g2"
REL_MAX, COS_MIN = 1e-2, 0.999
FAIL_EXIT = 3


def rank_batches(ctx, rows: list[dict], rollouts: dict, meta: dict):
    """The data of update_policy for each rank: its rows in the order of its balance_batch part (rank_place)."""
    for r in range(6):
        mine = sorted((x for x in rows if x["rank"] == r), key=lambda x: x["rank_place"])
        b = ctx.batch([rollouts[x["traj"]] for x in mine])
        b.meta_info.update(meta)
        yield b
        del b


def expected_partitions(rows: list[dict], count: int) -> list[list[list[int]]]:
    out = []
    for r in range(6):
        mine = [x for x in rows if x["rank"] == r]
        out.append([[x["rank_place"] for x in sorted((y for y in mine if y["micro"] == u), key=lambda y: y["position"])]
                    for u in range(count)])
    return out


def main():
    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, out = sys.argv[1:8]
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(HERE))
    import torch

    import grads
    import update_path

    if os.environ.get("VAD_PACKFIX") == "1":
        raise SystemExit("STOP: G2 uses forward O; VAD_PACKFIX must be unset")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(Path(rows_dir) / "ROWS_B8-f1.jsonl")]
    if any(r["rank_place"] is None for r in rows):
        raise SystemExit("STOP: ROWS_B8-f1.jsonl has rows without rank_place")
    count = max(r["micro"] for r in rows) + 1
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    t0 = time.time()
    ctx = grads.Context(repo, command_txt, model_dir, g3_params)
    t_load = time.time() - t0
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    opts = {"stop_after": 0, "commit_seconds": float(os.environ.get("COMMIT_SECONDS", "1800"))}
    res_a = grads.run_batch(ctx, "O", KEY, rows, rollouts, out, opts, {"micro": 0})
    t_analysis = time.time() - t0
    torch.cuda.empty_cache()

    t0 = time.time()
    meta = {"temperature": ctx.cfg.actor_rollout_ref.rollout.temperature, "pad_token_id": ctx.tokenizer.pad_token_id}
    acc, parts = update_path.update_path_gradient(ctx.actor, ctx.params, rank_batches(ctx, rows, rollouts, meta), count)
    g_update = torch.cat([a.reshape(-1) for a in acc]).cpu()
    del acc
    t_update = time.time() - t0

    vec_file = out / "vectors" / f"O__{KEY}__total.pt"
    g_analysis = torch.load(vec_file, mmap=True)
    cmp = update_path.compare(g_analysis, g_update)
    want = expected_partitions(rows, count)
    match = parts == want
    ok = match and cmp["rel_diff"] <= REL_MAX and cmp["cos"] >= COS_MIN
    res = {"gate": "G2", "pass": ok, "partitions_match": match, **cmp, "rel_diff_max": REL_MAX, "cos_min": COS_MIN,
           "model_dir": model_dir, "key": KEY, "rows": len(rows), "micro_batches": count,
           "rank_rows": [sum(1 for x in rows if x["rank"] == r) for r in range(6)],
           "dummies": rows[0].get("dummies"), "temperature": meta["temperature"], "analysis": res_a,
           "seconds_load": round(t_load, 1), "seconds_analysis": round(t_analysis, 1),
           "seconds_update": round(t_update, 1), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    if not match:
        res["partitions_update"], res["partitions_rows"] = parts, want
    (out / "G2.json").write_text(json.dumps(res, indent=1))
    del g_analysis
    vec_file.unlink()
    print(json.dumps({k: res[k] for k in ("pass", "partitions_match", "rel_diff", "cos", "norm_analysis", "norm_update")}))
    if not ok:
        sys.exit(FAIL_EXIT)


if __name__ == "__main__":
    main()
