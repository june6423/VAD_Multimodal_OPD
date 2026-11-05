"""Per-token diagnostic of the self-scores of one prompt (DIRECTIVE-VLM-VAD-GRAD-002-001, G5 failure on trajectory 2250).

score:   the score.py path. trainer_io.load_model(train=False), make_actor(cfg, model, None), assemble_batch of one
         sibling per forward, actor._forward_micro_batch under torch.no_grad with the kwargs of score.score_forward
         (temperature 1.0, calculate_entropy False, topk_indices of the option-letter ids). Records the per-token
         log-probabilities of that forward, and L of a second forward through score.score_forward itself.
moments: the moments.py path. a8 grads.Context (student train=True, teacher, the student, x-plus, x-minus and hint
         views), moments.Capture, Objectives.losses(mi, ["O4"]): the student forward with gradients of
         moments.trajectory_step. Records its per-token log-probabilities. No backward: it does not change the forward.
compare: per sibling, L of each run (fp32 sum as score.py and moments.py compute it), against logp_sum of TRAJ.jsonl
         and L of the FULL SCORES file, and the tokens where the runs differ.

Usage: python diag_prompt.py score   <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <prompt> <out json>
       python diag_prompt.py moments <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <prompt> <out json>
       python diag_prompt.py compare <TRAJ.jsonl> <SCORES jsonl> <out json> <run json>...
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fp32_sum(values: list[float]) -> float:
    """Sum of fp32 values in fp32 (torch), as (lp * mask).sum() in score.py and moments.py."""
    import torch

    return float(torch.tensor(values, dtype=torch.float32).sum())


def siblings(rollouts: str, prompt: int) -> list[dict]:
    rows = [r for r in map(json.loads, open(rollouts)) if r["prompt_pos"] == prompt]
    return sorted(rows, key=lambda r: r["sibling"])


def run_score(repo: str, command_txt: str, model_dir: str, rollouts: str, prompt: int) -> dict:
    import torch

    sys.path.insert(0, str(HERE.parent / "vad_rollout" / "a8"))
    sys.path.insert(0, str(HERE))
    import score
    import trainer_io

    cfg = trainer_io.compose_a1_config(repo, command_txt)
    tokenizer, processor = trainer_io.tokenizer_processor(cfg)
    dataset = trainer_io.load_dataset(cfg, tokenizer, processor)
    trainer_io.init_single_rank()
    model = trainer_io.load_model(model_dir, cfg, train=False)
    actor = trainer_io.make_actor(cfg, model, None)
    rows = siblings(rollouts, prompt)
    item = trainer_io.prompt_inputs(dataset, rows[0]["dataset_index"], processor, cfg)
    if hashlib.sha256(json.dumps(item["prompt_ids"]).encode()).hexdigest() != rows[0]["prompt_sha256"]:
        raise RuntimeError(f"prompt {prompt} differs from the rollout prompt")
    lids = score.letter_ids(tokenizer, score.option_letters(item["row"]["raw_prompt"]))
    ids = [lids[c] for c in sorted(lids)]
    out = []
    for r in rows:
        items = [{**item, "response_ids": r["response_ids"], "uid": str(r["prompt_pos"])}]
        b = trainer_io.to_device(trainer_io.assemble_batch(items, tokenizer, processor, cfg), "cuda")
        mi = {**b.batch, **b.non_tensor_batch, "pad_token_id": tokenizer.pad_token_id}
        length = mi["responses"].shape[1]
        kwargs = {}
        if ids:
            kwargs["topk_indices"] = torch.tensor(ids, device="cuda").view(1, 1, -1).expand(1, length, len(ids)).contiguous()
        with torch.no_grad():
            lp = actor._forward_micro_batch(mi, temperature=1.0, calculate_entropy=False, **kwargs)["log_probs"].float()
        n = int(mi["response_mask"].sum())
        rec, _ = score.score_forward(actor, mi, ids, [None])
        out.append({"traj": r["traj"], "sibling": r["sibling"], "tokens": lp[0, :n].tolist(),
                    "L": float((lp * mi["response_mask"].float()).sum()), "L_score_forward": rec[0]["L"]})
        del b, mi
    return {"path": "score", "letter_ids": ids, "siblings": out}


def run_moments(repo: str, command_txt: str, model_dir: str, rollouts: str, prompt: int) -> dict:
    sys.path.insert(0, str(HERE.parent / "vad_grad"))
    sys.path.insert(0, str(HERE.parent / "vad_rollout" / "a8"))
    import grads as a8
    import moments

    ctx = a8.Context(repo, command_txt, model_dir, rollouts)
    ctx.objectives = moments.Capture(ctx.actor, ctx.params)
    out = []
    for r in siblings(rollouts, prompt):
        mi = ctx.model_inputs([r["traj"]])[0]
        loss = ctx.objectives.losses(mi, ["O4"])["O4"]
        student = ctx.objectives.last_student
        ctx.objectives.last_student = None
        lp = student["log_probs"].detach().float()
        n = int(mi["response_mask"].sum())
        out.append({"traj": r["traj"], "sibling": r["sibling"], "tokens": lp[0, :n].tolist(),
                    "L": float((lp * mi["response_mask"]).sum()), "o4_loss": float(loss.detach())})
        del mi, loss, student, lp
    return {"path": "moments", "siblings": out}


def compare(traj_rows: dict[int, dict], scores: dict[int, dict], runs: list[dict]) -> dict:
    """For each sibling: L of each run, logp_sum of TRAJ.jsonl, L of the FULL SCORES file, and the tokens where the runs
    differ (index and the value of each run)."""
    names = [r["name"] for r in runs]
    per = {}
    for i, sib in enumerate(runs[0]["siblings"]):
        t = sib["traj"]
        toks = [r["siblings"][i]["tokens"] for r in runs]
        if any(r["siblings"][i]["traj"] != t for r in runs) or len({len(x) for x in toks}) != 1:
            raise SystemExit(f"STOP: runs do not hold the same tokens of trajectory {t}")
        diff = [{"index": k, **{n: x[k] for n, x in zip(names, toks)}}
                for k in range(len(toks[0])) if len({x[k] for x in toks}) > 1]
        L = {n: r["siblings"][i]["L"] for n, r in zip(names, runs)}
        ref = {"TRAJ_logp_sum": traj_rows[t]["logp_sum"], "FULL_L": scores[t]["L"]}
        groups = {}
        for k, v in {**L, **ref}.items():
            groups.setdefault(v, []).append(k)
        per[t] = {"sibling": sib["sibling"], "response_tokens": len(toks[0]), "L": L, **ref,
                  "L_score_forward": {n: r["siblings"][i]["L_score_forward"] for n, r in zip(names, runs)
                                      if "L_score_forward" in r["siblings"][i]},
                  "equal_groups": sorted(groups.values(), key=len, reverse=True), "differing_tokens": diff}
    return {"runs": names, "trajectories": per,
            "trajectories_with_any_difference": [t for t, v in per.items() if len(v["equal_groups"]) > 1]}


def main():
    cmd = sys.argv[1]
    if cmd in ("score", "moments"):
        repo, command_txt, model_dir, rollouts, prompt, out = sys.argv[2:8]
        t0 = time.time()
        fn = run_score if cmd == "score" else run_moments
        res = fn(repo, command_txt, model_dir, rollouts, int(prompt))
        res.update(prompt=int(prompt), model_dir=model_dir, seconds=round(time.time() - t0, 1))
        Path(out).write_text(json.dumps(res))
        print(json.dumps({s["traj"]: s["L"] for s in res["siblings"]}))
    elif cmd == "compare":
        traj_path, scores_path, out, *run_paths = sys.argv[2:]
        traj_rows = {r["traj"]: r for r in map(json.loads, open(traj_path))}
        scores = {r["traj"]: r for r in map(json.loads, open(scores_path))}
        runs = [{**json.loads(Path(p).read_text()), "name": Path(p).stem} for p in run_paths]
        res = compare(traj_rows, scores, runs)
        Path(out).write_text(json.dumps(res, indent=1))
        print(json.dumps({t: {"groups": v["equal_groups"], "differing_tokens": len(v["differing_tokens"])}
                          for t, v in res["trajectories"].items()}))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
