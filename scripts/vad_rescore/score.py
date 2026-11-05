"""Scores of fixed trajectories under one model (HANDOFF-VLM-VAD-GRAD-002-R1 Part A items 1 to 4).

The inputs are the student inputs of scripts/vad_grad/moments.py at 1d0621be (A8 trainer_io: the A1 config, the M0
tokenizer and processor, the agent-loop prompt, assemble_batch with the rollout response token ids). The model is
loaded with trainer_io.load_model (bf16 parameters, fp32 buffers) in eval mode. One forward of
DataParallelPPOActor._forward_micro_batch (the A1 code path, temperature 1.0) under torch.no_grad gives, for the
siblings of one prompt (up to BATCH rows):
  L  = sum of the response-token log-probabilities, L1 = the first response-token log-probability,
  Lc = the log-probability at the answer-letter position of ANSWER_POS.jsonl (NaN without a position),
  and, for the set S0, the probabilities of the option-letter token ids at the first response position
  (topk_logps = gathered logits - logsumexp over the full vocabulary).
A CUDA out-of-memory error scores the prompt again with one sibling per forward (fallback table) and records it.
Outputs in <out dir>: SCORES_<set>_<model>.jsonl, FIRSTTOK_<model>.jsonl (set S0), progress_<set>_<model>.json after
each prompt (a restart skips the completed prompts), runs_<model>.jsonl.

Usage: python score.py <repo root> <COMMAND_B200.txt> <model dir> <model name> <out dir> <set spec> [<set spec> ...]
  set spec: <set name>=<rollouts.jsonl>=<LABELS.jsonl>=<ANSWER_POS.jsonl>[=<prompt list json>]
Env: BATCH (1), STOP_AFTER_PROMPTS (0 = no forced stop; the pilot stops once to test the resume, exit 75).
BATCH must stay 1 for Qwen3.5: with use_remove_padding the rows of a micro-batch are packed into one sequence, and the
linear-attention layers (Qwen3_5GatedDeltaNet: causal_conv1d_fn with seq_idx=None, chunk_gated_delta_rule without
cu_seqlens) carry their state over the row boundaries. Rows after the first then get wrong log-probabilities
(pilot TASK-VLM-VAD-GRAD-002-002: BATCH 8, only sibling 0 of each prompt equals logp_sum of TRAJ.jsonl).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP_EXIT = 75
NO_POSITION = "no answer-letter position in ANSWER_POS.jsonl"
OPTION = re.compile(r"(?m)^\s*\(?([A-F])[\.\):]")


class Stop(Exception):
    """Forced stop after STOP_AFTER_PROMPTS prompts (pilot)."""


def option_letters(raw_prompt) -> list[str]:
    """Sorted option letters of the prompt text (lines that start with 'A.', 'B)', '(C)' and so on)."""
    text = "\n".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]) for m in raw_prompt)
    return sorted(set(OPTION.findall(text)))


def letter_ids(tokenizer, letters: list[str]) -> dict[str, int]:
    """Token id of each letter that is one token without a leading space (the form of a first response token)."""
    out = {}
    for c in letters:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            out[c] = ids[0]
    return out


def score_forward(actor, mi: dict, ids: list[int], positions: list[int | None]) -> tuple[list[dict], list[float] | None]:
    """L, L1, Lc of each row of one micro-batch, and the probabilities of ids at the first response position of row 0."""
    import torch

    bs, length = mi["responses"].shape
    kwargs = {}
    if ids:
        idx = torch.tensor(ids, device=mi["responses"].device).view(1, 1, -1).expand(bs, length, len(ids)).contiguous()
        kwargs["topk_indices"] = idx
    with torch.no_grad():
        out = actor._forward_micro_batch(mi, temperature=1.0, calculate_entropy=False, **kwargs)
    lp = out["log_probs"].float()
    mask = mi["response_mask"].float()
    total = (lp * mask).sum(-1)
    recs = []
    for i in range(bs):
        pos = positions[i]
        recs.append({"L": float(total[i]), "L1": float(lp[i, 0]), "Lc": float(lp[i, pos]) if pos is not None else math.nan,
                     "response_tokens": int(mask[i].sum())})
    first = out["topk_logps"][0, 0].float().exp().tolist() if ids else None
    return recs, first


def first_token_record(prompt: int, gt: str, lids: dict[str, int], probs: list[float] | None) -> dict:
    rec = {"prompt": prompt, "gt": gt, "letters": sorted(lids), "letter_ids": [lids[c] for c in sorted(lids)]}
    if not lids:
        return {**rec, "P_letter": None, "P_gt": None, "P_wrong": None, "cause": "no option letter"}
    p = dict(zip(sorted(lids), probs))
    rec["p"] = p
    rec["P_letter"] = sum(p.values())
    if gt not in p:
        return {**rec, "P_gt": None, "P_wrong": None, "cause": "ground-truth letter is not an option letter token"}
    rec["P_gt"] = p[gt]
    rec["P_wrong"] = rec["P_letter"] - p[gt]
    rec["cause"] = None
    return rec


class Progress:
    """progress_<set>_<model>.json: done prompts, trajectory records and first-token records, replaced atomically."""

    def __init__(self, path: Path):
        self.path = path
        self.state = json.loads(path.read_text()) if path.exists() else {"done": [], "records": [], "firsttok": []}

    @property
    def done(self) -> set[int]:
        return set(self.state["done"])

    def add(self, prompt: int, records: list[dict], first: dict | None):
        self.state["done"].append(prompt)
        self.state["records"] += records
        if first is not None:
            self.state["firsttok"].append(first)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.path)


def run_set(set_name: str, model_name: str, out: Path, by_prompt: dict[int, list[dict]], prompts: list[int],
            labels: dict[int, int], positions: dict[int, int], score_prompt, counter: dict, stop_after: int) -> dict:
    """Score the prompts of one set with score_prompt(prompt, rows, positions) -> (records, first-token record or None).

    Writes progress after each prompt and the SCORES and FIRSTTOK files at the end. Raises Stop after stop_after prompts
    of this process (counter["prompts"] counts them over the sets)."""
    progress = Progress(out / f"progress_{set_name}_{model_name}.json")
    timing = []
    for p in prompts:
        if p in progress.done:
            continue
        rows = sorted(by_prompt[p], key=lambda r: r["sibling"])
        start = time.time()
        recs, first = score_prompt(p, rows, [positions.get(r["traj"]) for r in rows])
        for r, rec in zip(rows, recs):
            if not all(math.isfinite(rec[k]) for k in ("L", "L1")) or (r["traj"] in positions and not math.isfinite(rec["Lc"])):
                (out / "STOP.txt").write_text(f"non-finite score, {set_name} {model_name} trajectory {r['traj']}\n")
                raise SystemExit(f"STOP (stop condition 2): non-finite score, trajectory {r['traj']}")
            rec.update(traj=r["traj"], prompt=p, sibling=r["sibling"], label=labels[r["traj"]],
                       position=positions.get(r["traj"]), Lc_cause=None if r["traj"] in positions else NO_POSITION)
        progress.add(p, recs, first if set_name == "S0" else None)
        timing.append({"prompt": p, "seconds": round(time.time() - start, 3),
                       "tokens": sum(r["prompt_len"] + len(r["response_ids"]) for r in rows),
                       "batch_size": recs[0].get("batch_size")})
        counter["prompts"] += 1
        if stop_after and counter["prompts"] == stop_after:
            raise Stop()
    with open(out / f"SCORES_{set_name}_{model_name}.jsonl", "w") as f:
        for rec in sorted(progress.state["records"], key=lambda r: r["traj"]):
            f.write(json.dumps(rec) + "\n")
    if set_name == "S0":
        with open(out / f"FIRSTTOK_{model_name}.jsonl", "w") as f:
            for rec in sorted(progress.state["firsttok"], key=lambda r: r["prompt"]):
                f.write(json.dumps(rec) + "\n")
    return {"set": set_name, "prompts_scored": len(timing), "prompts_done": len(progress.done),
            "trajectories": len(progress.state["records"]), "prompt_timing": timing}


def main():
    import torch

    sys.path.insert(0, str(HERE.parent / "vad_rollout" / "a8"))
    sys.path.insert(0, str(HERE.parent / "vad_grad"))
    import provenance
    import trainer_io

    repo, command_txt, model_dir, model_name, out = sys.argv[1:6]
    specs = [s.split("=") for s in sys.argv[6:]]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    batch = int(os.environ.get("BATCH", "1"))
    stop_after = int(os.environ.get("STOP_AFTER_PROMPTS", "0"))
    run = {"model": model_name, "model_dir": model_dir, "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "batch": batch, "stop_after": stop_after, "verl": provenance.verl_origin(repo), "sets": [], "oom_fallback": []}
    todo_specs = [s for s in specs if not (out / f"SCORES_{s[0]}_{model_name}.jsonl").exists()]
    if not todo_specs:
        print(f"{model_name}: all score files exist, nothing to do")
        return
    t0 = time.time()
    cfg = trainer_io.compose_a1_config(repo, command_txt)
    tokenizer, processor = trainer_io.tokenizer_processor(cfg)
    dataset = trainer_io.load_dataset(cfg, tokenizer, processor)
    trainer_io.init_single_rank()
    model = trainer_io.load_model(model_dir, cfg, train=False)
    actor = trainer_io.make_actor(cfg, model, None)
    run["load_seconds"] = round(time.time() - t0, 1)
    torch.cuda.reset_peak_memory_stats()

    def score_prompt(p: int, rows: list[dict], positions: list[int | None]):
        import hashlib

        item = trainer_io.prompt_inputs(dataset, rows[0]["dataset_index"], processor, cfg)
        if hashlib.sha256(json.dumps(item["prompt_ids"]).encode()).hexdigest() != rows[0]["prompt_sha256"]:
            raise RuntimeError(f"prompt {p} differs from the rollout prompt")
        lids = letter_ids(tokenizer, option_letters(item["row"]["raw_prompt"]))
        ids = [lids[c] for c in sorted(lids)]

        def forward_all(size: int):
            recs, first = [], None
            for s in range(0, len(rows), size):
                chunk = rows[s:s + size]
                items = [{**item, "response_ids": r["response_ids"], "uid": str(r["prompt_pos"])} for r in chunk]
                b = trainer_io.to_device(trainer_io.assemble_batch(items, tokenizer, processor, cfg), "cuda")
                mi = {**b.batch, **b.non_tensor_batch, "pad_token_id": tokenizer.pad_token_id}
                r_, f_ = score_forward(actor, mi, ids, positions[s:s + size])
                recs += [{**x, "batch_size": len(chunk)} for x in r_]
                first = f_ if first is None else first
                del b, mi
            return recs, first

        try:
            recs, first = forward_all(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            run["oom_fallback"].append(p)
            recs, first = forward_all(1)
        return recs, first_token_record(p, rows[0]["gt"], lids, first)

    counter = {"prompts": 0}
    try:
        for name, rollouts, labels_path, answer_pos, *plist in todo_specs:
            rows = [json.loads(line) for line in open(rollouts)]
            labels = {r["traj"]: int(r["label"]) for r in map(json.loads, open(labels_path))}
            positions = {r["traj"]: r["position"] for r in map(json.loads, open(answer_pos))}
            by_prompt = defaultdict(list)
            for r in rows:
                by_prompt[r["prompt_pos"]].append(r)
            prompts = sorted(by_prompt) if not plist else sorted(json.loads(Path(plist[0]).read_text()))
            missing = [r["traj"] for p in prompts for r in by_prompt[p] if r["traj"] not in labels]
            if missing:
                raise SystemExit(f"STOP: {len(missing)} trajectories have no label, first {missing[:5]}")
            run["sets"].append(run_set(name, model_name, out, by_prompt, prompts, labels, positions, score_prompt,
                                       counter, stop_after))
    except Stop:
        run.update(stopped_after=stop_after, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        with open(out / f"runs_{model_name}.jsonl", "a") as f:
            f.write(json.dumps(run) + "\n")
        print(f"{model_name}: forced stop after {stop_after} prompts")
        sys.exit(STOP_EXIT)
    run.update(end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    with open(out / f"runs_{model_name}.jsonl", "a") as f:
        f.write(json.dumps(run) + "\n")
    print(f"{model_name}: done, sets {[s['set'] for s in run['sets']]}, OOM fallback prompts {run['oom_fallback']}")


if __name__ == "__main__":
    main()
