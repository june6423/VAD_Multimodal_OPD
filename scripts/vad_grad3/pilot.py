"""PILOT=1 helpers of run_grad3.sh (HANDOFF-VLM-VAD-GRAD-003-R2 Part A item 9).

select: 2 folds x 6 prompts of one checkpoint. Fold 1 (it holds the random-partition and repeat batches) and the fold
  of the prompt with the longest response (the next fold when that is fold 1). In each fold, in this order: the
  longest-response prompt (if in the fold), the first mixed prompt (>= 2 correct and >= 2 incorrect siblings), the
  prompt with the most tokens, the prompt with the fewest tokens, then the next prompts in ascending order.
project: the FULL wall time and the peak disk from the pilot outputs. Per checkpoint and key class (B8 with 6 parts,
  B1 with 3 parts, O total only, Fx total only), seconds per token of the pilot micro-batches (inputs, forward,
  backward) times the FULL tokens of the class (row files of rows_full/); the batch-end seconds per vector times the
  FULL vector count; the Gram seconds times the FULL vector count / the pilot vector count; the model loads per
  process; the displacement seconds once. Peak disk = (FULL vectors + 6 partial vectors + 4 displacements) x 4 x
  elements bytes.

Usage: python pilot.py select <rollouts.jsonl> <LABELS.jsonl> <PILOT_PROMPTS.json>
       python pilot.py project <out dir> <PROJECTION.json> <checkpoint>...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import batches  # noqa: E402

PER_FOLD = 6
ELEMENTS = 3569090560


def select(rollouts: list[dict], labels: dict[int, int]) -> dict:
    by_prompt: dict[int, list[dict]] = {}
    for r in rollouts:
        by_prompt.setdefault(r["prompt_pos"], []).append(r)
    folds = {p: f + 1 for p, f in batches.fold_of(sorted(by_prompt)).items()}
    resp = {p: max(len(r["response_ids"]) for r in rs) for p, rs in by_prompt.items()}
    tokens = {p: sum(batches.seqlen(r) for r in rs) for p, rs in by_prompt.items()}
    n_c = {p: sum(labels[r["traj"]] for r in rs) for p, rs in by_prompt.items()}
    longest = max(sorted(by_prompt), key=lambda p: resp[p])
    second = folds[longest] if folds[longest] != 1 else 2
    chosen, mixed = [], []
    for f in (1, second):
        fp = sorted(p for p in by_prompt if folds[p] == f)
        mix = [p for p in fp if n_c[p] >= 2 and len(by_prompt[p]) - n_c[p] >= 2]
        order = ([longest] if longest in fp else []) + mix[:1] + [max(fp, key=lambda p: tokens[p]),
                                                                  min(fp, key=lambda p: tokens[p])] + fp
        pick = []
        for p in order:
            if p not in pick:
                pick.append(p)
        chosen += pick[:PER_FOLD]
        mixed += [p for p in pick[:PER_FOLD] if p in mix]
    return {"prompts": sorted(chosen), "folds": [1, second], "longest_response_prompt": longest,
            "longest_response_tokens": resp[longest], "mixed_prompts": mixed}


def key_class(tag: str) -> str:
    mode, key = tag.split("__", 1)
    if mode == "Fx":
        return "Fx"
    if "@" not in key and key.startswith("B8-"):
        return "B8"
    if "@" not in key and key.startswith("B1-"):
        return "B1"
    return "O_total"


def full_tokens(rows_full: Path) -> dict[str, float]:
    tok = lambda name: sum(json.loads(x)["total_tokens"] for x in open(rows_full / f"ROWS_{name}.jsonl"))  # noqa: E731
    b8 = sum(tok(f"B8-f{f}") for f in range(1, 5))
    f1 = tok("B8-f1")
    return {"B8": b8, "B1": sum(tok(f"B1-f{f}-j{j}") for f in range(1, 5) for j in range(8)), "O_total": 4 * f1,
            "Fx": 4 * f1}


FULL_VECTORS = {"B8": 4 * 6, "B1": 32 * 3, "O_total": 4, "Fx": 4}


def project(out: Path, cks: list[str]) -> dict:
    times = [json.loads(x) for x in open(out / "TIMES.jsonl")]
    seconds = {t["stage"]: t["seconds"] for t in times}
    res = {"checkpoints": {}, "elements": ELEMENTS}
    total = seconds.get("displacement", 0.0)
    for ck in cks:
        d = out / ck
        tok, sec, vec_s, vec_n = {}, {}, {}, {}
        for p in sorted(d.glob("progress_*.json")):
            tag = p.stem[len("progress_"):]
            c = key_class(tag)
            state = json.loads(p.read_text())
            for m in state["micro"]:
                tok[c] = tok.get(c, 0) + m["tokens"]
                sec[c] = sec.get(c, 0.0) + m["seconds_inputs"] + m["seconds_forward"] + m["seconds_backward"]
        loads, ends = [], []
        for line in open(d / "runs.jsonl"):
            run = json.loads(line)
            if "load_seconds" in run:
                loads.append(run["load_seconds"])
            for r in run.get("results", []):
                if "seconds_end" in r:
                    c = key_class(r["key"])
                    vec_s[c] = vec_s.get(c, 0.0) + r["seconds_end"]
                    vec_n[c] = vec_n.get(c, 0) + len(r["sq_norm"])
        ft = full_tokens(d / "rows_full")
        compute = {c: sec[c] / tok[c] * ft[c] for c in tok}
        saves = {c: vec_s[c] / vec_n[c] * FULL_VECTORS[c] for c in vec_n}
        pilot_vectors = sum(vec_n.values()) + 4
        gram = seconds[f"gram_{ck}"] * (sum(FULL_VECTORS.values()) + 4) / pilot_vectors
        ck_total = sum(compute.values()) + sum(saves.values()) + gram + 2 * max(loads)
        res["checkpoints"][ck] = {"pilot_tokens": tok, "pilot_seconds": sec, "full_tokens": ft,
                                  "compute_seconds": compute, "save_seconds": saves, "gram_seconds": gram,
                                  "load_seconds": 2 * max(loads), "total_seconds": ck_total,
                                  "pilot_vectors": pilot_vectors}
        total += ck_total
    res["displacement_seconds"] = seconds.get("displacement", 0.0)
    res["total_hours"] = total / 3600
    res["peak_disk_tb"] = (sum(FULL_VECTORS.values()) + 6 + 4) * 4 * ELEMENTS / 1e12
    res["gate_72h"] = res["total_hours"] <= 72
    return res


def main():
    cmd = sys.argv[1]
    if cmd == "select":
        rollouts = [json.loads(x) for x in open(sys.argv[2])]
        labels = {r["traj"]: int(r["label"]) for r in map(json.loads, open(sys.argv[3]))}
        Path(sys.argv[4]).write_text(json.dumps(select(rollouts, labels), indent=1))
    elif cmd == "project":
        res = project(Path(sys.argv[2]), sys.argv[4:])
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
        print(json.dumps({"total_hours": res["total_hours"], "peak_disk_tb": res["peak_disk_tb"]}))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
