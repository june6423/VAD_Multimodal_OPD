"""Pilot subset and FULL projection (HANDOFF-VLM-VAD-GRAD-002-R1 Part A item 7, gate G3).

select: 4 prompts of one set (labels of TASK-VLM-VAD-GRAD-001-005): the prompt with the longest response, the prompt
  with the longest input (prompt_len) among the others, then the lowest all_correct and the lowest mixed prompt
  (at least 2 correct and 2 incorrect siblings) that are not already chosen.
project: the projected FULL wall time from the runs_<model>.jsonl records of the pilot:
  per prompt: least squares of seconds on [1, tokens of the 8 siblings] over the pilot prompts of all models, applied
    to every prompt of the 9 pairs (S0 under 5 models, S80 under 4 models);
  model loads: the largest pilot load time x 5;
  statistics: the pilot stats.py seconds x (10000 / pilot resamples) x (375 / pilot prompts per set).
Usage: python pilot.py select <rollouts.jsonl> <LABELS.jsonl> <out json>
       python pilot.py project <pilot output root> <S0 rollouts.jsonl> <S80 rollouts.jsonl> <stats seconds>
         <pilot resamples> <out json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

LIMIT_H = 16.0
MODELS = {"S0": 5, "S80": 4}


def prompt_class(labels: list[int]) -> str:
    c, i = sum(labels), len(labels) - sum(labels)
    return "mixed" if c >= 2 and i >= 2 else "all_correct" if i == 0 else "all_wrong" if c == 0 else "other"


def select(rows: list[dict], labels: dict[int, int]) -> list[int]:
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_pos"]].append(r)
    chosen = [max(rows, key=lambda r: (len(r["response_ids"]), -r["prompt_pos"]))["prompt_pos"]]
    chosen.append(max((r for r in rows if r["prompt_pos"] not in chosen), key=lambda r: (r["prompt_len"], -r["prompt_pos"]))["prompt_pos"])
    for cls in ("all_correct", "mixed"):
        cands = sorted(p for p, rs in by_prompt.items()
                       if p not in chosen and prompt_class([labels[r["traj"]] for r in rs]) == cls)
        if not cands:
            raise SystemExit(f"STOP: no {cls} prompt")
        chosen.append(cands[0])
    return chosen


def lstsq(x: list[list[float]], y: list[float]) -> list[float]:
    import numpy as np

    return np.linalg.lstsq(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rcond=None)[0].tolist()


def project(root: Path, rollouts: dict[str, Path], stats_seconds: float, resamples: int) -> dict:
    runs = [json.loads(line) for path in sorted(root.glob("runs_*.jsonl")) for line in open(path)]
    timing = [t for r in runs for s in r["sets"] for t in s["prompt_timing"]]
    coef = lstsq([[1.0, t["tokens"]] for t in timing], [t["seconds"] for t in timing])
    prompt_s = 0.0
    for name, path in rollouts.items():
        tokens = defaultdict(int)
        for line in open(path):
            r = json.loads(line)
            tokens[r["prompt_pos"]] += r["prompt_len"] + len(r["response_ids"])
        prompt_s += MODELS[name] * sum(max(0.0, coef[0] + coef[1] * v) for v in tokens.values())
    pilot_prompts = max(len(s["prompt_timing"]) for r in runs for s in r["sets"]) or 1
    parts = {"prompts_s": prompt_s, "model_loads_s": 5 * max(r["load_seconds"] for r in runs),
             "stats_s": stats_seconds * (10000 / resamples) * (375 / pilot_prompts)}
    total_h = sum(parts.values()) / 3600
    return {"coefficients_seconds": {"const": coef[0], "per_token": coef[1]}, "pilot_prompt_records": len(timing),
            "parts_s": parts, "total_h": total_h, "limit_h": LIMIT_H, "within_limit": total_h <= LIMIT_H,
            "peak_gpu_gb": max(r.get("peak_gpu_gb", 0.0) for r in runs),
            "oom_fallback_prompts": sorted({p for r in runs for p in r["oom_fallback"]})}


def main():
    if sys.argv[1] == "select":
        rows = [json.loads(line) for line in open(sys.argv[2])]
        labels = {r["traj"]: int(r["label"]) for r in map(json.loads, open(sys.argv[3]))}
        chosen = select(rows, labels)
        print(f"{Path(sys.argv[2]).parent.name} pilot prompts {chosen} roles [longest_response, longest_input, all_correct, mixed]")
        Path(sys.argv[4]).write_text(json.dumps(chosen))
    elif sys.argv[1] == "project":
        root, s0, s80, seconds, resamples, out = sys.argv[2:8]
        res = project(Path(root), {"S0": Path(s0), "S80": Path(s80)}, float(seconds), int(resamples))
        Path(out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
    else:
        raise SystemExit(f"unknown command {sys.argv[1]}")


if __name__ == "__main__":
    main()
