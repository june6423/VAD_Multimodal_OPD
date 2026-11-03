"""Pilot subset and FULL projection (HANDOFF-VLM-VAD-GRAD-001-R1 Part A item 10, gate G6).

select: per checkpoint 4 prompts (labels of label.py): the prompt with the longest response (up to 1024 tokens), then
  the lowest all_correct, the lowest all_wrong and the lowest mixed prompt that are not already chosen and are in a
  fold (accum.fold_of) that no earlier chosen prompt uses. When no prompt of a role is in a free fold: the lowest
  prompt of that role (DIRECTIVE-VLM-VAD-GRAD-001-005 Decisions 5 and 6). It prints the roles and folds.
project: the projected FULL wall time from the pilot records of run_grad.sh:
  per trajectory: least squares of seconds on [1, total tokens, weight != 0, answer position] over the pilot
    trajectories, applied to every trajectory of the 3 checkpoints;
  per prompt: the mean finish time (sibling Gram, group sums, host adds) x 375 x 3;
  saves: the measured seconds per byte x 18 saves of the 7 vectors of one fold x 3 checkpoints (15 every 25 prompts
    and 3 before the first prompt of folds 1 to 3);
  Gram file: the pilot Gram seconds scaled from the allocated vectors to 28, x 3;
  model loads: the pilot context seconds x 4 (3 checkpoints and the cross-check);
  cross-check: 16 trials x 16 gradients at the mean weight-0 trajectory time of step 80.
Usage: python pilot.py select <rollouts.jsonl> <labels.jsonl> <out json>
       python pilot.py project <pilot output root> <TASK-053 root> <labels root> <teacher root> <out json>
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accum  # noqa: E402

CHECKPOINTS = ("step0", "step80", "step130")
ROLES = ("longest", "all_correct", "all_wrong", "mixed")
SAVES_PER_CHECKPOINT = 18
FOLD_VECTORS = 7
VECTOR_BYTES = 3569090560 * 4


def prompt_class(labels: list[int]) -> str:
    c, i = sum(labels), len(labels) - sum(labels)
    return "mixed" if c >= 2 and i >= 2 else "all_correct" if i == 0 else "all_wrong" if c == 0 else "other"


def select(rollouts: str, labels_path: str) -> list[int]:
    rows = [json.loads(line) for line in open(rollouts)]
    labels = {r["traj"]: r["label"] for r in map(json.loads, open(labels_path))}
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_pos"]].append(r)
    folds = accum.fold_of(sorted(by_prompt))
    longest = max(rows, key=lambda r: (min(len(r["response_ids"]), 1024), -r["prompt_pos"]))["prompt_pos"]
    chosen = [longest]
    for cls in ROLES[1:]:
        cands = sorted(p for p, rs in by_prompt.items() if p not in chosen and prompt_class([labels[r["traj"]] for r in rs]) == cls)
        if not cands:
            raise SystemExit(f"STOP: no {cls} prompt")
        free = [p for p in cands if folds[p] not in {folds[c] for c in chosen}]
        chosen.append(free[0] if free else cands[0])
    print(f"{Path(rollouts).parent.name} pilot prompts {chosen} roles {list(ROLES)} folds {[folds[p] for p in chosen]}")
    return chosen


def lstsq(x: list[list[float]], y: list[float]) -> list[float]:
    import torch

    sol = torch.linalg.lstsq(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64).unsqueeze(1))
    return sol.solution.squeeze(1).tolist()


def features(total_tokens: float, weight_nonzero: bool, has_position: bool) -> list[float]:
    return [1.0, float(total_tokens), float(weight_nonzero), float(has_position)]


def project(root: Path, r053: Path, labels_root: Path, teacher_root: Path) -> dict:
    x, y, runs = [], [], []
    for name in CHECKPOINTS:
        for line in open(root / name / "TRAJ.jsonl"):
            t = json.loads(line)
            x.append(features(t["total_tokens"], t["weighted"], "position" in t))
            y.append(t["seconds"])
        runs += [json.loads(line) for line in open(root / name / "runs.jsonl")]
    coef = lstsq(x, y)
    traj_s = 0.0
    for name in CHECKPOINTS:
        rows = [json.loads(line) for line in open(r053 / name / "rollouts.jsonl")]
        labels = {r["traj"]: r["label"] for r in map(json.loads, open(labels_root / name / "LABELS.jsonl"))}
        teacher = {r["traj"]: r for p in glob.glob(str(teacher_root / name / "teacher_*.jsonl")) for r in map(json.loads, open(p))}
        by_prompt = defaultdict(list)
        for r in rows:
            by_prompt[r["prompt_pos"]].append(labels[r["traj"]])
        for r in rows:
            mixed = 0 < sum(by_prompt[r["prompt_pos"]]) < len(by_prompt[r["prompt_pos"]])
            f = features(r["prompt_len"] + len(r["response_ids"]), mixed, teacher[r["traj"]].get("position") is not None)
            traj_s += max(0.0, sum(c * v for c, v in zip(coef, f)))
    finish = [s for r in runs for s in r["finish_seconds"]]
    saves = [s for r in runs for s in r["saves"]]
    per_byte = sum(s["seconds"] for s in saves) / sum(s["bytes"] for s in saves) if saves else 0.0
    gram = [r["gram_seconds"] * 28 / r["vectors_allocated"] for r in runs if "gram_seconds" in r]
    context = max(r["context_seconds"] for r in runs)
    w0 = [yy for xx, yy in zip(x, y) if xx[2] == 0.0]
    parts = {"trajectories_s": traj_s, "prompt_finish_s": (sum(finish) / len(finish)) * 375 * 3,
             "saves_s": per_byte * SAVES_PER_CHECKPOINT * FOLD_VECTORS * VECTOR_BYTES * 3,
             "gram_s": (sum(gram) / len(gram)) * 3, "model_loads_s": context * 4,
             "crosscheck_s": 16 * 16 * (sum(w0) / len(w0))}
    total = sum(parts.values())
    return {"coefficients_seconds": dict(zip(("const", "per_token", "logp_backward", "answer_position"), coef)),
            "pilot_trajectories": len(y), "parts_s": parts, "total_h": total / 3600, "limit_h": 36.0,
            "within_limit": total / 3600 <= 36.0, "peak_gpu_gb": max(r.get("peak_gpu_gb", 0) for r in runs),
            "peak_rss_gb": max(r.get("peak_rss_gb", 0) for r in runs), "save_bytes_per_s": 1 / per_byte if per_byte else None}


def main():
    if sys.argv[1] == "select":
        Path(sys.argv[4]).write_text(json.dumps(select(sys.argv[2], sys.argv[3])))
    elif sys.argv[1] == "project":
        out = project(*(Path(a) for a in sys.argv[2:6]))
        Path(sys.argv[6]).write_text(json.dumps(out, indent=1))
        print(json.dumps(out))
    else:
        raise SystemExit(f"unknown command {sys.argv[1]}")


if __name__ == "__main__":
    main()
