"""Trainer metrics of each step for one arm (HANDOFF-VLM-VAD-PACKFIX-001-R2 Definition 7).

Sources, in this order for each metric:
1. metrics_<run>_<start>.jsonl of the verl file logger ({"step": s, "data": {...}}), one file for each run.
2. console.log lines "step:<s> - key:value - ..." (numbers only, as in train_arm.sh).
The rollout accuracy (HANDOFF-VLM-VAD-PACKFIX-001-R6 Definition 8) is acc_mean of the step in ROLLOUT_ACC.json
(rollout_acc.py, the judge session after the training). The A1 training reward is zero, so the trainer metrics and the
"score" of the rollout dumps do not hold it. STEP_METRICS.json records the source of each value.
The check fails (exit 1) when a step in 1..<steps> lacks a metric or a value is not finite.
Usage: python metrics.py <arm output root> <steps>
"""

import json
import math
import re
import sys
from pathlib import Path

KEYS = {
    "rollout_corr/kl": ["rollout_corr/kl"],
    "response_length/mean": ["response_length/mean"],
    "actor/grad_norm": ["actor/grad_norm"],
    "actor/entropy": ["actor/entropy"],
    "actor/pg_loss": ["actor/pg_loss"],
    "actor/vopd_loss": ["actor/vopd_loss"],
    "timing_s/step": ["timing_s/step"],
}
OPTIONAL = ("timing_s/testing", "timing_s/save_checkpoint", "timing_s/gen", "timing_s/update_actor")


def number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def file_logger(out: Path) -> dict:
    """metrics_<run>_<start seconds>.jsonl files in start order; a later run replaces the values of a step."""
    steps = {}
    for path in sorted(out.glob("metrics_*.jsonl"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        for line in path.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            steps.setdefault(int(rec["step"]), {}).update(
                {k: number(v) for k, v in rec["data"].items() if number(v) is not None})
    return steps


def console(path: Path) -> dict:
    steps = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"step:(\d+) - (.*)$", line)
            if not match:
                continue
            metrics = {}
            for item in match.group(2).split(" - "):
                key, sep, value = item.rpartition(":")
                if sep:
                    try:
                        metrics[key.strip()] = float(value)
                    except ValueError:
                        pass
            steps.setdefault(int(match.group(1)), {}).update(metrics)
    return steps


def rollout_acc(out: Path) -> dict:
    path = out / "ROLLOUT_ACC.json"
    return json.loads(path.read_text())["steps"] if path.exists() else {}


def main():
    out, n_steps = Path(sys.argv[1]), int(sys.argv[2])
    sources = {"file": file_logger(out), "console": console(out / "console.log")}
    acc = rollout_acc(out)
    result, missing, nonfinite = {}, [], []
    for step in range(1, n_steps + 1):
        row = {}
        for name, keys in KEYS.items():
            for src in ("file", "console"):
                vals = sources[src].get(step, {})
                hit = next((k for k in keys if k in vals), None)
                if hit is not None:
                    row[name] = {"value": vals[hit], "source": src}
                    break
            else:
                missing.append([step, name])
        for name in OPTIONAL:  # recorded when present (projection), never a failure
            for src in ("file", "console"):
                if name in sources[src].get(step, {}):
                    row[name] = {"value": sources[src][step][name], "source": src}
                    break
        a = acc.get(str(step), {})
        if a.get("acc_mean") is None:
            missing.append([step, "rollout_acc"])
        else:
            row["rollout_acc"] = {"value": a["acc_mean"], "source": "ROLLOUT_ACC.json", "rows": a["rows"]}
        for name, v in row.items():
            if not math.isfinite(v["value"]):
                nonfinite.append([step, name])
        result[str(step)] = row
    for step in (0,):  # validation before training (timing only)
        for src in ("file", "console"):
            if "timing_s/testing" in sources[src].get(step, {}):
                result["0"] = {"timing_s/testing": {"value": sources[src][step]["timing_s/testing"], "source": src}}
                break
    check = {"steps": n_steps, "missing": missing, "nonfinite": nonfinite, "pass": not missing and not nonfinite,
             "file_logger_steps": sorted(sources["file"]), "console_steps": sorted(sources["console"])}
    (out / "STEP_METRICS.json").write_text(json.dumps({"check": check, "steps": result}, indent=1, sort_keys=True))
    print(json.dumps(check))
    sys.exit(0 if check["pass"] else 1)


if __name__ == "__main__":
    main()
