"""FULL wall-time projection of VLM-VAD-VIEW-002 from the pilot (HANDOFF-VLM-VAD-VIEW-002-R1 Part A item 7.5, G4).

Reference R = arm B of the PACKFIX FULL task (TASK-VLM-VAD-PACKFIX-001-029, 96 prompts x 8 per step, 20 steps): mean
timing_s/gen, mean timing_s/update_actor and mean timing_s/step of its STEP_METRICS.json; its full-set validation =
the mean over its checkpoint validations of the sum of the 4 val_only wall seconds (RUN_PACKFIX.txt); its rollout
scoring = the mean "seconds" of ROLLOUT_ACC.json. Pilot P (16 prompts per step) of each arm X: the means over the
pilot steps of the same three timings.
  step_X = R_gen x P_gen(X) / P_gen(B) + R_update x P_update(X) / P_update(B)
           + (R_step - R_gen - R_update) x P_rest(X) / P_rest(B),  P_rest = P_step - P_gen - P_update
  arm_X = startup_X + 40 x step_X + judge start + 2 x validation + 40 x scoring x rows_X / 768
          + 2 x (merge_X + heldout load_X + 500 x heldout item seconds_X)
  startup_X = pilot run1 wall seconds - pilot step 1 seconds; judge start = largest judge_start_seconds of the pilot;
  merge_X = the pilot merge wall seconds; heldout = the pilot heldout wall seconds minus the item seconds (load) and
  the mean item seconds of HELDOUT_B.jsonl; rows_X = 96 for N1R, else 768.
Task total = the sum of the 4 arms + the pilot time outside the arms (RUN_ALL.txt: total minus the arm wall seconds).
Writes PROJECTION.json with each part, step_X (the projection of stop condition 4), each arm total and the 66 h limit.
Usage: python projection.py <pilot root> <PACKFIX arm B dir> <out json>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ARMS = ("N1R", "N1V8", "B", "T8")
LIMIT_H, STEPS, HELDOUT_ITEMS = 66.0, 40, 500


def means(sm: dict, steps) -> dict:
    out = {}
    for k in ("timing_s/gen", "timing_s/update_actor", "timing_s/step"):
        xs = [sm[str(s)][k]["value"] for s in steps if k in sm.get(str(s), {})]
        out[k] = sum(xs) / len(xs)
    out["rest"] = out["timing_s/step"] - out["timing_s/gen"] - out["timing_s/update_actor"]
    return out


def reference(d: Path) -> dict:
    sm = json.loads((d / "STEP_METRICS.json").read_text())["steps"]
    m = means(sm, [s for s in sm if s != "0"])
    log = (d / "RUN_PACKFIX.txt").read_text()
    parts = {}
    for step, wall in re.findall(r"validate val(\d+)_p\d exit 0 wall_seconds (\d+)", log):
        parts.setdefault(step, []).append(float(wall))
    full = [sum(v) for v in parts.values() if len(v) == 4]
    ra = json.loads((d / "ROLLOUT_ACC.json").read_text())["steps"]
    return {**m, "validation_s": sum(full) / len(full), "validation_runs": len(full),
            "scoring_s": sum(v["seconds"] for v in ra.values()) / len(ra)}


def pilot_arm(d: Path) -> dict:
    sm = json.loads((d / "STEP_METRICS.json").read_text())["steps"]
    m = means(sm, [1, 2])
    log = (d / "RUN_VIEW2.txt").read_text()
    run1 = float(re.search(r"train run1 exit 0 wall_seconds (\d+)", log).group(1))
    items = [json.loads(x) for x in open(d / "heldout" / "2" / "HELDOUT_B.jsonl") if x.strip()]
    item_s = sum(x["seconds"] for x in items) / len(items)
    held_wall = float(re.search(r"heldout step 2 exit 0 wall_seconds (\d+)", log).group(1))
    return {**m, "startup_s": max(0.0, run1 - sm["1"]["timing_s/step"]["value"]),
            "judge_start_s": max(float(x) for x in re.findall(r"judge_start_seconds (\d+)", log)),
            "merge_s": float(re.search(r"merge step 2 wall_seconds (\d+)", log).group(1)),
            "heldout_item_s": item_s, "heldout_load_s": max(0.0, held_wall - item_s * len(items))}


def project(ref: dict, pil: dict) -> dict:
    b = pil["B"]
    res = {}
    for a in ARMS:
        p = pil[a]
        step = (ref["timing_s/gen"] * p["timing_s/gen"] / b["timing_s/gen"]
                + ref["timing_s/update_actor"] * p["timing_s/update_actor"] / b["timing_s/update_actor"]
                + ref["rest"] * p["rest"] / b["rest"])
        rows = 96 if a == "N1R" else 768
        parts = {"startup_s": p["startup_s"], "steps_s": STEPS * step, "judge_start_s": p["judge_start_s"],
                 "validation_s": 2 * ref["validation_s"], "scoring_s": STEPS * ref["scoring_s"] * rows / 768,
                 "merge_heldout_s": 2 * (p["merge_s"] + p["heldout_load_s"] + HELDOUT_ITEMS * p["heldout_item_s"])}
        res[a] = {"step_s": step, "parts": parts, "total_h": sum(parts.values()) / 3600}
    return res


def main():
    root, ref_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ref = reference(ref_dir)
    pil = {a: pilot_arm(root / a) for a in ARMS}
    arms = project(ref, pil)
    log = (root / "RUN_ALL.txt").read_text()
    arm_walls = sum(float(x) for x in re.findall(r"arm \w+ exit 0 wall_seconds (\d+)", log))
    total_wall = float(re.search(r"pilot wall_seconds (\d+)", log).group(1))
    overhead = max(0.0, total_wall - arm_walls)
    total = sum(v["total_h"] for v in arms.values()) + overhead / 3600
    res = {"reference": ref, "reference_dir": str(ref_dir), "pilot": pil, "arms": arms, "overhead_s": overhead,
           "total_h": total, "limit_h": LIMIT_H, "within_limit": total <= LIMIT_H}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"arms": {a: round(v["total_h"], 2) for a, v in arms.items()}, "total_h": round(total, 2),
                      "within_limit": total <= LIMIT_H}))


if __name__ == "__main__":
    main()
