"""FULL wall-time projection of both arms from the pilot (HANDOFF-VLM-VAD-PACKFIX-001-R6 Part A item 5, gate G3,
DIRECTIVE-VLM-VAD-PACKFIX-001-007 action 7).

The training runs without the judge. After the training, one judge session runs the fast validations and the
rollout scoring of the arm. For each arm:
  FULL = startup + 20 x step + judge start + validations + 20 x rollout scoring of one step, with
  startup = run1 wall seconds - timing_s/step of the steps of run1 (1 and 2),
  step = the largest timing_s/step of the pilot steps (it includes the checkpoint save of steps 2 and 3),
  judge start = the largest "judge_start_seconds" of RUN_PACKFIX.txt,
  validation of the full set = the measured wall seconds of the pilot step-0 validation of arm F: the sum of its 4
    val_only processes val0_p0 .. val0_p3 on all rows (DIRECTIVE-VLM-VAD-PACKFIX-001-008, pilot item 1). The
    checkpoint validations of the FULL task use the same split, so each gets this time,
  validations = the step-0 validation (only the arm that ran it) + 2 checkpoint validations (steps 10 and 20),
  rollout scoring = the largest "seconds" of ROLLOUT_ACC.json.
Writes PROJECTION.json with the parts, each arm total, the sum and the 80 h limit. The check fails when the step-0
dump of arm F does not have the full validation rows.
Usage: python projection.py <pilot root with F/ and B/> <full validation rows> <out json>
"""

import json
import re
import sys
from pathlib import Path

LIMIT_H = 80.0
VAL0_PART = re.compile(r"validate val0_p\d exit 0 wall_seconds (\d+)")


def full_validation(d: Path, full_rows: int) -> dict:
    """Measured full-set validation of arm F: the wall seconds of the 4 step-0 part processes."""
    walls = [float(x) for x in VAL0_PART.findall((d / "RUN_PACKFIX.txt").read_text())]
    rows = json.loads((d / "VAL_ACC.json").read_text())["steps"]["0"]["rows"]
    if len(walls) != 4 or rows != full_rows:
        raise ValueError(f"step-0 validation of {d}: {len(walls)} part runs, {rows} rows (expected 4, {full_rows})")
    return {"part_wall_s": walls, "rows": rows, "full_s": sum(walls)}


def arm(d: Path, val_full: dict) -> dict:
    sm = json.loads((d / "STEP_METRICS.json").read_text())["steps"]
    ra = json.loads((d / "ROLLOUT_ACC.json").read_text())["steps"]
    log = (d / "RUN_PACKFIX.txt").read_text()
    run1 = int(re.search(r"train run1 exit 0 wall_seconds (\d+)", log).group(1))
    step_s = {int(k): v["timing_s/step"]["value"] for k, v in sm.items() if "timing_s/step" in v}
    step = max(step_s.values())
    startup = max(0.0, run1 - step_s.get(1, 0.0) - step_s.get(2, 0.0))
    judge_start = max(float(x) for x in re.findall(r"judge_start_seconds (\d+)", log))
    ran_val0 = VAL0_PART.search(log) is not None
    score = max(v["seconds"] for v in ra.values())
    validations = (1 + 2 if ran_val0 else 2) * val_full["full_s"]
    total = startup + 20 * step + judge_start + validations + 20 * score
    return {"startup_s": startup, "step_s": step, "judge_start_s": judge_start, "ran_val0": ran_val0,
            "validations_s": validations, "rollout_score_step_s": score,
            "judge_calls_per_step": max(v["judge_calls"] for v in ra.values()), "total_h": total / 3600}


def main():
    root, full_rows, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    val_full = full_validation(root / "F", full_rows)
    arms = {a: arm(root / a, val_full) for a in ("F", "B")}
    total = sum(v["total_h"] for v in arms.values())
    res = {"arms": arms, "val_full": val_full, "total_h": total, "limit_h": LIMIT_H, "within_limit": total <= LIMIT_H,
           "full_val_rows": full_rows}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
