"""FULL wall-time projection of both arms from the pilot (HANDOFF-VLM-VAD-PACKFIX-001-R6 Part A item 5, gate G3,
DIRECTIVE-VLM-VAD-PACKFIX-001-007 action 7).

The training runs without the judge. After the training, one judge session runs the fast validations and the
rollout scoring of the arm. For each arm:
  FULL = startup + 20 x step + judge start + validations + 20 x rollout scoring of one step, with
  startup = run1 wall seconds - timing_s/step of the steps of run1 (1 and 2),
  step = the largest timing_s/step of the pilot steps (it includes the checkpoint save of steps 2 and 3),
  judge start = the largest "judge_start_seconds" of RUN_PACKFIX.txt,
  validation of the full set = (val run wall - val part) + val part x full rows / pilot rows. The val part of a
    validation run starts at its last "copy offload_model" line (the weight sync to the rollout engine) before the
    step_log line, and ends at the step_log line (console_<tag>.log),
  validations = the step-0 validation (only the arm that ran it) + 2 checkpoint validations (steps 10 and 20),
  rollout scoring = the largest "seconds" of ROLLOUT_ACC.json.
Writes PROJECTION.json with the parts, each arm total, the sum and the 80 h limit.
Usage: python projection.py <pilot root with F/ and B/> <full validation rows> <out json>
"""

import json
import re
import sys
from pathlib import Path

LIMIT_H = 80.0
COPY_OFFLOAD = re.compile(r"VAD_HOSTMEM pid=\d+ copy offload_model seconds [0-9.]+.* utc (\d\d:\d\d:\d\d)")
STEP_LOG = re.compile(r"VAD_HOSTMEM pid=\d+ step_log step \d+ utc (\d\d:\d\d:\d\d)")


def seconds(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return 3600 * h + 60 * m + s


def val_part(console: Path) -> float:
    """Seconds from the last offload_model copy before the step_log line to the step_log line."""
    last_offload = None
    for line in console.read_text(errors="replace").splitlines():
        m = COPY_OFFLOAD.search(line)
        if m:
            last_offload = m.group(1)
        m = STEP_LOG.search(line)
        if m and last_offload is not None:
            return float((seconds(m.group(1)) - seconds(last_offload)) % 86400)
    raise ValueError(f"no validation window in {console}")


def validation(d: Path, log: str, tag: str, scale: float):
    m = re.search(rf"validate {tag} exit 0 wall_seconds (\d+)", log)
    if not m:
        return None
    wall, part = float(m.group(1)), val_part(d / f"console_{tag}.log")
    return {"pilot_wall_s": wall, "pilot_val_part_s": part, "full_s": wall - part + part * scale}


def arm(d: Path, full_rows: int) -> dict:
    sm = json.loads((d / "STEP_METRICS.json").read_text())["steps"]
    va = json.loads((d / "VAL_ACC.json").read_text())["steps"]
    ra = json.loads((d / "ROLLOUT_ACC.json").read_text())["steps"]
    log = (d / "RUN_PACKFIX.txt").read_text()
    run1 = int(re.search(r"train run1 exit 0 wall_seconds (\d+)", log).group(1))
    step_s = {int(k): v["timing_s/step"]["value"] for k, v in sm.items() if "timing_s/step" in v}
    step = max(step_s.values())
    startup = max(0.0, run1 - step_s.get(1, 0.0) - step_s.get(2, 0.0))
    judge_start = max(float(x) for x in re.findall(r"judge_start_seconds (\d+)", log))
    pilot_rows = max(v["rows"] for v in va.values())
    scale = full_rows / pilot_rows
    val0 = validation(d, log, "val0", scale)
    ckpt_tags = sorted(set(re.findall(r"validate (val[1-9]\d*) exit 0", log)))
    val_ckpt = max((validation(d, log, t, scale) for t in ckpt_tags), key=lambda v: v["full_s"])
    score = max(v["seconds"] for v in ra.values())
    validations = (val0["full_s"] if val0 else 0.0) + 2 * val_ckpt["full_s"]
    total = startup + 20 * step + judge_start + validations + 20 * score
    return {"startup_s": startup, "step_s": step, "judge_start_s": judge_start, "pilot_val_rows": pilot_rows,
            "val0": val0, "val_checkpoint": val_ckpt, "validations_s": validations, "rollout_score_step_s": score,
            "judge_calls_per_step": max(v["judge_calls"] for v in ra.values()), "total_h": total / 3600}


def main():
    root, full_rows, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    arms = {a: arm(root / a, full_rows) for a in ("F", "B")}
    total = sum(v["total_h"] for v in arms.values())
    res = {"arms": arms, "total_h": total, "limit_h": LIMIT_H, "within_limit": total <= LIMIT_H,
           "full_val_rows": full_rows}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
