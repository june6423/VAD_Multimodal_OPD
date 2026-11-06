"""FULL wall-time projection of both arms from the pilot (HANDOFF-VLM-VAD-PACKFIX-001-R2 Part A item 5, gate G3).

For each arm: FULL = startup + 20 x step + 3 x validation of the full set + 2 x checkpoint save, with
  startup = run1 wall seconds - step 1 - its validations - its save (RUN_PACKFIX.txt and STEP_METRICS.json),
  step = the largest timing_s/step of the pilot steps,
  validation of the full set = the largest pilot timing_s/testing x full rows / pilot rows,
  save = the largest timing_s/save_checkpoint.
Writes PROJECTION.json with the parts, each arm total, the sum and the 80 h limit.
Usage: python projection.py <pilot root with F/ and B/> <full validation rows> <out json>
"""

import json
import re
import sys
from pathlib import Path

LIMIT_H = 80.0


def arm(d: Path, full_rows: int) -> dict:
    sm = json.loads((d / "STEP_METRICS.json").read_text())["steps"]
    va = json.loads((d / "VAL_ACC.json").read_text())["steps"]
    log = (d / "RUN_PACKFIX.txt").read_text()
    run1 = int(re.search(r"train run1 exit \d+ wall_seconds (\d+)", log).group(1))
    vals = lambda key: [v[key]["value"] for v in sm.values() if key in v]  # noqa: E731
    step = max(vals("timing_s/step"))
    tests = vals("timing_s/testing")
    test = max(tests) if tests else 0.0
    saves = vals("timing_s/save_checkpoint")
    save = max(saves) if saves else 0.0
    pilot_rows = max(v["rows"] for v in va.values())
    s1 = sm["1"]
    run1_tests = sum(v["timing_s/testing"]["value"] for k, v in sm.items() if k in ("0", "1") and "timing_s/testing" in v)
    startup = max(0.0, run1 - s1["timing_s/step"]["value"] - run1_tests
                  - s1.get("timing_s/save_checkpoint", {"value": 0.0})["value"])
    val_full = test * full_rows / pilot_rows
    total = startup + 20 * step + 3 * val_full + 2 * save
    return {"startup_s": startup, "step_s": step, "pilot_val_s": test, "pilot_val_rows": pilot_rows,
            "full_val_s": val_full, "save_s": save, "total_h": total / 3600}


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
