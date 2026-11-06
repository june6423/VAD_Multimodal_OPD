"""Container memory and copy times of each pilot step (DIRECTIVE-VLM-VAD-PACKFIX-001-005 action 8).

python steppeak.py <task output root> <STEPPEAK.json>
Reads <root>/memwatch.log (one file for the task) and, for each arm directory F and B, RUN_PACKFIX.txt (the start
time of each train run) and console_<run>.log (the hostmem.py lines "step_log step <s> utc <t>" and
"copy <name> seconds <x> ... utc <t>").
A step window starts at the run start or at the previous step_log line of the same run, and ends at its step_log
line. Step 0 is the validation before training (with the model init). The window of step s holds the generation,
the update, the checkpoint save and the validation of step s. For each window: the memwatch sample with the
largest ctr_cur (ctr_cur, ctr_anon, ctr_shmem, ctr_file, ctr_kernel), ctr_oom_kill at the first and last sample,
the wall seconds, and the count and sum of seconds of each copy name. Times are UTC HH:MM:SS of one day.
"""

import json
import re
import sys
from pathlib import Path

from scripts.vad_packfix.mempeak import parse

RUN_START = re.compile(r"== train (\w+): .* start \d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)Z")
STEP_LOG = re.compile(r"VAD_HOSTMEM pid=\d+ step_log step (\d+) utc (\d\d:\d\d:\d\d)")
COPY = re.compile(r"VAD_HOSTMEM pid=\d+ copy (\w+) seconds ([0-9.]+).* utc (\d\d:\d\d:\d\d)")
KEYS = ("ctr_cur", "ctr_anon", "ctr_shmem", "ctr_file", "ctr_kernel")


def seconds(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return 3600 * h + 60 * m + s


def windows(run_start, console_lines):
    """[(step, start, end)] from the run start and the step_log lines of one run."""
    out, start = [], run_start
    for line in console_lines:
        m = STEP_LOG.search(line)
        if m:
            out.append((int(m.group(1)), start, m.group(2)))
            start = m.group(2)
    return out


def step_rows(arm, run, wins, samples, console_lines):
    copies = [(m.group(1), float(m.group(2)), seconds(m.group(3))) for m in map(COPY.search, console_lines) if m]
    rows = []
    for step, start, end in wins:
        lo, hi = seconds(start), seconds(end)
        inside = [s for s in samples if lo <= seconds(s["time"]) <= hi]
        row = {"arm": arm, "run": run, "step": step, "start": start, "end": end, "wall_seconds": hi - lo,
               "samples": len(inside)}
        if inside:
            peak = max(inside, key=lambda s: s["ctr_cur"])
            row.update({"peak_time": peak["time"], **{k: peak.get(k) for k in KEYS},
                        "oom_kill_first": inside[0].get("ctr_oom_kill"), "oom_kill_last": inside[-1].get("ctr_oom_kill")})
        row["copies"] = {}
        for name, secs, t in copies:
            if lo < t <= hi:
                c = row["copies"].setdefault(name, {"count": 0, "seconds": 0.0})
                c["count"] += 1
                c["seconds"] = round(c["seconds"] + secs, 2)
        rows.append(row)
    return rows


def main(root, dest):
    root = Path(root)
    samples = parse(root / "memwatch.log")
    rows = []
    for arm in ("F", "B"):
        run_log = root / arm / "RUN_PACKFIX.txt"
        if not run_log.exists():
            continue
        for m in RUN_START.finditer(run_log.read_text()):
            console = root / arm / f"console_{m.group(1)}.log"
            if console.exists():
                lines = console.read_text(errors="replace").splitlines()
                rows += step_rows(arm, m.group(1), windows(m.group(2), lines), samples, lines)
    result = {"rows": rows, "max_ctr_cur": max((r["ctr_cur"] for r in rows if "ctr_cur" in r), default=None),
              "limit_note": "STATUS to DT before the FULL task if a step peak of ctr_cur is above 134 GiB"}
    Path(dest).write_text(json.dumps(result, indent=1))
    for r in rows:
        print(f"{r['arm']} {r['run']} step {r['step']} {r['start']}-{r['end']} wall {r['wall_seconds']} s "
              f"peak ctr_cur {r.get('ctr_cur')} anon {r.get('ctr_anon')} shmem {r.get('ctr_shmem')} "
              f"oom_kill {r.get('oom_kill_first')}->{r.get('oom_kill_last')} copies {r['copies']}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
