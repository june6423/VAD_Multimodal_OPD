"""steppeak.py with dates: windows across 00:00Z and runs longer than 24 h (DT ANSWER on
REPORT-VLM-VAD-PACKFIX-001-002 item 3, VERDICT-VLM-VAD-PACKFIX-001-002 O3).

memwatch.log, the step_log lines and the copy lines have times of day without a date. TASK-VLM-VAD-PACKFIX-001-028:
STEPPEAK.json gave no value for a validation window across 00:00Z. This script gives each time a date:
1. memwatch.log: the samples in file order. The first sample is on the date of the last "== date" header of
   RUN_BOTH.txt (memwatch.sh starts after it). The day increases by 1 when the time of day decreases.
2. Each run starts at the full date and time of its "== train|validate <tag>: ... start" line in RUN_PACKFIX.txt.
   The step_log and copy lines of its console get, in file order, the first date-time at or after the previous one
   (60 s tolerance).
The windows and the values are as in steppeak.py (it stays unchanged), with the date-times in ISO form. Each window
also gives the process group with the largest RSS at the anon+shmem peak (the memwatch "groups" line).
Usage: python steppeak_days.py <task output root> <out json>
"""

import calendar
import json
import re
import sys
import time
from pathlib import Path

from scripts.vad_packfix.steppeak import COPY, KEYS, RESUME, RUN_END, STEP_LOG, seconds

DATE = re.compile(r"== date (\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)Z")
RUN_START = re.compile(r"== (?:train|validate) (\w+): .* start (\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)Z")
SAMPLE = re.compile(r"^(\d\d:\d\d:\d\d) anon ")
FIELD = re.compile(r"(ctr_\w+) ([0-9.]+)")
GROUP = re.compile(r"(\S+) ([0-9.]+)/\d+")


def epoch(day, hms):
    return calendar.timegm(time.strptime(day, "%Y-%m-%d")) + seconds(hms)


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def after(prev, hms):
    """The first date-time with time of day <hms> at or after prev - 60 s."""
    t = prev - prev % 86400 + seconds(hms)
    while t < prev - 60:
        t += 86400
    return t


def samples(path, start_day):
    out, day, last = [], epoch(start_day, "00:00:00"), None
    for line in open(path):
        m = SAMPLE.match(line)
        if m:
            s = seconds(m.group(1))
            if last is not None and s < last:
                day += 86400
            last = s
            out.append({"t": day + s, **{k: float(v) for k, v in FIELD.findall(line)}, "groups": {}})
        elif out and line.strip().startswith("groups"):
            out[-1]["groups"] = {g: float(v) for g, v in GROUP.findall(line)}
    return out


def run_rows(arm, run, start, lines, end, mw):
    """Rows of one run: windows from the run start to each step_log line, and an incomplete window to a failed end."""
    wins, copies, prev, lo, step = [], [], start, start, None
    for line in lines:
        m = RESUME.search(line)
        if m and step is None:
            step = int(m.group(1))
        m = COPY.search(line)
        if m:
            prev = after(prev, m.group(3))
            copies.append((m.group(1), float(m.group(2)), prev))
        m = STEP_LOG.search(line)
        if m:
            prev = after(prev, m.group(2))
            step = int(m.group(1))
            wins.append((step, lo, prev, True))
            lo = prev
    if end is not None and end > lo:
        wins.append((0 if step is None else step + 1, lo, end, False))
    rows = []
    for step, a, b, complete in wins:
        inside = [s for s in mw if a <= s["t"] <= b]
        row = {"arm": arm, "run": run, "step": step, "complete": complete, "start": iso(a), "end": iso(b),
               "wall_seconds": b - a, "samples": len(inside)}
        if inside:
            peak = max(inside, key=lambda s: s["ctr_cur"])
            row.update({"peak_time": iso(peak["t"]), **{k: peak.get(k) for k in KEYS},
                        "oom_kill_first": inside[0].get("ctr_oom_kill"), "oom_kill_last": inside[-1].get("ctr_oom_kill")})
            peak = max(inside, key=lambda s: s["ctr_anon"] + s["ctr_shmem"])
            row["anon_shmem"] = round(peak["ctr_anon"] + peak["ctr_shmem"], 1)
            row["anon_shmem_peak"] = {"time": iso(peak["t"]), **{k: peak.get(k) for k in KEYS[:4]}}
            row["largest_group"] = list(max(peak["groups"].items(), key=lambda kv: kv[1])) if peak["groups"] else None
        row["copies"] = {}
        for name, secs, t in copies:
            if a < t <= b:
                c = row["copies"].setdefault(name, {"count": 0, "seconds": 0.0})
                c["count"] += 1
                c["seconds"] = round(c["seconds"] + secs, 2)
        rows.append(row)
    return rows


def main(root, dest):
    root = Path(root)
    start_day = DATE.findall((root / "RUN_BOTH.txt").read_text())[-1][0]
    mw = samples(root / "memwatch.log", start_day)
    rows = []
    for arm in ("F", "B"):
        run_log = root / arm / "RUN_PACKFIX.txt"
        if not run_log.exists():
            continue
        text = run_log.read_text()
        text = text[max(0, text.rfind("== date")):]  # only the runs of this memwatch.log, as in steppeak.py
        ends = {m.group(1): int(m.group(2)) for m in RUN_END.finditer(text)}
        for m in RUN_START.finditer(text):
            console = root / arm / f"console_{m.group(1)}.log"
            if console.exists():
                start = epoch(m.group(2), m.group(3))
                end = start + ends[m.group(1)] if m.group(1) in ends else None
                rows += run_rows(arm, m.group(1), start, console.read_text(errors="replace").splitlines(), end, mw)
    result = {"rows": rows, "max_ctr_cur": max((r["ctr_cur"] for r in rows if "ctr_cur" in r), default=None),
              "max_anon_shmem": max((r["anon_shmem"] for r in rows if "anon_shmem" in r), default=None)}
    Path(dest).write_text(json.dumps(result, indent=1))
    for r in rows:
        p = r.get("anon_shmem_peak", {})
        print(f"{r['arm']} {r['run']} step {r['step']}{'' if r['complete'] else ' (incomplete)'} "
              f"{r['start']}-{r['end']} wall {r['wall_seconds']} s peak anon+shmem {r.get('anon_shmem')} at "
              f"{p.get('time')} largest group {r.get('largest_group')} "
              f"oom_kill {r.get('oom_kill_first')}->{r.get('oom_kill_last')} samples {r['samples']}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
