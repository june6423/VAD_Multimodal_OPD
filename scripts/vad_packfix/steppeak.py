"""Container memory and copy times of each pilot step (DIRECTIVE-VLM-VAD-PACKFIX-001-005 action 8).

python steppeak.py <task output root> <STEPPEAK.json>
Reads <root>/memwatch.log (one file for the task) and, for each arm directory F and B, RUN_PACKFIX.txt (the start
time of each train run) and console_<run>.log (the hostmem.py lines "step_log step <s> utc <t>" and
"copy <name> seconds <x> ... utc <t>").
A step window starts at the run start or at the previous step_log line of the same run, and ends at its step_log
line. The window of step s holds the generation, the update and the checkpoint save of step s (the window of the
first step of a run also holds the model init). DIRECTIVE-VLM-VAD-PACKFIX-001-007: each fast validation runs in its own
process ("== validate <tag>: ... start" in RUN_PACKFIX.txt). Its window (run <tag>, the step of the validated model)
holds the model init, the checkpoint load and the validation. For each window: the memwatch sample with the
largest ctr_cur (ctr_cur, ctr_anon, ctr_shmem, ctr_file, ctr_kernel), ctr_oom_kill at the first and last sample,
the wall seconds, and the count and sum of seconds of each copy name. Times are UTC HH:MM:SS of one day.
DT ANSWER on TASK-VLM-VAD-PACKFIX-001-022: each window also gives the sample with the largest ctr_anon + ctr_shmem
(the 134 GiB threshold applies to it, because ctr_cur counts page cache). A run that ends without the step_log line
of its last step (an error) gets one more window, "complete": false, from its last step_log line to its end.
"""

import json
import re
import sys
from pathlib import Path

from scripts.vad_packfix.mempeak import parse

RUN_START = re.compile(r"== (?:train|validate) (\w+): .* start \d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)Z")
STEP_LOG = re.compile(r"VAD_HOSTMEM pid=\d+ step_log step (\d+) utc (\d\d:\d\d:\d\d)")
COPY = re.compile(r"VAD_HOSTMEM pid=\d+ copy (\w+) seconds ([0-9.]+).* utc (\d\d:\d\d:\d\d)")
RUN_END = re.compile(r"(?:train|validate) (\w+) exit [1-9]\d* wall_seconds (\d+)")  # failed runs only
RESUME = re.compile(r"Resuming from \S*global_step_(\d+)")
KEYS = ("ctr_cur", "ctr_anon", "ctr_shmem", "ctr_file", "ctr_kernel")


def seconds(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return 3600 * h + 60 * m + s


def hms(secs):
    return f"{secs // 3600 % 24:02d}:{secs // 60 % 60:02d}:{secs % 60:02d}"


def windows(run_start, console_lines, run_end=None):
    """[(step, start, end, complete)] from the run start, the step_log lines of one run, and the run end if known."""
    out, start, step = [], run_start, None
    for line in console_lines:
        m = RESUME.search(line)
        if m and step is None:  # a resumed run without a step_log line fails in the step after the checkpoint
            step = int(m.group(1))
        m = STEP_LOG.search(line)
        if m:
            step = int(m.group(1))
            out.append((step, start, m.group(2), True))
            start = m.group(2)
    if run_end is not None and seconds(run_end) > seconds(start):
        out.append((0 if step is None else step + 1, start, run_end, False))
    return out


def anon_shmem(sample):
    return round(sample["ctr_anon"] + sample["ctr_shmem"], 1)


def step_rows(arm, run, wins, samples, console_lines):
    copies = [(m.group(1), float(m.group(2)), seconds(m.group(3))) for m in map(COPY.search, console_lines) if m]
    rows = []
    for step, start, end, complete in wins:
        lo, hi = seconds(start), seconds(end)
        inside = [s for s in samples if lo <= seconds(s["time"]) <= hi]
        row = {"arm": arm, "run": run, "step": step, "complete": complete, "start": start, "end": end,
               "wall_seconds": hi - lo, "samples": len(inside)}
        if inside:
            peak = max(inside, key=lambda s: s["ctr_cur"])
            row.update({"peak_time": peak["time"], **{k: peak.get(k) for k in KEYS},
                        "oom_kill_first": inside[0].get("ctr_oom_kill"), "oom_kill_last": inside[-1].get("ctr_oom_kill")})
            peak = max(inside, key=anon_shmem)
            row["anon_shmem"] = anon_shmem(peak)
            row["anon_shmem_peak"] = {"time": peak["time"], **{k: peak.get(k) for k in KEYS[:4]}}
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
        text = run_log.read_text()
        # A restarted task appends to RUN_PACKFIX.txt, and memwatch.log holds only this run (run_both.sh renames
        # the earlier log). The runs before the last "== date" header of run_packfix.sh are not in this memwatch.log.
        text = text[max(0, text.rfind("== date")):]
        ends = {m.group(1): int(m.group(2)) for m in RUN_END.finditer(text)}
        for m in RUN_START.finditer(text):
            console = root / arm / f"console_{m.group(1)}.log"
            if console.exists():
                lines = console.read_text(errors="replace").splitlines()
                end = hms(seconds(m.group(2)) + ends[m.group(1)]) if m.group(1) in ends else None
                rows += step_rows(arm, m.group(1), windows(m.group(2), lines, end), samples, lines)
    result = {"rows": rows, "max_ctr_cur": max((r["ctr_cur"] for r in rows if "ctr_cur" in r), default=None),
              "max_anon_shmem": max((r["anon_shmem"] for r in rows if "anon_shmem" in r), default=None),
              "limit_note": "STATUS to DT before the FULL task if a step peak of ctr_anon + ctr_shmem is above 134 GiB"}
    Path(dest).write_text(json.dumps(result, indent=1))
    for r in rows:
        p = r.get("anon_shmem_peak", {})
        print(f"{r['arm']} {r['run']} step {r['step']}{'' if r['complete'] else ' (incomplete)'} "
              f"{r['start']}-{r['end']} wall {r['wall_seconds']} s peak anon+shmem {r.get('anon_shmem')} at "
              f"{p.get('time')} (anon {p.get('ctr_anon')} shmem {p.get('ctr_shmem')} file {p.get('ctr_file')} "
              f"cur {p.get('ctr_cur')}) peak ctr_cur {r.get('ctr_cur')} "
              f"oom_kill {r.get('oom_kill_first')}->{r.get('oom_kill_last')} copies {r['copies']}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
