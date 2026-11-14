"""STEPPEAK.json of VLM-VAD-VIEW-002: scripts/vad_packfix/steppeak_days.py for the arms of run_all.sh.

steppeak_days.py reads the arms F and B (RUN_PACKFIX.txt, header RUN_BOTH.txt). TASK-VLM-VAD-VIEW-002-003 thus got a
STEPPEAK.json without rows. This script uses the functions of steppeak_days.py without change, with the header file
RUN_ALL.txt, the arms N1R, N1V8, B and T8, and RUN_VIEW2.txt. Each train or validate run of an arm gives its windows
(the step windows and the validation windows, with dates) and the peaks of ctr_cur and of ctr_anon + ctr_shmem.
Usage: python steppeak.py <task output root> <out json>. Exit 1 when no window is found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.vad_packfix.steppeak_days import DATE, RUN_END, RUN_START, epoch, run_rows, samples

ARMS = ("N1R", "N1V8", "B", "T8")


def main(root, dest) -> int:
    root = Path(root)
    start_day = DATE.findall((root / "RUN_ALL.txt").read_text())[-1][0]
    mw = samples(root / "memwatch.log", start_day)
    rows = []
    for arm in ARMS:
        run_log = root / arm / "RUN_VIEW2.txt"
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
              "max_anon_shmem": max((r["anon_shmem"] for r in rows if "anon_shmem" in r), default=None),
              "limit_note": "STATUS to DT if a step peak of ctr_anon + ctr_shmem is above 134 GiB"}
    Path(dest).write_text(json.dumps(result, indent=1))
    for r in rows:
        print(f"{r['arm']} {r['run']} step {r['step']}{'' if r['complete'] else ' (incomplete)'} "
              f"{r['start']}-{r['end']} wall {r['wall_seconds']} s peak anon+shmem {r.get('anon_shmem')} "
              f"oom_kill {r.get('oom_kill_first')}->{r.get('oom_kill_last')} samples {r['samples']}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
