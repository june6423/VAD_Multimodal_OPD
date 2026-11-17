"""STEPPEAK.json of VLM-VAD-VIEW-002: scripts/vad_packfix/steppeak_days.py for the arms of run_all.sh.

steppeak_days.py reads the arms F and B (RUN_PACKFIX.txt, header RUN_BOTH.txt). TASK-VLM-VAD-VIEW-002-003 thus got a
STEPPEAK.json without rows. This script uses the functions of steppeak_days.py without change, with the header file
RUN_ALL.txt, the arms N1R, N1V8, B and T8, and RUN_VIEW2.txt. Each train or validate run of an arm gives its windows
(the step windows and the validation windows, with dates) and the peaks of ctr_cur and of ctr_anon + ctr_shmem.
DIRECTIVE-VLM-VAD-VIEW-002-005 action 6: each row also gets "peak_groups", the RSS (GiB) of the process groups
AgentLoopW, TaskRunner, WorkerDict, vLLMHttpSe and EngineCore and the /dev/shm use of the Ray object store
(memwatch devshm_used) at the anon+shmem peak sample. The same values go to <out json stem>_GROUPS.tsv.
Usage: python steppeak.py <task output root> <out json>. Exit 1 when no window is found.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from scripts.vad_packfix.steppeak_days import DATE, RUN_END, RUN_START, SAMPLE, epoch, iso, run_rows, samples

ARMS = ("N1R", "N1V8", "B", "T8")
GROUPS = {"AgentLoopW": "ray::AgentLoopW", "TaskRunner": "ray::TaskRunner", "WorkerDict": "ray::WorkerDict",
          "vLLMHttpSe": "ray::vLLMHttpSe", "EngineCore": "EngineCore(ray::vLLMHttpSe)"}
DEVSHM = re.compile(r"devshm_used ([0-9.]+)")


def devshm(path):
    """devshm_used of each memwatch sample line, in the order of samples() (None when the line has no value)."""
    out = []
    for line in open(path):
        if SAMPLE.match(line):
            m = DEVSHM.search(line)
            out.append(float(m.group(1)) if m else None)
    return out


def peak_groups(rows, mw) -> None:
    """Add the group RSS and the object store /dev/shm use at the anon+shmem peak sample to each row."""
    at = {iso(s["t"]): s for s in mw}
    for r in rows:
        s = at.get(r.get("anon_shmem_peak", {}).get("time"))
        if s is not None:
            r["peak_groups"] = {**{k: s["groups"].get(g) for k, g in GROUPS.items()},
                                "object_store_devshm": s.get("devshm_used")}


def main(root, dest) -> int:
    root = Path(root)
    start_day = DATE.findall((root / "RUN_ALL.txt").read_text())[-1][0]
    mw = samples(root / "memwatch.log", start_day)
    for s, d in zip(mw, devshm(root / "memwatch.log")):
        s["devshm_used"] = d
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
    peak_groups(rows, mw)
    cols = list(GROUPS) + ["object_store_devshm"]
    table = ["\t".join(["arm", "run", "step", "complete", "peak_time", "anon_shmem"] + cols)]
    for r in rows:
        g = r.get("peak_groups", {})
        table.append("\t".join(str(x) for x in [r["arm"], r["run"], r["step"], r["complete"],
                                                r.get("anon_shmem_peak", {}).get("time"), r.get("anon_shmem")]
                               + [g.get(c) for c in cols]))
    Path(dest).with_name(Path(dest).stem + "_GROUPS.tsv").write_text("\n".join(table) + "\n")
    result = {"rows": rows, "max_ctr_cur": max((r["ctr_cur"] for r in rows if "ctr_cur" in r), default=None),
              "max_anon_shmem": max((r["anon_shmem"] for r in rows if "anon_shmem" in r), default=None),
              "limit_note": "STATUS to DT if a step peak of ctr_anon + ctr_shmem is above 134 GiB"}
    Path(dest).write_text(json.dumps(result, indent=1))
    for r in rows:
        print(f"{r['arm']} {r['run']} step {r['step']}{'' if r['complete'] else ' (incomplete)'} "
              f"{r['start']}-{r['end']} wall {r['wall_seconds']} s peak anon+shmem {r.get('anon_shmem')} "
              f"oom_kill {r.get('oom_kill_first')}->{r.get('oom_kill_last')} samples {r['samples']} "
              f"groups {r.get('peak_groups')}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
