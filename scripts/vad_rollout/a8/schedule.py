"""Run A8 jobs on a pool of GPUs, one job per GPU at a time (HANDOFF-VLM-VAD-ROLLOUT-001-R9, Part B items 3 and 4).

Usage: python schedule.py <jobs file> <gpus, for example 0,1,2,3,4,5> <log dir> [min MemAvailable GB, default 100]
The jobs file has one job per line: <phase>\t<name>\t<command>. A phase starts after every job of the earlier
phases ended with exit 0. A failed job stops the run. R9 stop condition 2: when MemAvailable falls below the
limit, every job gets SIGTERM and the run exits 3. <log dir>/SCHEDULE.txt records each job and the lowest
MemAvailable.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def main():
    jobs_file, gpus, log_dir = sys.argv[1], sys.argv[2].split(","), Path(sys.argv[3])
    limit = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
    log_dir.mkdir(parents=True, exist_ok=True)
    phases = defaultdict(list)
    for line in open(jobs_file):
        if line.strip():
            phase, name, command = line.rstrip("\n").split("\t")
            phases[int(phase)].append((name, command))
    record = open(log_dir / "SCHEDULE.txt", "a")
    lowest = mem_available_gb()
    running: dict[str, tuple] = {}

    def stop_all(code: int, why: str):
        for proc, *_ in running.values():
            os.killpg(proc.pid, signal.SIGTERM)
        record.write(f"STOP {why} lowest_mem_available_gb {lowest:.1f}\n")
        record.close()
        sys.exit(code)

    for phase in sorted(phases):
        queue = list(phases[phase])
        while queue or running:
            for gpu in gpus:
                if gpu not in running and queue:
                    name, command = queue.pop(0)
                    log = open(log_dir / f"{name}.log", "w")
                    proc = subprocess.Popen(shlex.split(command), stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, start_new_session=True)
                    running[gpu] = (proc, name, time.time())
            time.sleep(10)
            lowest = min(lowest, mem_available_gb())
            if lowest < limit:
                stop_all(3, f"R9 stop condition 2: MemAvailable {lowest:.1f} GB < {limit} GB")
            for gpu, (proc, name, start) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del running[gpu]
                record.write(f"{phase}\t{name}\tgpu {gpu}\tseconds {time.time() - start:.0f}\texit {code}\n")
                record.flush()
                if code != 0:
                    stop_all(1, f"job {name} exit {code}")
    record.write(f"DONE lowest_mem_available_gb {lowest:.1f}\n")
    record.close()


if __name__ == "__main__":
    main()
