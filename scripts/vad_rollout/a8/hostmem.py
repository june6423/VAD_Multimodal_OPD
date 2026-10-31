"""Host memory of one process tree, sampled every 2 s until the root process ends (HANDOFF-VLM-VAD-ROLLOUT-001-R9 G3).

Usage: python hostmem.py <root pid> <label> <out file>
Appends one line to <out file>: the host MemAvailable at the start and its lowest value, and the peak RSS sum of the
process tree (GB = 10^9 bytes). The RSS sum counts shared pages once per process, so it is an upper bound.
"""

import os
import sys
import time


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def tree_rss_gb(root: int) -> float:
    children, rss = {}, {}
    page = os.sysconf("SC_PAGE_SIZE")
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            ppid = int(open(f"/proc/{name}/stat").read().rsplit(")", 1)[1].split()[1])
            pages = int(open(f"/proc/{name}/statm").read().split()[1])
        except (OSError, ValueError, IndexError):
            continue  # the process ended during the scan
        children.setdefault(ppid, []).append(int(name))
        rss[int(name)] = pages * page
    total, stack = 0, [root]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack += children.get(pid, [])
    return total / 1e9


def alive(pid: int) -> bool:
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def main():
    root, label, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    start = lowest = mem_available_gb()
    peak, samples = 0.0, 0
    while alive(root):
        lowest = min(lowest, mem_available_gb())
        peak = max(peak, tree_rss_gb(root))
        samples += 1
        time.sleep(2)
    with open(out, "a") as f:
        f.write(f"{label} start_mem_available_gb {start:.1f} min_mem_available_gb {lowest:.1f} "
                f"drop_gb {start - lowest:.1f} peak_tree_rss_gb {peak:.1f} samples {samples}\n")


if __name__ == "__main__":
    main()
