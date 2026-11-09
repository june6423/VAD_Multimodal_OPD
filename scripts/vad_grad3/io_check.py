"""Disk timing for the projection: write <count> fp32 vectors of <elements> elements with torch.save (as grads.py
writes the vectors), read them back in the block loop of grads.gram (mmap, GPU fp64), then remove them.
Writes IO_CHECK.json. Usage: python io_check.py <dir> <elements> <count>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


def main():
    d, n, k = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    tmp = d / "io_check"
    tmp.mkdir(parents=True, exist_ok=True)
    vec = torch.randn(n, device="cuda").cpu()
    write = []
    for i in range(k):
        t = time.time()
        torch.save(vec, tmp / f"v{i}.pt")
        os.sync()
        write.append(time.time() - t)
    del vec
    for i in range(k):  # drop the written pages from the page cache, so that the read comes from the disk
        fd = os.open(tmp / f"v{i}.pt", os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    block = int(os.environ.get("BLOCK", str(1 << 23)))
    vecs = [torch.load(tmp / f"v{i}.pt", mmap=True) for i in range(k)]
    t = time.time()
    g = torch.zeros(k, k, dtype=torch.float64, device="cuda")
    for s in range(0, n, block):
        x = torch.stack([v[s:s + block].to("cuda", torch.float64) for v in vecs])
        g += x @ x.T
    torch.cuda.synchronize()
    read = time.time() - t
    gb = n * 4 / 1e9
    res = {"elements": n, "vectors": k, "gb_per_vector": gb, "write_seconds": [round(w, 1) for w in write],
           "write_gb_s": round(gb * k / sum(write), 3), "read_seconds": round(read, 1),
           "read_gb_s": round(gb * k / read, 3), "block": block, "diag_finite": bool(torch.isfinite(g.diag()).all())}
    del vecs
    for i in range(k):
        (tmp / f"v{i}.pt").unlink()
    tmp.rmdir()
    (d / "IO_CHECK.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
