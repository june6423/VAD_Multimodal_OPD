"""Gram files and fold means of HANDOFF-VLM-VAD-VIEW-001-R2 Part A item 6.

gram: GRAM_<ck>.json holds the fp64 inner products raw <x, y>, M1 <x, D y> and M2 <x, D^2 y> of all vectors of
  Definition 9 at the checkpoint: G__f<f>__<view> (vectors/O__B8-f<f>@<view>__total.pt), H__skill__h<h> and
  H__fmt__h<h> (vectors/H__*.pt), and the displacements D__A1_0_80 and D__A1_80_130. D is the GRAD-003 metric
  (scripts/vad_grad3/grads.py metric_d, Definition 8.1 of HANDOFF-VLM-VAD-GRAD-003-R3). The same block pass writes
  the fold means MEAN__<view> = mean over f of G__f<f>__<view> and MEAN__V = their mean over the views (fp32 .npy) for
  the line search. Then GRAM_<ck>.sha256.
remove: after the line search, remove the vector files and the fold means of the checkpoint; VECTORS_REMOVED.json.

Usage: python vgram.py gram <ck dir> <displacement dir> <V file> <G3_PARAMS.txt> <checkpoint>
       python vgram.py remove <ck dir>
Env: BLOCK (8388608), READERS (16).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vad_grad3"))
DISPLACEMENTS = ("D__A1_0_80", "D__A1_80_130")


def vector_name(path: Path) -> str | None:
    """Gram name of a vector file, None for other files."""
    stem = path.stem
    if stem.startswith("O__B8-f") and stem.endswith("__total") and "@" in stem:
        key = stem[len("O__"):-len("__total")]
        batch, view = key.split("@")
        return f"G__f{batch[len('B8-f'):]}__{view}"
    if stem.startswith("H__"):
        return stem
    return None


def inputs(ck_dir: Path, disp_dir: Path) -> tuple[list[Path], list[str]]:
    files, names = [], []
    for p in sorted((ck_dir / "vectors").glob("*.pt")):
        n = vector_name(p)
        if n is not None:
            files.append(p)
            names.append(n)
    for d in DISPLACEMENTS:
        files.append(disp_dir / f"{d}.pt")
        names.append(d)
    return files, names


def fold_mean_groups(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        if n.startswith("G__"):
            groups.setdefault(n.split("__")[2], []).append(i)
    return groups


def gram(argv: list[str]):
    import grads as g3
    import ckpt
    import torch

    ck_dir, disp_dir, v_file, g3_params, checkpoint = argv
    ck_dir, disp_dir = Path(ck_dir), Path(disp_dir)
    files, names = inputs(ck_dir, disp_dir)
    vecs = [torch.load(f, mmap=True) for f in files]
    n = vecs[0].numel()
    if any(v.numel() != n or v.dtype != torch.float32 for v in vecs):
        raise SystemExit("STOP: vector files differ in length or are not fp32")
    d, d_info = g3.metric_d(v_file, ckpt.grad_set(g3_params))
    if d.numel() != n:
        raise SystemExit(f"STOP: D has {d.numel()} elements, the vectors {n}")
    groups = fold_mean_groups(names)
    means = {v: np.lib.format.open_memmap(ck_dir / f"MEAN__{v}.npy.tmp", mode="w+", dtype=np.float32, shape=(n,))
             for v in groups}
    means["V"] = np.lib.format.open_memmap(ck_dir / "MEAN__V.npy.tmp", mode="w+", dtype=np.float32, shape=(n,))
    block = int(os.environ.get("BLOCK", str(1 << 23)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = len(files)
    fds = [os.open(f, os.O_RDONLY) for f in files]
    offs = [g3.data_offset(f, v) for f, v in zip(files, vecs)]
    for j, v in enumerate(vecs):
        for s in (0, max(0, n - 1024)):
            if not torch.equal(g3.read_into(fds[j], offs[j], s, np.empty(min(1024, n - s), np.float32)), v[s:s + 1024]):
                raise SystemExit(f"STOP: the pread of {files[j].name} differs from torch.load")
    del vecs
    t0 = time.time()
    buf = np.empty((k, min(block, n)), np.float32)
    g = {m: torch.zeros(k, k, dtype=torch.float64, device=dev) for m in ("raw", "M1", "M2")}
    with ThreadPoolExecutor(int(os.environ.get("READERS", "16"))) as ex:
        for s in range(0, n, block):
            m_s = min(block, n - s)
            list(ex.map(lambda j: g3.read_into(fds[j], offs[j], s, buf[j, :m_s]), range(k)))
            x = torch.from_numpy(buf[:, :m_s]).to(dev, torch.float64)
            db = d[s:s + block].to(dev, torch.float64)
            xd = x * db
            g["raw"] += x @ x.T
            g["M1"] += xd @ x.T
            g["M2"] += xd @ xd.T
            fm = {v: x[idx].mean(0) for v, idx in groups.items()}
            for v, val in fm.items():
                means[v][s:s + m_s] = val.float().cpu().numpy()
            means["V"][s:s + m_s] = torch.stack(list(fm.values())).mean(0).float().cpu().numpy()
            del x, xd, db, fm
    for fd in fds:
        os.close(fd)
    for v, mm in means.items():
        mm.flush()
        del mm
        os.replace(ck_dir / f"MEAN__{v}.npy.tmp", ck_dir / f"MEAN__{v}.npy")
    res = {"checkpoint": checkpoint, "names": names, "files": [str(f) for f in files], "elements": n, "block": block,
           "metric": d_info, **{m: g[m].cpu().tolist() for m in g},
           "fold_means": {v: [names[i] for i in idx] for v, idx in groups.items()}, "seconds": round(time.time() - t0, 1)}
    path = ck_dir / f"GRAM_{checkpoint}.json"
    path.write_text(json.dumps(res))
    (ck_dir / f"GRAM_{checkpoint}.sha256").write_text(f"{g3.sha256_file(path)}  {path.name}\n")
    print(f"gram {checkpoint}: {k} vectors of {n} elements, {res['seconds']} s")


def remove(argv: list[str]):
    import grads as g3

    ck_dir = Path(argv[0])
    gram_files = sorted(ck_dir.glob("GRAM_*.json"))
    if len(gram_files) != 1:
        raise SystemExit(f"STOP: {ck_dir} has {len(gram_files)} Gram files")
    removed = []
    for p in sorted((ck_dir / "vectors").glob("*.pt")) + sorted(ck_dir.glob("MEAN__*.npy")):
        p.unlink()
        removed.append(p.name)
    left = [p.name for p in (ck_dir / "vectors").glob("*")] + [p.name for p in ck_dir.glob("MEAN__*")]
    g3.write_json(ck_dir / "VECTORS_REMOVED.json", {"at": g3.now(), "gram": gram_files[0].name,
                                                    "gram_sha256": g3.sha256_file(gram_files[0]),
                                                    "removed": removed, "left": left})
    print(f"removed {len(removed)} files; left {left}")


def main():
    {"gram": gram, "remove": remove}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
