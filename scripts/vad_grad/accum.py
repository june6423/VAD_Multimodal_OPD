"""Bookkeeping of the moment pass (HANDOFF-VLM-VAD-GRAD-001-R1 Part A item 5; DIRECTIVE-VLM-VAD-GRAD-001-002 D1, D3).

Device-independent: a vector is a list of tensors (one per parameter of the gradient set). moments.py calls these
functions on the GPU; the tests call them on CPU with synthetic vectors.
- fold_of: the 375 prompts in K=4 disjoint folds (fixed seed).
- Store: the 28 fp32 host accumulators S|fold|class|flag (24) and A|fold (4), with versioned saves and a resume.
- sibling Gram: all dot products of the siblings of one prompt, raw and with the diagonal metric D. Group sums,
  pair cosines and squared norms of every subset follow from it.
- write_gram: the 28 x 28 dot products of the accumulators, raw and with D, in fp64, computed tensor by tensor.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

K = 4
FOLD_SEED = 20261103
CLASSES = ("C", "I")
FLAGS = ("f1", "f0", "fx")


def fold_of(prompts: list[int], k: int = K, seed: int = FOLD_SEED) -> dict[int, int]:
    order = sorted(prompts)
    random.Random(seed).shuffle(order)
    return {p: i % k for i, p in enumerate(order)}


def s_name(fold: int, cls: str, flag: str) -> str:
    return f"S|{fold}|{cls}|{flag}"


def a_name(fold: int) -> str:
    return f"A|{fold}"


def vector_names(k: int = K) -> list[str]:
    return [s_name(a, c, f) for a in range(k) for c in CLASSES for f in FLAGS] + [a_name(a) for a in range(k)]


def fold_names(fold: int) -> list[str]:
    return [s_name(fold, c, f) for c in CLASSES for f in FLAGS] + [a_name(fold)]


def dot(a: list[torch.Tensor], b: list[torch.Tensor], d: list[torch.Tensor] | None = None) -> float:
    """sum_k a_k b_k (d_k), accumulated in fp64 tensor by tensor."""
    total = torch.zeros((), dtype=torch.float64, device=a[0].device)
    for i, (x, y) in enumerate(zip(a, b)):
        x64 = x.double()
        if d is not None:
            x64 = x64 * d[i].double()
        total += torch.dot(x64.reshape(-1), y.double().reshape(-1))
    return float(total)


def sibling_gram(grads: list[list[torch.Tensor]], d: list[torch.Tensor]) -> tuple[list[list[float]], list[list[float]]]:
    """All dot products of the siblings, raw and with D: per parameter tensor one fp64 product of the stacked rows."""
    n = len(grads)
    dev = grads[0][0].device
    raw = torch.zeros((n, n), dtype=torch.float64, device=dev)
    met = torch.zeros((n, n), dtype=torch.float64, device=dev)
    for t in range(len(grads[0])):
        m = torch.stack([g[t].reshape(-1) for g in grads]).double()
        raw += m @ m.T
        met += (m * d[t].reshape(-1).double()) @ m.T
        del m
    return raw.tolist(), met.tolist()


def weighted_sum(vecs: list[list[torch.Tensor]], weights: list[float], out: list[torch.Tensor]) -> list[torch.Tensor]:
    """out = sum_i w_i vecs_i in fp32 (out is overwritten)."""
    for o in out:
        o.zero_()
    for v, w in zip(vecs, weights):
        if w != 0.0:
            for o, x in zip(out, v):
                o.add_(x.float(), alpha=w)
    return out


class Store:
    """fp32 host accumulators. A fold allocates its 7 vectors at its first add."""

    def __init__(self, shapes: list[torch.Size], pin: bool = False):
        self.shapes, self.pin = shapes, pin
        self.vec: dict[str, list[torch.Tensor]] = {}
        self.count: dict[str, int] = {}
        self.dirty: set[int] = set()
        self.staging = None

    def _alloc(self, name: str) -> list[torch.Tensor]:
        if name not in self.vec:
            self.vec[name] = [torch.zeros(s, dtype=torch.float32, pin_memory=self.pin) for s in self.shapes]
            self.count[name] = 0
        return self.vec[name]

    def add(self, name: str, src: list[torch.Tensor], count: int):
        """vec[name] += src (src may be on the GPU; it goes through one reusable host staging vector)."""
        acc = self._alloc(name)
        if src[0].device.type != "cpu":
            if self.staging is None:  # pinned: the copy from the GPU runs at full speed
                self.staging = [torch.empty(s, dtype=torch.float32, pin_memory=True) for s in self.shapes]
            for h, g in zip(self.staging, src):
                h.copy_(g)
            src = self.staging
        for a, x in zip(acc, src):
            a.add_(x)
        self.count[name] += count
        self.dirty.add(int(name.split("|")[1]))

    def add_count(self, name: str, count: int):
        self._alloc(name)
        self.count[name] += count
        self.dirty.add(int(name.split("|")[1]))

    def get(self, name: str) -> list[torch.Tensor]:
        return self._alloc(name)

    def save(self, directory: Path, seq: int, folds: set[int]) -> dict[str, str]:
        """Write the vectors of the given folds as <name>.v<seq>.safetensors. Returns {name: file name}."""
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for fold in sorted(folds):
            for name in fold_names(fold):
                fname = name.replace("|", "_") + f".v{seq}.safetensors"
                save_file({f"t{i}": t for i, t in enumerate(self._alloc(name))}, str(directory / fname))
                files[name] = fname
        return files

    def load(self, directory: Path, files: dict[str, str], counts: dict[str, int]):
        from safetensors.torch import load_file

        for name, fname in files.items():
            data = load_file(str(directory / fname))
            acc = self._alloc(name)
            for i, a in enumerate(acc):
                a.copy_(data[f"t{i}"])
            self.count[name] = counts[name]


class Progress:
    """Resume state. A save writes new versioned vector files first, then progress.json by rename, then removes
    the replaced files. A stop before the rename leaves the earlier progress.json and its files valid."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.path = directory / "progress.json"
        self.state = {"seq": 0, "files": {}, "counts": {}, "done": [], "prompts": []}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())

    @property
    def done(self) -> set[int]:
        return set(self.state["done"])

    def restore(self, store: Store):
        store.load(self.dir, self.state["files"], self.state["counts"])

    def save(self, store: Store, prompts: list[dict]):
        seq = self.state["seq"] + 1
        new_files = store.save(self.dir, seq, store.dirty)
        old = dict(self.state["files"])
        files = {**old, **new_files}
        state = {"seq": seq, "files": files, "counts": dict(store.count),
                 "done": sorted({r["prompt"] for r in prompts}), "prompts": prompts}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, self.path)
        for name, fname in old.items():
            if files[name] != fname:
                (self.dir / fname).unlink(missing_ok=True)
        store.dirty.clear()
        self.state = state

    def remove_vectors(self):
        for fname in self.state["files"].values():
            (self.dir / fname).unlink(missing_ok=True)


def a_weights(labels: list[int]) -> list[float]:
    """r_i - mean_p r of the siblings of one prompt (item 5.2). All zero when the prompt is all correct or all wrong."""
    mean = sum(labels) / len(labels)
    return [float(r) - mean for r in labels]


def finish_prompt(store: Store, prompt: int, fold: int, trajs: list[int], labels: list[int], flags: list[str],
                  grads: list[list[torch.Tensor]], d: list[torch.Tensor], a_p: list[torch.Tensor] | None,
                  buf: list[torch.Tensor], extra: dict | None = None) -> dict:
    """Sibling Gram (raw and D), group sums into S|fold|class|flag, the prompt term a_p into A|fold, prompt record.

    grads: the O4 gradients of the siblings. a_p: sum_i (r_i - mean r) grad log p(y_i), None when all weights are 0.
    """
    raw, met = sibling_gram(grads, d)
    for c in CLASSES:
        for f in FLAGS:
            members = [i for i, (lab, fl) in enumerate(zip(labels, flags)) if ("C" if lab else "I") == c and fl == f]
            if members:
                weighted_sum(grads, [1.0 if i in members else 0.0 for i in range(len(grads))], buf)
                store.add(s_name(fold, c, f), buf, len(members))
    if a_p is None:
        store.add_count(a_name(fold), 1)
    else:
        store.add(a_name(fold), a_p, 1)
    return {"prompt": prompt, "fold": fold, "trajs": trajs, "labels": labels, "flags": flags,
            "raw": raw, "D": met, **(extra or {})}


def gram_file(store: Store, d: list[torch.Tensor], device: str, records: list[dict], checkpoint: str, k: int = K) -> dict:
    """The Gram file of one checkpoint (the only input of stats.py)."""
    names = vector_names(k)
    raw, met = write_gram(store, d, device, names)
    return {"checkpoint": checkpoint, "k": k, "fold_seed": FOLD_SEED, "names": names,
            "counts": {n: store.count.get(n, 0) for n in names}, "raw": raw, "D": met,
            "prompts": sorted(records, key=lambda r: r["prompt"])}


def write_gram(store: Store, d: list[torch.Tensor], device: str, names: list[str]) -> tuple[list[list[float]], list[list[float]]]:
    """28 x 28 dot products of the accumulators, raw and with D, fp64, one parameter tensor at a time."""
    n = len(names)
    raw = torch.zeros((n, n), dtype=torch.float64, device=device)
    met = torch.zeros((n, n), dtype=torch.float64, device=device)
    for t in range(len(store.shapes)):
        m = torch.stack([store.get(name)[t].reshape(-1) for name in names]).to(device=device, dtype=torch.float64)
        raw += m @ m.T
        met += (m * d[t].reshape(-1).to(device=device, dtype=torch.float64)) @ m.T
        del m
    return raw.tolist(), met.tolist()
