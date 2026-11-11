"""Input building in worker processes ahead of the GPU (DIRECTIVE-VLM-VAD-VIEW-001-002 items 1 and 2).

Pool: spawn processes (no CUDA), each with one torch CPU thread and one builder object (initializer: sys.path, then
<module>.<class>(*args)). A job calls a builder method; the result returns through torch.multiprocessing shared memory.
ordered(): the results of a job sequence in the order of the sequence. Jobs are submitted ahead while the summed cost
of the submitted, not yet taken jobs is below the budget (at least one job). Each result comes with the build seconds
in the worker and the seconds the parent waited for it.
The builders call the same input code as the in-process path, so the inputs are the same values.
DataProto transfer: parts() gives the tensors, the non-tensor arrays and meta_info; proto() makes the DataProto again.

Env: PREFETCH_WORKERS (16), PREFETCH_BUDGET (cost units of the caller; see the caller).
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections import deque

_BUILDER = None


def _init(paths: list[str], module: str, cls: str, args: tuple):
    global _BUILDER
    import torch
    import torch.multiprocessing  # noqa: F401  (shared-memory reductions for the results)

    torch.set_num_threads(1)
    for p in reversed(paths):
        sys.path.insert(0, p)
    _BUILDER = getattr(importlib.import_module(module), cls)(*args)


def _call(method: str, args: tuple):
    t = time.time()
    res = getattr(_BUILDER, method)(*args)
    return res, time.time() - t


def parts(b) -> dict:
    """Picklable parts of a DataProto (flat TensorDict)."""
    import torch

    tensors = dict(b.batch.items()) if b.batch is not None else {}
    if not all(isinstance(v, torch.Tensor) for v in tensors.values()):
        raise RuntimeError("prefetch.parts: nested TensorDict")
    return {"tensors": tensors, "batch_size": list(b.batch.batch_size) if b.batch is not None else None,
            "non_tensors": b.non_tensor_batch, "meta": b.meta_info}


def proto(p: dict):
    from tensordict import TensorDict

    from verl import DataProto

    batch = TensorDict(p["tensors"], batch_size=p["batch_size"]) if p["batch_size"] is not None else None
    return DataProto(batch=batch, non_tensor_batch=p["non_tensors"], meta_info=p["meta"])


def workers() -> int:
    return int(os.environ.get("PREFETCH_WORKERS", "16"))


class Pool:
    def __init__(self, paths: list[str], module: str, cls: str, args: tuple, n: int | None = None):
        import torch.multiprocessing as tmp
        from concurrent.futures import ProcessPoolExecutor

        self.n = n or workers()
        self.ex = ProcessPoolExecutor(self.n, mp_context=tmp.get_context("spawn"), initializer=_init,
                                      initargs=(paths, module, cls, args))

    def ordered(self, jobs, budget: float):
        """jobs: iterable of (cost, method, args). Yields (result, build seconds, wait seconds) in job order."""
        q = deque()
        it = iter(jobs)
        pending_cost = 0.0
        nxt = next(it, None)
        while True:
            while nxt is not None and (not q or pending_cost + nxt[0] <= budget):
                cost, method, args = nxt
                q.append((cost, self.ex.submit(_call, method, args)))
                pending_cost += cost
                nxt = next(it, None)
            if not q:
                return
            cost, fut = q.popleft()
            t = time.time()
            res, build = fut.result()
            pending_cost -= cost
            yield res, build, time.time() - t

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)
