"""Driver host-memory change of VLM-VAD-VIEW-002 (DIRECTIVE-VLM-VAD-VIEW-002-002, TASK-VLM-VAD-VIEW-002-005 failure).

RayPPOTrainer.fit keeps the return value of _maybe_build_self_distillation_batch in the loop locals
self_distillation_data and self_distillation_batch until the same lines of the next step. The returned DataProto holds
the teacher multi_modal_inputs (pixel_values) of each distinct (prompt, teacher image) pair. T8 and N1V8 have 8 teacher
images for each prompt, so step N-1 holds 864 pixel_values tensors (B: 192) while step N builds its own set.
install() wraps the method. Each call first empties the DataProto that the previous call returned (non_tensor_batch,
batch, meta_info), then runs gc.collect() and malloc_trim(0), then builds the new step as before. At that time the
update of the previous step is complete, and fit no longer reads that object. The values of the new step do not change.
Each call writes one line to <VAD_VIEW_LOG_DIR>/memfix.jsonl: step, RSS of the driver at the start, after the release
and after the build, and the count and bytes of the distinct teacher tensors of the new step.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import json
import os
import time
from pathlib import Path

import torch

_prev: list = []


def rss_gib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return -1.0


def teacher_tensors(dp) -> tuple[int, int]:
    """Count and bytes of the distinct tensors in the multi_modal_inputs columns of a DataProto (each tensor once)."""
    seen: dict[int, int] = {}
    for key, col in dp.non_tensor_batch.items():
        if "multi_modal_inputs" not in key:
            continue
        for item in col:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, torch.Tensor):
                        seen[id(v)] = v.numel() * v.element_size()
    return len(seen), sum(seen.values())


def release(dp) -> None:
    """Remove the references of a DataProto to its data. The arrays and tensors are not changed."""
    dp.non_tensor_batch = {}
    dp.batch = None
    dp.meta_info = {}


def _log(rec: dict) -> None:
    d = os.environ.get("VAD_VIEW_LOG_DIR")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        with open(Path(d) / "memfix.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def install(cls) -> None:
    """Wrap cls._maybe_build_self_distillation_batch once."""
    if getattr(cls, "_vad_view2_memfix", False):
        return
    orig = cls._maybe_build_self_distillation_batch

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        rec = {"step": getattr(self, "global_steps", None), "time": time.time(), "rss_start_gib": rss_gib(),
               "released": len(_prev)}
        while _prev:
            release(_prev.pop())
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rec["rss_after_release_gib"] = rss_gib()
        out = orig(self, *args, **kwargs)
        if out is not None:
            _prev.append(out[0])
            rec["teacher_tensors"], rec["teacher_bytes"] = teacher_tensors(out[0])
        rec["rss_after_build_gib"] = rss_gib()
        _log(rec)
        return out

    cls._maybe_build_self_distillation_batch = wrapped
    cls._vad_view2_memfix = True
