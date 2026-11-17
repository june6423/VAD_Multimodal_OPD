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
TASK-VLM-VAD-VIEW-002-007 showed that the release alone lowers the step-2 peak only from 146.0 to 142.1 (N1V8) and
145.9 GiB (T8): the transfer of the batch to the WorkerDict makes more copies of the teacher tensors, and the WorkerDict
keeps about 28 GiB between steps. Two more changes:
2. bf16 teacher pixel_values: after the build, the wrapper casts each float32 pixel_values tensor of the teacher
   multi_modal_inputs columns to bfloat16 (dicts that are also in the student column multi_modal_inputs stay). The
   actor and the fixed teacher are FSDP1 modules with MixedPrecision(param_dtype=bfloat16) (fsdp_config.dtype
   bfloat16): FSDP casts the floating-point inputs of the root forward to bfloat16, and the Qwen3.5 vision
   PatchEmbed casts to the bfloat16 weight dtype. The forward thus reads the same bfloat16 values. Each copy of the
   teacher pixel_values (driver, serialization, object store, worker) is half the size.
3. worker_setup(): the Ray worker hook. It runs scripts.vad_packfix.hostmem.setup() and wraps
   ActorRolloutRefWorker.update_actor: after the update, gc.collect() and malloc_trim(0), and one log line with the
   worker RSS at the start, at the end and after the trim (VAD_VIEW2_MEMFIX worker).
4. DIRECTIVE-VLM-VAD-VIEW-002-005 action 3: worker_setup() also wraps AgentLoopWorker.generate_sequences of the
   training. After the generation of each step, the agent loop worker runs gc.collect() and malloc_trim(0) (as
   scripts/vad_packfix/valtrim.py in the validation) and writes one line with its RSS before and after the trim
   (VAD_VIEW2_MEMFIX agent). The wrapper returns the object that the generation returned, without a change.
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


def teacher_pixel_values_bf16(dp, student) -> int:
    """Cast the float32 pixel_values of the teacher multi_modal_inputs dicts of dp to bfloat16, each dict once.
    Dicts that are also in the student multi_modal_inputs column (student) do not change. Return the cast count."""
    skip = {id(d) for d in student} if student is not None else set()
    done: set[int] = set()
    for key, col in dp.non_tensor_batch.items():
        if "multi_modal_inputs" not in key:
            continue
        for d in col:
            if not isinstance(d, dict) or id(d) in skip or id(d) in done:
                continue
            done.add(id(d))
            pv = d.get("pixel_values")
            if isinstance(pv, torch.Tensor) and pv.dtype == torch.float32:
                d["pixel_values"] = pv.to(torch.bfloat16)
    return len(done)


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
            rec["teacher_tensors_fp32"], rec["teacher_bytes_fp32"] = teacher_tensors(out[0])
            batch = args[0] if args else kwargs["batch"]
            rec["bf16_dicts"] = teacher_pixel_values_bf16(out[0], batch.non_tensor_batch.get("multi_modal_inputs"))
            rec["teacher_tensors"], rec["teacher_bytes"] = teacher_tensors(out[0])
        rec["rss_after_build_gib"] = rss_gib()
        _log(rec)
        return out

    cls._maybe_build_self_distillation_batch = wrapped
    cls._vad_view2_memfix = True


def patch_update_actor(cls) -> None:
    """Wrap cls.update_actor once: gc.collect() and malloc_trim(0) after the update, with one RSS log line."""
    if getattr(cls, "_vad_view2_memfix_worker", False):
        return
    orig = cls.update_actor

    @functools.wraps(orig)
    def update_actor(self, *args, **kwargs):
        r0 = rss_gib()
        out = orig(self, *args, **kwargs)
        r1 = rss_gib()
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        print(f"VAD_VIEW2_MEMFIX worker pid={os.getpid()} update_actor rss_gib start {r0:.2f} end {r1:.2f} "
              f"after_trim {rss_gib():.2f}", flush=True)
        return out

    cls.update_actor = update_actor
    cls._vad_view2_memfix_worker = True


def patch_agent_loop_trim(cls) -> None:
    """Wrap cls.generate_sequences once: gc.collect() and malloc_trim(0) after the generation, with one RSS log line."""
    if getattr(cls, "_vad_view2_agent_trim", False):
        return
    orig = cls.generate_sequences

    @functools.wraps(orig)
    async def generate_sequences(self, batch, *args, **kwargs):
        out = await orig(self, batch, *args, **kwargs)
        r0 = rss_gib()
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        print(f"VAD_VIEW2_MEMFIX agent pid={os.getpid()} step={batch.meta_info.get('global_steps', -1)} rows={len(batch)} "
              f"rss_before_gib {r0:.2f} rss_after_gib {rss_gib():.2f} utc {time.strftime('%H:%M:%S', time.gmtime())}",
              flush=True)
        return out

    cls.generate_sequences = generate_sequences
    cls._vad_view2_agent_trim = True


def worker_setup():
    """Ray worker_process_setup_hook of run_view2.sh: hostmem.setup(), the update_actor wrapper and the agent loop trim."""
    from scripts.vad_packfix import hostmem
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    hostmem.setup()
    patch_update_actor(ActorRolloutRefWorker)
    patch_agent_loop_trim(AgentLoopWorker)
