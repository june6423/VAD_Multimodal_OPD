"""Ray worker setup hook of the val_only processes (DIRECTIVE-VLM-VAD-PACKFIX-001-008, option A).

TASK-VLM-VAD-PACKFIX-001-026: the agent worker RSS grew from 13.7 to 120 GiB across the batches of one full-set
validation. setup() installs hostmem.setup() unchanged and then, only when the environment flag VAD_VALTRIM is "1",
runs gc.collect() and malloc_trim(0) after each AgentLoopWorker.generate_sequences call. The returned batch does not
change. The worker prints its RSS before and after the trim for each call (one call for each validation batch).
The training processes load hostmem.setup, not this hook, so the training code path stays as at bf054e26.
DIRECTIVE-VLM-VAD-PACKFIX-001-009 action 6: with the same flag, the TaskRunner prints one VAD_VALBATCH line at the
start and one at the end of each AgentLoopManager.generate_sequences call (one call for each validation batch), with
the batch index in this process and the rows of the call (the batch after the padding to the worker count).
val_batches.py gives the row range and the image pixels of each batch index.
"""

import ctypes
import functools
import gc
import itertools
import os
import time

from scripts.vad_packfix import hostmem


def trim():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def _patch_trim():
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    orig = AgentLoopWorker.generate_sequences

    @functools.wraps(orig)
    async def generate_sequences(self, batch):
        try:
            return await orig(self, batch)
        finally:
            before = hostmem._rss_gib()
            trim()
            print(f"VAD_VALTRIM pid={os.getpid()} rows={len(batch)} rss_before_gib {before:.2f} "
                  f"rss_after_gib {hostmem._rss_gib():.2f} utc {time.strftime('%H:%M:%S', time.gmtime())}", flush=True)

    AgentLoopWorker.generate_sequences = generate_sequences


def _utc():
    return time.strftime('%H:%M:%S', time.gmtime())


def _patch_batch_log():
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    orig = AgentLoopManager.generate_sequences
    count = itertools.count()

    @functools.wraps(orig)
    def generate_sequences(self, prompts):
        k, start = next(count), time.time()
        print(f"VAD_VALBATCH pid={os.getpid()} batch={k} rows={len(prompts)} start_utc {_utc()}", flush=True)
        out = orig(self, prompts)
        print(f"VAD_VALBATCH pid={os.getpid()} batch={k} rows={len(prompts)} end_utc {_utc()} "
              f"seconds {time.time() - start:.1f}", flush=True)
        return out

    AgentLoopManager.generate_sequences = generate_sequences


def setup():
    hostmem.setup()
    if os.environ.get("VAD_VALTRIM") == "1":
        _patch_trim()
        _patch_batch_log()
        print(f"VAD_VALTRIM hook installed pid={os.getpid()}", flush=True)
