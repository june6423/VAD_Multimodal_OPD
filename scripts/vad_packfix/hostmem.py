"""Host-memory fix of VLM-VAD-PACKFIX-001 for the b200 container limit (DIRECTIVE-VLM-VAD-PACKFIX-001-002).

Ray loads setup() in each worker process (runtime_env.worker_process_setup_hook, set by run_packfix.sh in both arms).
The fix changes no value that training reads. TASK-VLM-VAD-PACKFIX-001-011 measured each saving.
1. pixel_values: AgentLoopWorker computes multi_modal_inputs as before. When the pixel_values tensor is equal
   (torch.equal) to a tensor of the same generate_sequences call, the row gets that tensor object. The rollout.n rows
   of a prompt then hold one copy (57.57 GiB -> 7.20 GiB at step 1), also after the Ray transfer (pickle memo).
2. Decoded images: process_vision_info decodes the images of one message list once for each generate_sequences call.
   The rollout.n rows of a prompt share one raw_prompt object, so they get the same PIL objects (7.20 -> 0.90 GiB).
3. Reference model: the WorkerDict builds the reference model only because the fixed teacher is built in the same
   code branch. With actor.use_kl_loss False and a separate teacher, no computation reads it (use_kl_in_reward is
   False in A1). After init_model the worker removes it (16.91 GiB). A later reference call raises an error.
"""

import contextvars
import ctypes
import functools
import gc
import os
import weakref

import torch

_CACHE = contextvars.ContextVar("vad_hostmem_cache", default=None)


def _rss_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return -1.0


def share_pixel_values(cache, multi_modal_inputs):
    """Replace multi_modal_inputs['pixel_values'] by an equal tensor of the cache. Return True when shared."""
    pv = multi_modal_inputs.get("pixel_values")
    if cache is None or pv is None:
        return False
    thw = multi_modal_inputs.get("image_grid_thw")
    key = (tuple(pv.shape), str(pv.dtype), tuple(thw.flatten().tolist()) if thw is not None else None)
    for cand in cache["pixel_values"].setdefault(key, []):
        if torch.equal(cand, pv):
            multi_modal_inputs["pixel_values"] = cand
            cache["shared"] += 1
            return True
    cache["pixel_values"][key].append(pv)
    cache["unique"] += 1
    return False


def _patch_agent_loop():
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopWorker

    orig_generate = AgentLoopWorker.generate_sequences
    orig_mmi = AgentLoopWorker._compute_multi_modal_inputs
    orig_vision = AgentLoopBase.process_vision_info

    @functools.wraps(orig_generate)
    async def generate_sequences(self, batch):
        cache = {"pixel_values": {}, "vision": {}, "shared": 0, "unique": 0, "vision_hits": 0}
        token = _CACHE.set(cache)  # the agent loop tasks of this call copy this context
        try:
            return await orig_generate(self, batch)
        finally:
            _CACHE.reset(token)
            print(f"VAD_HOSTMEM pid={os.getpid()} rows={len(batch)} pixel_values unique={cache['unique']} "
                  f"shared={cache['shared']} vision_cache_hits={cache['vision_hits']}", flush=True)

    @functools.wraps(orig_mmi)
    def _compute_multi_modal_inputs(self, output, input_ids):
        multi_modal_inputs = orig_mmi(self, output, input_ids)
        share_pixel_values(_CACHE.get(), multi_modal_inputs)
        return multi_modal_inputs

    @functools.wraps(orig_vision)
    async def process_vision_info(self, messages):
        cache = _CACHE.get()
        if cache is None:
            return await orig_vision(self, messages)
        key = tuple(id(m) for m in messages)  # the message objects stay alive in the batch during the call
        hit = cache["vision"].get(key)
        if hit is None:
            hit = (await orig_vision(self, messages), list(messages))
            cache["vision"][key] = hit
        else:
            cache["vision_hits"] += 1
        return {k: list(v) for k, v in hit[0].items()}

    AgentLoopWorker.generate_sequences = generate_sequences
    AgentLoopWorker._compute_multi_modal_inputs = _compute_multi_modal_inputs
    AgentLoopBase.process_vision_info = process_vision_info


class _ReleasedReference:
    def __getattr__(self, name):
        raise RuntimeError("VAD_HOSTMEM: the reference model was removed after init_model; a reference call is unexpected")


def _mem_state(step):
    """One release-log line: the RSS and, with CUDA, the pinned host bytes in use and reserved (GiB)."""
    line = f"{step} rss_gib {_rss_gib():.1f}"
    if torch.cuda.is_available():
        s = torch.cuda.host_memory_stats()
        line += (f" pinned_allocated_gib {s.get('allocated_bytes.current', -2**30) / 2**30:.1f}"
                 f" pinned_reserved_gib {s.get('reserved_bytes.current', -2**30) / 2**30:.1f}")
    return line


def release_reference(worker):
    """Remove the unused reference model of the actor worker. Return the release log (one line per step) or None."""
    ref = getattr(worker, "ref_module_fsdp", None)
    if ref is None or not getattr(worker, "_is_actor", False) or worker.config.actor.get("use_kl_loss", False):
        return None
    if getattr(worker.actor, "teacher_module", None) is ref:
        return None
    log = [_mem_state("before")]
    # TASK-VLM-VAD-PACKFIX-001-015: after the release the 57 FSDP flat parameters (16.91 GiB, CPU) stayed alive
    # without a Python referrer. Thus the release frees their storages. A storage that the actor or the teacher also
    # uses stays (TASK-015: 0 shared storages).
    kept = {t.untyped_storage().data_ptr() for m in (getattr(worker, "actor_module_fsdp", None),
                                                    getattr(worker, "teacher_module_fsdp", None)) if m is not None
            for t in list(m.parameters()) + list(m.buffers())}
    storages = {p.untyped_storage().data_ptr(): p.untyped_storage() for p in ref.parameters()}
    alive = weakref.ref(ref)
    worker.ref_module_fsdp = None
    worker.ref_policy = _ReleasedReference()
    del ref
    gc.collect()
    log.append(_mem_state("after_gc"))
    if alive() is not None:  # a reference stays: name the referrer types (DT STATUS on TASK-012, item 2.2)
        kinds = sorted({type(r).__name__ for r in gc.get_referrers(alive())})
        log.append(f"module_still_referenced referrer_types {','.join(kinds)}")
    freed = 0
    for ptr, s in storages.items():
        if ptr not in kept and s.nbytes() > 0:
            freed += s.nbytes()
            s.resize_(0)
    log.append(_mem_state(f"after_storage_free freed_storages {sum(p not in kept for p in storages)} "
                          f"freed_gib {freed / 2**30:.2f} kept_shared {sum(p in kept for p in storages)}"))
    del storages
    if torch.cuda.is_available():
        # The reference parameters are pinned (CPUOffloadPolicy(pin_memory=True)). The caching host allocator keeps
        # freed pinned blocks, so the RSS stays (TASK-VLM-VAD-PACKFIX-001-012: 60.2 -> 60.2 GiB) until this call.
        torch._C._host_emptyCache()
        log.append(_mem_state("after_host_empty_cache"))
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    log.append(_mem_state("after_malloc_trim"))
    return log


def _patch_workerdict():
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    orig = ActorRolloutRefWorker.init_model

    @functools.wraps(orig)
    def init_model(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        log = release_reference(self)
        for line in log or []:
            print(f"VAD_HOSTMEM pid={os.getpid()} reference model removed {line}", flush=True)
        return result

    ActorRolloutRefWorker.init_model = init_model


def setup():
    _patch_workerdict()
    _patch_agent_loop()
    print(f"VAD_HOSTMEM hook installed pid={os.getpid()}", flush=True)
