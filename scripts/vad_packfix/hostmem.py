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
DIRECTIVE-VLM-VAD-PACKFIX-001-005 (TASK-VLM-VAD-PACKFIX-001-018: pinned host memory counts as container shmem, and the
caching host allocator rounds each pinned block up to a power of two):
4. Teacher (candidate 2): after init_model the worker casts the CPU flat parameters of the fixed teacher (FSDP1,
   CPUOffload, param_dtype bf16) to bf16, before the FSDP lazy init. The buffers stay. FSDP1 then copies the bf16 shard
   to the GPU in place of the fp32 shard that it cast to bf16 on the GPU (expected bit-equal: teacher_check.py).
5. AdamW states (candidate 4): offload_fsdp_optimizer copies them into pageable CPU memory of the exact size.
6. Actor flat parameters (candidate 5): offload_fsdp_model_to_cpu copies them into pageable CPU memory of the exact size.
   Items 5 and 6 change only the memory type of a copy. Each offload and load logs its wall time.
DT ANSWER on TASK-VLM-VAD-PACKFIX-001-022 (CUDA out of memory in the step-2 update; P1 and P2):
7. AdamW load (P1): update_actor records the load, and the actor loads the states before clip and step. The forward
   and backward passes then run without the 33.82 GiB of states on the GPU. The values do not change.
8. Teacher _cpu_grad (P2): after the first teacher forward (the FSDP1 lazy init) the worker frees the pinned _cpu_grad
   of each teacher handle. The fixed teacher has no backward pass and no optimizer.
HANDOFF-VLM-VAD-PACKFIX-001-R6 Definition 8 (logging only):
9. Rollout dumps: each row also gets data_source and the extra_info fields that the A1 reward function reads, so that
   rollout_acc.py can score the dumps after the training.
"""

import contextvars
import ctypes
import functools
import gc
import os
import time
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


@torch.no_grad()
def cast_flat_params(module, dtype=torch.bfloat16):
    """Candidate 2: cast the flat parameters of an FSDP1 module before its lazy init. Return (count, GiB before, after).

    The module attributes of the original parameters are views into the flat parameter. The handle sets them again,
    so the old storage is free at once. Buffers are not flat parameters and keep their dtype.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if module._is_root is not None:
        raise RuntimeError("VAD_HOSTMEM: cast_flat_params needs an FSDP module before its lazy init")
    handles = [m._handle for m in module.modules() if isinstance(m, FSDP) and m._handle is not None]
    before = after = 0
    for h in handles:
        if h._use_orig_params or h.uses_sharded_strategy:
            raise RuntimeError("VAD_HOSTMEM: cast_flat_params supports use_orig_params=False and NO_SHARD only")
        fp = h.flat_param
        before += fp.numel() * fp.element_size()
        fp.data = fp.data.to(dtype)
        h._use_unsharded_views(as_params=False)
        after += fp.numel() * fp.element_size()
    return len(handles), before / 2**30, after / 2**30


def drop_cpu_grad(module):
    """P2: free the pinned _cpu_grad of each CPUOffload handle of the fixed teacher. Return the freed GiB.

    FSDP1 allocates _cpu_grad in its lazy init for the backward pass and the optimizer. The fixed teacher has neither.
    An empty tensor replaces it, so a later gradient path fails with a size error instead of a silent change.
    """
    freed, grad = 0, None
    for h in module._all_handles:
        grad = getattr(h.flat_param, "_cpu_grad", None)
        if grad is not None and grad.numel() > 0:
            freed += grad.numel() * grad.element_size()
            h.flat_param._cpu_grad = torch.empty(0, dtype=grad.dtype)
    del grad
    if torch.cuda.is_available():
        torch._C._host_emptyCache()  # the caching host allocator keeps freed pinned blocks until this call
    return freed / 2**30


def _drop_teacher_cpu_grad_after_first_forward(teacher):
    """P2: FSDP1 runs the lazy init in the first forward. A forward hook frees _cpu_grad after it, once."""

    def hook(module, args, output):
        remove.remove()
        gib = drop_cpu_grad(module)
        print(f"VAD_HOSTMEM pid={os.getpid()} teacher cpu_grad freed gib {gib:.2f} {_mem_state('after_free')}",
              flush=True)

    remove = teacher.register_forward_hook(hook)


@torch.no_grad()
def offload_model_pageable(model, empty_cache: bool = True):
    """Candidate 5: verl offload_fsdp_model_to_cpu (FSDP1) with a blocking copy into pageable memory of the exact size.

    The verl copy uses non_blocking=True, which allocates a pinned block of the next power-of-two size.
    """
    from torch.distributed.fsdp._runtime_utils import _lazy_init

    t0 = time.perf_counter()
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    nbytes = 0
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to(torch.device("cpu"), non_blocking=False)
        flat_param._local_shard = flat_param.data
        nbytes += flat_param.numel() * flat_param.element_size()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_copy("offload_model", t0, nbytes)


def load_optimizer_deferred(optimizer, device_id):
    """P1: update_actor calls this in place of load_fsdp_optimizer. The load runs in the next _optimizer_step."""
    optimizer._vad_deferred_load = device_id


def _patch_deferred_optimizer_load(load):
    """P1 (TASK-VLM-VAD-PACKFIX-001-022): the AdamW states stay on the CPU during the forward and backward passes.

    The verl update_actor loads them before update_policy, so they use 33.82 GiB of GPU memory during the forward and
    backward passes (CUDA out of memory at step 2). Here the actor loads them before clip and step. The values stay.
    """
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    orig = DataParallelPPOActor._optimizer_step

    @functools.wraps(orig)
    def _optimizer_step(self):
        device_id = getattr(self.actor_optimizer, "_vad_deferred_load", None)
        if device_id is not None:
            self.actor_optimizer._vad_deferred_load = None
            load(optimizer=self.actor_optimizer, device_id=device_id)
        return orig(self)

    DataParallelPPOActor._optimizer_step = _optimizer_step


@torch.no_grad()
def offload_optimizer_pageable(optimizer):
    """Candidate 4: verl offload_fsdp_optimizer with a blocking copy into pageable memory of the exact size."""
    optimizer._vad_deferred_load = None  # P1: an update without an optimizer step leaves the states on the CPU
    if not optimizer.state:
        return
    t0 = time.perf_counter()
    nbytes = 0
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=False)
                    nbytes += value.numel() * value.element_size()
    _log_copy("offload_optimizer", t0, nbytes)


def _utc():
    return time.strftime("%H:%M:%S", time.gmtime())


def _log_copy(name, t0, nbytes=None):
    size = "" if nbytes is None else f" gib {nbytes / 2**30:.2f}"
    print(f"VAD_HOSTMEM pid={os.getpid()} copy {name} seconds {time.perf_counter() - t0:.2f}{size} utc {_utc()}",
          flush=True)


def _patch_tracking():
    """Log the UTC time of each metrics log call (the end of each step) for steppeak.py. Logging only."""
    from verl.utils.tracking import Tracking

    orig = Tracking.log

    @functools.wraps(orig)
    def log(self, data, step, *args, **kwargs):
        print(f"VAD_HOSTMEM pid={os.getpid()} step_log step {step} utc {_utc()}", flush=True)
        return orig(self, data, step, *args, **kwargs)

    Tracking.log = log


ROLLOUT_EXTRA_KEYS = ("benchmark", "query", "question")


def rollout_dump_fields(non_tensor_batch):
    """The reward-function inputs that the rollout dump lacks (HANDOFF-VLM-VAD-PACKFIX-001-R6 Definition 8).

    judge_batched reads data_source and extra_info benchmark, query and question. The dump holds the response and the
    ground truth. Logging only.
    """
    n = len(non_tensor_batch["data_source"])
    extras = non_tensor_batch.get("extra_info", [{}] * n)
    return {
        "data_source": [str(x) for x in non_tensor_batch["data_source"]],
        "extra_info": [{k: str(e[k]) for k in ROLLOUT_EXTRA_KEYS if k in (e or {})} for e in extras],
    }


def _patch_rollout_dump():
    """Add data_source and the extra_info fields of rollout_dump_fields to each rollout dump row. Logging only.

    The wrapper gives the dump a copy of reward_extra_infos_dict with the two fields. A key that the original adds to
    its argument (request_id) goes back to the caller dict, as before.
    """
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    orig = RayPPOTrainer._log_rollout_data

    @functools.wraps(orig)
    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        added = rollout_dump_fields(batch.non_tensor_batch)
        proxy = {**reward_extra_infos_dict, **added}
        result = orig(self, batch, proxy, timing_raw, rollout_data_dir)
        for key, value in proxy.items():
            if key not in reward_extra_infos_dict and key not in added:
                reward_extra_infos_dict[key] = value
        return result

    RayPPOTrainer._log_rollout_data = _log_rollout_data


def _timed_load(name, fn):
    """Log the wall time of a verl load (asynchronous copies: the wrapper waits for the GPU at the end)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _log_copy(name, t0)
        return result

    return wrapper


def _patch_workerdict(teacher_bf16):
    from verl.workers import fsdp_workers
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    orig = ActorRolloutRefWorker.init_model

    @functools.wraps(orig)
    def init_model(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        log = release_reference(self)
        for line in log or []:
            print(f"VAD_HOSTMEM pid={os.getpid()} reference model removed {line}", flush=True)
        teacher = getattr(self, "teacher_module_fsdp", None)
        if teacher_bf16 and teacher is not None:
            count, before, after = cast_flat_params(teacher)
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print(f"VAD_HOSTMEM pid={os.getpid()} teacher flat params bf16 count {count} gib {before:.2f} -> "
                  f"{after:.2f} {_mem_state('after_cast')}", flush=True)
            _drop_teacher_cpu_grad_after_first_forward(teacher)
        return result

    ActorRolloutRefWorker.init_model = init_model
    # fsdp_workers imported these helpers by name: replace the names that ActorRolloutRefWorker calls.
    fsdp_workers.offload_fsdp_model_to_cpu = offload_model_pageable
    fsdp_workers.offload_fsdp_optimizer = offload_optimizer_pageable
    fsdp_workers.load_fsdp_model_to_gpu = _timed_load("load_model", fsdp_workers.load_fsdp_model_to_gpu)
    _patch_deferred_optimizer_load(_timed_load("load_optimizer", fsdp_workers.load_fsdp_optimizer))
    fsdp_workers.load_fsdp_optimizer = load_optimizer_deferred


def setup(teacher_bf16=True):
    _patch_workerdict(teacher_bf16)
    _patch_agent_loop()
    _patch_tracking()
    _patch_rollout_dump()
    print(f"VAD_HOSTMEM hook installed pid={os.getpid()}", flush=True)
