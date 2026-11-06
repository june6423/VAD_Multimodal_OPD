"""Host-memory diagnosis hook of VLM-VAD-PACKFIX-001 (DIRECTIVE-VLM-VAD-PACKFIX-001-002). Diagnosis only.

Ray loads setup() in each worker process (runtime_env.worker_process_setup_hook). With MEMDIAG_OUT set:
1. ActorRolloutRefWorker.init_model (the WorkerDict): after the original call, write workerdict_init.json with the
   bytes of the unique CPU tensor storages by part (actor parameters, actor gradients, optimizer states, reference
   model, teacher model, other CPU tensors) and the process RSS.
2. AgentLoopWorker.generate_sequences: validation batches run unchanged. For the first training batch (step 1), the
   worker runs the real agent loop and post-processing for each row with a fixed response of response_length tokens
   in place of the vLLM generation. It writes the bytes of each trajectory field, the sha256 of pixel_values in each
   group of rollout.n rows, and a pickle test of shared pixel_values (agentloop_<pid>.json, agentloop_<pid>.jsonl).
   Then it raises an error: the run stops before the generation by design.
"""

import functools
import gc
import hashlib
import json
import os
import pickle
import time

import numpy as np
import torch

OUT = os.environ.get("MEMDIAG_OUT")
GIB = 2**30


def _rss_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return None


def _write_json(name, obj):
    path = os.path.join(OUT, name)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _storages(tensors):
    """Unique CPU storages of the tensors: {data_ptr: (nbytes, pinned, dtype)}."""
    out = {}
    for t in tensors:
        if t is None or not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            continue
        s = t.untyped_storage()
        if s.data_ptr() not in out:
            out[s.data_ptr()] = (s.nbytes(), bool(t.is_pinned()), str(t.dtype))
    return out


def _module_tensors(module):
    if module is None:
        return [], []
    params = list(module.parameters()) + list(module.buffers())
    grads = [p.grad for p in module.parameters()]
    return params, grads


def _optimizer_tensors(optimizer):
    if optimizer is None:
        return []
    return [v for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)]


def _summ(storages):
    return {
        "gib": sum(v[0] for v in storages.values()) / GIB,
        "pinned_gib": sum(v[0] for v in storages.values() if v[1]) / GIB,
        "count": len(storages),
        "dtypes": sorted({v[2] for v in storages.values()}),
    }


def _workerdict_census(worker):
    actor = getattr(worker, "actor_module_fsdp", None)
    ref = getattr(worker, "ref_module_fsdp", None)
    teacher = getattr(worker, "teacher_module_fsdp", None)
    optimizer = getattr(worker, "actor_optimizer", None)
    a_params, a_grads = _module_tensors(actor)
    parts = {
        "actor_parameters": _storages(a_params),
        "actor_gradients": _storages(a_grads),
        "optimizer_states": _storages(_optimizer_tensors(optimizer)),
        "reference_model": _storages(_module_tensors(ref)[0]),
        "teacher_model": _storages(_module_tensors(teacher)[0]),
    }
    seen = set()
    for st in parts.values():
        seen.update(st)
    gc.collect()
    all_cpu = _storages(o for o in gc.get_objects() if isinstance(o, torch.Tensor))
    parts["other_cpu_tensors"] = {k: v for k, v in all_cpu.items() if k not in seen}
    trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad) if actor is not None else 0
    param_bytes = a_params[0].element_size() if a_params else 4
    return {
        "pid": os.getpid(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rss_gib": _rss_gib(),
        "parts": {k: _summ(v) for k, v in parts.items()},
        "all_cpu_tensors_gib": sum(v[0] for v in all_cpu.values()) / GIB,
        "teacher_is_reference": teacher is not None and teacher is ref,
        "actor_trainable_numel": trainable,
        "actor_parameter_element_bytes": param_bytes,
        "optimizer_class": type(optimizer).__name__ if optimizer is not None else None,
        "predicted_adam_state_gib_after_first_update": 2 * trainable * param_bytes / GIB,
    }


def _patch_workerdict():
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    orig = ActorRolloutRefWorker.init_model

    @functools.wraps(orig)
    def init_model(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        if getattr(self, "_is_actor", False):
            _write_json("workerdict_init.json", _workerdict_census(self))
            print(f"MEMDIAG workerdict census written pid={os.getpid()}", flush=True)
        return result

    ActorRolloutRefWorker.init_model = init_model


def _pickled_len(obj):
    return len(pickle.dumps(obj, protocol=5))


def _sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _obj_array(items):
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


class _FakeServerManager:
    """Returns a fixed response of response_length tokens. Records the image bytes sent to vLLM."""

    def __init__(self, token_id, length):
        self.token_id, self.length, self.image_bytes = token_id, length, 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        from verl.workers.rollout.replica import TokenOutput

        for im in image_data or []:
            self.image_bytes += im.width * im.height * len(im.getbands())
        return TokenOutput(token_ids=[self.token_id] * self.length, log_probs=[0.0] * self.length)


async def _measure(self, batch, n_per_prompt, pickle_groups=3):
    from verl.experimental.agent_loop.agent_loop import get_trajectory_info

    pid = os.getpid()
    t0 = time.time()
    rss0 = _rss_gib()
    ntb = batch.non_tensor_batch
    n = len(batch)
    response_length = self._get_response_length(validate=False)
    token_id = self.tokenizer.encode(" the", add_special_tokens=False)[0]
    fake = _FakeServerManager(token_id, response_length)
    real_server, self.server_manager = self.server_manager, fake

    async def _no_score(*args, **kwargs):
        return None

    self._compute_score = _no_score
    if "agent_name" not in ntb:
        ntb["agent_name"] = _obj_array([self.config.actor_rollout_ref.rollout.agent.default_agent_loop] * n)
    index = ntb["index"] if "index" in ntb else np.arange(n)
    traj = await get_trajectory_info(-1, index.tolist(), False)
    config = self.config.actor_rollout_ref.rollout
    sampling_params = dict(temperature=config.temperature, top_p=config.top_p, repetition_penalty=1.0,
                           logprobs=config.calculate_log_probs)

    # Input fields (the non_tensor_batch from the trainer, after the Ray transfer).
    input_fields = {}
    for k, v in ntb.items():
        sizes = [_pickled_len(v[i]) for i in range(min(n, 2 * n_per_prompt))]
        shared = all(v[g + j] is v[g] for g in range(0, n - n_per_prompt + 1, n_per_prompt)
                     for j in range(n_per_prompt))
        input_fields[k] = {"pickled_bytes_first_rows": sizes, "same_object_in_each_group": bool(shared)}
    tensor_fields = {k: int(t.nbytes) for k, t in batch.batch.items()} if batch.batch is not None else {}

    totals, mm_meta, groups, pickle_tests = {}, {}, [], []
    jl = open(os.path.join(OUT, f"agentloop_{pid}.jsonl"), "w")
    rss_max = rss0
    for g in range(0, n, n_per_prompt):
        outs, shas = [], []
        for i in range(g, min(g + n_per_prompt, n)):
            kwargs = {k: v[i] for k, v in ntb.items()}
            kwargs["__validate__"] = False
            ts = time.time()
            out = await self._run_agent_loop(sampling_params, traj[i], trace=False, **kwargs)
            row = {"row": i, "seconds": round(time.time() - ts, 3)}
            for f in ["prompt_ids", "response_ids", "input_ids", "position_ids", "response_mask",
                      "attention_mask", "response_logprobs"]:
                t = getattr(out, f)
                row[f] = int(t.nbytes) if t is not None else 0
            for k, t in (out.multi_modal_inputs or {}).items():
                row["mmi." + k] = int(t.nbytes)
                mm_meta.setdefault(k, {"dtype": str(t.dtype), "shape_row0": list(t.shape)})
            ims = (out.multi_modal_data or {}).get("images") or []
            row["mmd.images_pil"] = int(sum(im.width * im.height * len(im.getbands()) for im in ims))
            row["mmd.image_sizes"] = [[im.width, im.height] for im in ims]
            row["extra_fields_pickled"] = _pickled_len(out.extra_fields)
            pv = (out.multi_modal_inputs or {}).get("pixel_values")
            row["pixel_values_sha256"] = _sha(pv) if pv is not None else None
            for k, val in row.items():
                if isinstance(val, int) and k != "row":
                    totals[k] = totals.get(k, 0) + val
            jl.write(json.dumps(row) + "\n")
            outs.append(out)
            shas.append(row["pixel_values_sha256"])
        jl.flush()
        thw = [o.multi_modal_inputs.get("image_grid_thw") for o in outs]
        groups.append({
            "first_row": g,
            "pixel_values_equal_in_group": len(set(shas)) == 1,
            "image_grid_thw_equal_in_group": all(t is not None and torch.equal(t, thw[0]) for t in thw),
        })
        if len(pickle_tests) < pickle_groups and outs[0].multi_modal_inputs is not None:
            mmis = [o.multi_modal_inputs for o in outs]
            unshared = _obj_array(mmis)
            shared_items = [dict(m, pixel_values=mmis[0]["pixel_values"]) for m in mmis]
            shared = _obj_array(shared_items)
            data = pickle.dumps(shared, protocol=5)
            back = pickle.loads(data)
            pickle_tests.append({
                "first_row": g,
                "unshared_pickled_bytes": _pickled_len(unshared),
                "shared_pickled_bytes": len(data),
                "shared_values_equal_after_load": all(
                    torch.equal(back[j][k], mmis[j][k]) for j in range(len(mmis)) for k in mmis[j]),
                "shared_objects_after_load": all(back[j]["pixel_values"] is back[0]["pixel_values"]
                                                 for j in range(len(mmis))),
            })
            del unshared, shared, shared_items, back, data
        rss_max = max(rss_max, _rss_gib())
        del outs
    jl.close()
    self.server_manager = real_server
    pv_total = totals.get("mmi.pixel_values", 0)
    summary = {
        "pid": pid,
        "rows": n,
        "rollout_n": n_per_prompt,
        "response_length_used": response_length,
        "seconds": round(time.time() - t0, 1),
        "rss_gib_start": rss0,
        "rss_gib_max_during_measure": rss_max,
        "input_tensor_fields_bytes": tensor_fields,
        "input_non_tensor_fields": input_fields,
        "multi_modal_inputs_meta": mm_meta,
        "output_totals_gib": {k: v / GIB for k, v in sorted(totals.items())},
        "image_bytes_sent_to_vllm_gib": fake.image_bytes / GIB,
        "groups": len(groups),
        "groups_pixel_values_equal": sum(x["pixel_values_equal_in_group"] for x in groups),
        "groups_image_grid_thw_equal": sum(x["image_grid_thw_equal_in_group"] for x in groups),
        "pixel_values_gib_unshared": pv_total / GIB,
        "pixel_values_gib_if_shared_per_group": pv_total / n_per_prompt / GIB,
        "pickle_tests": pickle_tests,
    }
    _write_json(f"agentloop_{pid}.json", summary)
    print(f"MEMDIAG agent loop measurement written pid={pid}", flush=True)
    raise RuntimeError("MEMDIAG: measurement done; the diagnosis run stops before the step-1 generation by design")


def _patch_agent_loop():
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    orig = AgentLoopWorker.generate_sequences

    @functools.wraps(orig)
    async def generate_sequences(self, batch):
        if batch.meta_info.get("validate", False):
            return await orig(self, batch)
        return await _measure(self, batch, int(self.config.actor_rollout_ref.rollout.n))

    AgentLoopWorker.generate_sequences = generate_sequences


def setup():
    if not OUT:
        return
    _patch_workerdict()
    _patch_agent_loop()
    print(f"MEMDIAG hook installed pid={os.getpid()}", flush=True)
