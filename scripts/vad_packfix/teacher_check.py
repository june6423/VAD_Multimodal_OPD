"""Bit-equal check of candidate 2 of hostmem.py on b200 (DIRECTIVE-VLM-VAD-PACKFIX-001-005 action 6). Check only.

Ray loads setup() in each worker (TEACHER_CHECK_OUT set). setup() installs hostmem.py without the teacher cast, so the
WorkerDict builds the fixed teacher with fp32 flat parameters as before. The hook records the arguments of the teacher
build. At the first update_policy call (the real step-1 batch, before any update), the hook:
  1. selects the rows of the first TEACHER_CHECK_GROUPS uid groups,
  2. splits them into micro-batches as update_policy does (prepare_dynamic_batch, ppo_max_token_len_per_gpu),
  3. runs the teacher forwards of each micro-batch (teacher view and teacher_minus view, distillation_topk) with the
     fp32 teacher (run A),
  4. removes the fp32 teacher, builds the teacher again with the recorded arguments, and casts it with
     hostmem.cast_flat_params as hostmem.setup() does (run B),
  5. runs the same forwards with the bf16 teacher,
  6. compares log_probs, topk_logps and topk_indices of runs A and B (torch.equal and the largest absolute difference).
It writes TEACHER_CHECK.json and then stops the run by design. The teacher gets topk_indices=None: it returns its own
top-k, so the check needs no student forward.
python teacher_check.py <TEACHER_CHECK.json> prints the summary. Exit 0 only when each compared tensor is equal.
"""

import ctypes
import functools
import gc
import json
import os
import sys
import time

import numpy as np
import torch

from scripts.vad_packfix import hostmem

OUT = os.environ.get("TEACHER_CHECK_OUT")
GROUPS = int(os.environ.get("TEACHER_CHECK_GROUPS", "4"))
_STATE = {"worker": None, "teacher_kwargs": None}
KEYS = ("log_probs", "topk_logps", "topk_indices")


def first_groups(uids, groups):
    """Row indices of the first `groups` distinct uids, in the batch order."""
    chosen = list(dict.fromkeys(uids))[:groups]
    return np.array([i for i, u in enumerate(uids) if u in chosen], dtype=np.int64), chosen


def compare(a, b):
    """Per key: torch.equal and the largest absolute difference (float64) of two forward outputs."""
    result = {}
    for key in KEYS:
        if key not in a and key not in b:
            continue
        x, y = a[key], b[key]
        same_shape = x.shape == y.shape
        diff = (x.double() - y.double()).abs().max().item() if same_shape and x.numel() else None
        result[key] = {"equal": bool(same_shape and torch.equal(x, y)), "max_abs_diff": diff,
                       "shape": list(x.shape), "dtype": str(x.dtype)}
    return result


def _views(model_inputs):
    teacher = {
        "responses": model_inputs["responses"],
        "input_ids": model_inputs["teacher_input_ids"],
        "attention_mask": model_inputs["teacher_attention_mask"],
        "position_ids": model_inputs["teacher_position_ids"],
        "response_start_idx": model_inputs["teacher_response_start_idx"],
    }
    if "teacher_multi_modal_inputs" in model_inputs:
        teacher["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
    views = {"teacher": teacher}
    if "teacher_minus_input_ids" in model_inputs:
        minus = {
            "responses": model_inputs["responses"],
            "input_ids": model_inputs["teacher_minus_input_ids"],
            "attention_mask": model_inputs["teacher_minus_attention_mask"],
            "position_ids": model_inputs["teacher_minus_position_ids"],
            "response_start_idx": model_inputs["teacher_minus_response_start_idx"],
        }
        if "teacher_minus_multi_modal_inputs" in model_inputs:
            minus["multi_modal_inputs"] = model_inputs["teacher_minus_multi_modal_inputs"]
        views["teacher_minus"] = minus
    return views


@torch.no_grad()
def _forwards(actor, micro_batches, temperature, pad_token_id, distill_topk):
    from verl.utils.device import get_device_id

    outputs = []
    for mb in micro_batches:
        mb = mb.to(get_device_id())
        model_inputs = {**mb.batch, **mb.non_tensor_batch, "pad_token_id": pad_token_id}
        per_view = {}
        for name, inputs in _views(model_inputs).items():
            out = actor._forward_micro_batch(inputs, temperature=temperature, calculate_entropy=False,
                                             return_all_logps=False, distill_topk=distill_topk, topk_indices=None,
                                             module=actor.teacher_module)
            per_view[name] = {k: out[k].detach().cpu() for k in KEYS if k in out}
        outputs.append(per_view)
    return outputs


def _teacher_facts(teacher):
    flat = [h.flat_param for h in teacher._all_handles]
    return {"flat_params": len(flat), "flat_dtypes": sorted({str(p.dtype) for p in flat}),
            "local_shard_dtypes": sorted({str(p._local_shard.dtype) for p in flat if hasattr(p, "_local_shard")}),
            "local_shard_pinned": sorted({bool(p._local_shard.is_pinned()) for p in flat if hasattr(p, "_local_shard")}),
            "buffer_dtypes": sorted({str(b.dtype) for b in teacher.buffers()}), "training": teacher.training}


def _remove_teacher(worker, actor):
    """Remove the fp32 teacher and free its storages (flat parameters, pinned local shards, pinned gradient buffers)."""
    teacher = worker.teacher_module_fsdp
    storages = {}
    for h in teacher._all_handles:
        fp = h.flat_param
        for t in (fp, getattr(fp, "_local_shard", None), getattr(fp, "_cpu_grad", None)):
            if t is not None and t.untyped_storage().nbytes() > 0:
                storages[t.untyped_storage().data_ptr()] = t.untyped_storage()
    worker.teacher_module_fsdp = None
    actor.teacher_module = None
    del teacher
    gc.collect()
    freed = sum(s.nbytes() for s in storages.values())
    for s in storages.values():
        s.resize_(0)
    del storages
    torch._C._host_emptyCache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
    return freed / 2**30


def run_check(actor, data):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch

    worker = _STATE["worker"]
    log = {"groups": GROUPS, "mem": [hostmem._mem_state("start")],
           "teacher_is_root_after_init": _STATE.get("teacher_is_root_after_init")}
    idx, chosen = first_groups(list(data.non_tensor_batch["uid"]), GROUPS)
    sub = data.select_idxs(idx)
    cfg = actor.config.self_distillation
    distill_topk = cfg.distillation_topk if cfg.full_logit_distillation else None
    max_token_len = actor.config.ppo_max_token_len_per_gpu * actor.ulysses_sequence_parallel_size
    micro_batches, _ = prepare_dynamic_batch(sub, max_token_len=max_token_len)
    temperature = data.meta_info["temperature"]
    pad_token_id = data.meta_info.get("pad_token_id", 0)
    log.update({"rows": int(len(idx)), "batch_rows": len(data), "uids": [str(u) for u in chosen],
                "micro_batches": len(micro_batches), "micro_batch_rows": [len(mb) for mb in micro_batches],
                "teacher_tokens": int(sub.batch["teacher_attention_mask"].sum().item()),
                "distill_topk": distill_topk, "max_token_len": max_token_len})

    t0 = time.perf_counter()
    out_a = _forwards(actor, micro_batches, temperature, pad_token_id, distill_topk)
    log["run_a"] = {"seconds": time.perf_counter() - t0, **_teacher_facts(actor.teacher_module)}
    log["mem"].append(hostmem._mem_state("after_run_a"))

    log["fp32_teacher_freed_gib"] = _remove_teacher(worker, actor)
    log["mem"].append(hostmem._mem_state("after_remove_fp32_teacher"))
    t0 = time.perf_counter()
    teacher = worker._build_model_optimizer(**_STATE["teacher_kwargs"])[0]
    count, before, after = hostmem.cast_flat_params(teacher)
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    worker.teacher_module_fsdp = teacher
    actor.teacher_module = teacher
    log["rebuild"] = {"seconds": time.perf_counter() - t0, "cast_count": count, "gib_before": before,
                      "gib_after": after}
    log["mem"].append(hostmem._mem_state("after_rebuild_and_cast"))

    t0 = time.perf_counter()
    out_b = _forwards(actor, micro_batches, temperature, pad_token_id, distill_topk)
    log["run_b"] = {"seconds": time.perf_counter() - t0, **_teacher_facts(actor.teacher_module)}
    log["mem"].append(hostmem._mem_state("after_run_b"))

    table = []
    for i, (a, b) in enumerate(zip(out_a, out_b)):
        for view in a:
            table.append({"micro_batch": i, "view": view, **compare(a[view], b[view])})
    diffs = [t[k]["max_abs_diff"] for t in table for k in KEYS if k in t and t[k]["max_abs_diff"] is not None]
    log["compared"] = table
    log["all_equal"] = bool(table) and all(t[k]["equal"] for t in table for k in KEYS if k in t)
    log["max_abs_diff"] = max(diffs) if diffs else None
    log["pass"] = log["all_equal"] and log["run_a"]["flat_dtypes"] == ["torch.float32"] \
        and log["run_b"]["flat_dtypes"] == ["torch.bfloat16"] and log["teacher_is_root_after_init"] is None
    return log


def setup():
    if not OUT:
        raise RuntimeError("TEACHER_CHECK_OUT is not set")
    hostmem.setup(teacher_bf16=False)
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    orig_build = ActorRolloutRefWorker._build_model_optimizer
    orig_init = ActorRolloutRefWorker.init_model
    orig_update = DataParallelPPOActor.update_policy

    @functools.wraps(orig_build)
    def _build_model_optimizer(self, *args, **kwargs):
        if kwargs.get("role") == "teacher":
            if args:
                raise RuntimeError("TEACHER_CHECK: the teacher build uses keyword arguments only")
            _STATE["teacher_kwargs"] = dict(kwargs)
        return orig_build(self, *args, **kwargs)

    @functools.wraps(orig_init)
    def init_model(self, *args, **kwargs):
        _STATE["worker"] = self
        result = orig_init(self, *args, **kwargs)
        # hostmem.setup() casts the teacher at this point; cast_flat_params needs a teacher before its lazy init.
        teacher = getattr(self, "teacher_module_fsdp", None)
        _STATE["teacher_is_root_after_init"] = None if teacher is None else teacher._is_root
        return result

    @functools.wraps(orig_update)
    def update_policy(self, data):
        if _STATE["worker"] is None or _STATE["teacher_kwargs"] is None:
            raise RuntimeError("TEACHER_CHECK: the hook did not see the teacher build")
        log = run_check(self, data)
        with open(os.path.join(OUT, "TEACHER_CHECK.json"), "w") as f:
            json.dump(log, f, indent=1)
        print(f"TEACHER_CHECK pid={os.getpid()} pass {log['pass']} max_abs_diff {log['max_abs_diff']}", flush=True)
        raise RuntimeError("TEACHER_CHECK: check done; the check run stops before the update by design")

    ActorRolloutRefWorker._build_model_optimizer = _build_model_optimizer
    ActorRolloutRefWorker.init_model = init_model
    DataParallelPPOActor.update_policy = update_policy
    print(f"TEACHER_CHECK hook installed pid={os.getpid()}", flush=True)


def main(path):
    if not os.path.exists(path):
        print(f"no {path}")
        return 1
    log = json.load(open(path))
    print(json.dumps({k: v for k, v in log.items() if k != "compared"}))
    for t in log["compared"]:
        print(t["micro_batch"], t["view"], " ".join(f"{k} equal {t[k]['equal']} max_abs {t[k]['max_abs_diff']}"
                                                    for k in KEYS if k in t))
    return 0 if log["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
