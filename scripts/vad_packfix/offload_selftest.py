"""Checks of the DIRECTIVE-VLM-VAD-PACKFIX-001-005 parts of hostmem.py (candidates 2, 4 and 5), without pytest.

tests/vad_packfix/test_hostmem_offload.py calls each check on the CPU. On b200 (no pytest in the env)
run_teacher_check.sh runs this file with CUDA: FSDP1 in one process (world size 1, nccl), the GPU round trips, and
the teacher setup with CPUOffload. On the CPU, FSDP1 runs with gloo and without CPUOffload (pinned memory needs CUDA).
Usage: python offload_selftest.py <OFFLOAD_SELFTEST.json>. Exit 0 = each check passed.
"""

import json
import os
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from scripts.vad_packfix import hostmem

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda", 0) if CUDA else torch.device("cpu")


def init_process_group(store_dir):
    dist.init_process_group("nccl" if CUDA else "gloo", init_method=f"file://{os.path.join(store_dir, 'store')}",
                            rank=0, world_size=1)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, 16))

    def forward(self, x):
        return torch.tanh(self.lin(x)) * self.scale.to(x.dtype)


def _model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(_Block(), _Block(), torch.nn.Linear(16, 7))


def _fsdp(module, offload):
    return FSDP(module, device_id=DEVICE, auto_wrap_policy=ModuleWrapPolicy({_Block}),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                               buffer_dtype=torch.float32),
                cpu_offload=CPUOffload(offload_params=True) if offload else None)


def _flat_params(model):
    return [h.flat_param for h in model._all_handles]


@torch.no_grad()
def check_cast_forward(offload):
    """Candidate 2: the FSDP1 forward with bf16 flat parameters is bit-equal to the forward with fp32 ones."""
    x = torch.randn(5, 16, device=DEVICE)
    plain = _fsdp(_model(), offload).eval()
    cast = _fsdp(_model(), offload).eval()
    count, before, after = hostmem.cast_flat_params(cast)
    assert count == 3 and abs(after - before / 2) < 1e-12, (count, before, after)
    out_plain, out_cast = plain(x), cast(x)
    assert out_cast.dtype == out_plain.dtype == torch.bfloat16
    assert torch.equal(out_plain, out_cast), (out_plain - out_cast).abs().max().item()
    assert all(p.dtype == torch.bfloat16 for p in _flat_params(cast))
    assert all(b.dtype == torch.float32 for b in cast.buffers())
    if offload:  # the shard that FSDP1 keeps on the CPU (pinned) is the bf16 one
        assert all(p._local_shard.dtype == torch.bfloat16 and p._local_shard.device.type == "cpu"
                   for p in _flat_params(cast))
    return {"offload": offload, "max_abs_diff": (out_plain.float() - out_cast.float()).abs().max().item()}


def check_cast_releases_fp32_storage():
    """Candidate 2: after the cast, the module attributes are views of the bf16 flat parameter."""
    # TASK-VLM-VAD-PACKFIX-001-019: an address test fails on CUDA, because the caching allocator gives the freed fp32
    # block of one handle to the bf16 tensor of the next handle. Thus the check compares with the new flat parameter.
    cast = _fsdp(_model(), False)
    hostmem.cast_flat_params(cast)
    lin = cast.module[0].module.lin
    flat = cast.module[0]._handle.flat_param
    assert lin.weight.dtype == torch.bfloat16
    assert lin.weight.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
    return {}


def check_cast_refuses_after_lazy_init():
    model = _fsdp(_model(), False)
    model(torch.randn(2, 16, device=DEVICE))
    try:
        hostmem.cast_flat_params(model)
    except RuntimeError as e:
        assert "before its lazy init" in str(e)
        return {}
    raise AssertionError("cast_flat_params accepted a module after its lazy init")


def check_model_round_trip():
    """Candidate 5: offload into pageable memory of the exact size, then the verl load, gives bit-equal tensors."""
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu

    model = _fsdp(_model(), False)
    model(torch.randn(2, 16, device=DEVICE))  # lazy init, as the actor before its first offload
    before = [p.detach().clone() for p in _flat_params(model)]
    hostmem.offload_model_pageable(model)
    for p, b in zip(_flat_params(model), before):
        assert p.device.type == "cpu" and not p.is_pinned()
        assert p.untyped_storage().nbytes() == p.numel() * p.element_size()
        assert torch.equal(p.detach(), b.cpu())
        assert p._local_shard.data_ptr() == p.data_ptr()
    if CUDA:
        load_fsdp_model_to_gpu(model)
        torch.cuda.synchronize()
        assert all(p.device == DEVICE and torch.equal(p.detach(), b) for p, b in zip(_flat_params(model), before))
    return {"gpu_round_trip": CUDA}


def check_optimizer_round_trip():
    """Candidate 4: AdamW states offload into pageable memory, then the verl load, gives bit-equal tensors."""
    from verl.utils.fsdp_utils import load_fsdp_optimizer

    model = _model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    hostmem.offload_optimizer_pageable(opt)  # no state yet: no copy
    model(torch.randn(3, 16, device=DEVICE)).square().sum().backward()
    opt.step()
    before = {(id(p), k): v.clone() for p, s in opt.state.items() for k, v in s.items() if torch.is_tensor(v)}
    hostmem.offload_optimizer_pageable(opt)
    for p, s in opt.state.items():
        for k, v in s.items():
            assert v.device.type == "cpu" and not v.is_pinned()
            assert torch.equal(v, before[(id(p), k)].cpu())
    load_fsdp_optimizer(opt, DEVICE if CUDA else "cpu")
    if CUDA:
        torch.cuda.synchronize()
    for p, s in opt.state.items():
        for k, v in s.items():
            assert torch.equal(v, before[(id(p), k)].to(v.device))
            if k != "step":
                assert v.device.type == DEVICE.type
    return {"gpu_round_trip": CUDA}


def check_deferred_optimizer_load():
    """P1: with the deferred load, 3 AdamW steps give bit-equal parameters, and the forward and backward passes run
    with the states on the CPU."""
    import types

    from verl.utils.fsdp_utils import load_fsdp_optimizer
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    device_id = DEVICE if CUDA else "cpu"
    orig = DataParallelPPOActor._optimizer_step
    try:
        hostmem._patch_deferred_optimizer_load(load_fsdp_optimizer)
        results = []
        for deferred in (False, True):
            model = _model().to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            actor = types.SimpleNamespace(config=types.SimpleNamespace(grad_clip=1.0), scaler=None,
                                          actor_module=model, actor_optimizer=opt)
            torch.manual_seed(1)
            for _ in range(3):
                if deferred:
                    hostmem.load_optimizer_deferred(optimizer=opt, device_id=device_id)
                else:
                    load_fsdp_optimizer(opt, device_id)
                model(torch.randn(3, 16, device=DEVICE)).square().sum().backward()
                if deferred and CUDA and opt.state:
                    assert all(v.device.type == "cpu" for s in opt.state.values() for v in s.values()), "not deferred"
                if deferred:
                    DataParallelPPOActor._optimizer_step(actor)
                else:
                    orig(actor)
                if CUDA and opt.state:
                    assert all(v.device.type == "cuda" for s in opt.state.values() for k, v in s.items()
                               if k != "step")
                opt.zero_grad()
                hostmem.offload_optimizer_pageable(opt)
                assert getattr(opt, "_vad_deferred_load", None) is None
            results.append([p.detach().clone() for p in model.parameters()])
        assert all(torch.equal(a, b) for a, b in zip(*results)), "deferred load changed the parameters"
    finally:
        DataParallelPPOActor._optimizer_step = orig
    return {"steps": 3, "gpu": CUDA}


@torch.no_grad()
def check_drop_teacher_cpu_grad():
    """P2: after the first forward the teacher handles hold an empty _cpu_grad, and the outputs stay bit-equal."""
    x = torch.randn(5, 16, device=DEVICE)
    plain = _fsdp(_model(), CUDA).eval()
    teacher = _fsdp(_model(), CUDA).eval()
    hostmem.cast_flat_params(teacher)
    hostmem._drop_teacher_cpu_grad_after_first_forward(teacher)
    first = teacher(x)
    grads = [getattr(p, "_cpu_grad", None) for p in _flat_params(teacher)]
    if CUDA:  # CPUOffload (pinned memory) needs CUDA
        assert all(g is not None and g.numel() == 0 and g.dtype == torch.bfloat16 for g in grads), grads
    assert not teacher._forward_hooks, "the hook must remove itself"
    second, ref = teacher(x), plain(x)
    assert torch.equal(first, second) and torch.equal(first, ref)
    return {"cpu_offload": CUDA}


CHECKS = [("cast_forward_no_offload", lambda: check_cast_forward(False))]
if CUDA:
    CHECKS.append(("cast_forward_cpu_offload", lambda: check_cast_forward(True)))
CHECKS += [("cast_releases_fp32_storage", check_cast_releases_fp32_storage),
           ("cast_refuses_after_lazy_init", check_cast_refuses_after_lazy_init),
           ("model_round_trip", check_model_round_trip),
           ("optimizer_round_trip", check_optimizer_round_trip),
           ("deferred_optimizer_load", check_deferred_optimizer_load),
           ("drop_teacher_cpu_grad", check_drop_teacher_cpu_grad)]


def main(dest):
    results = {"cuda": CUDA, "device": torch.cuda.get_device_name(0) if CUDA else "cpu", "torch": torch.__version__}
    with tempfile.TemporaryDirectory() as store_dir:
        init_process_group(store_dir)
        try:
            for name, fn in CHECKS:
                try:
                    results[name] = {"pass": True, **fn()}
                except Exception:
                    results[name] = {"pass": False, "error": traceback.format_exc()}
                print(f"OFFLOAD_SELFTEST {name} pass {results[name]['pass']}", flush=True)
        finally:
            dist.destroy_process_group()
    results["pass"] = all(v["pass"] for k, v in results.items() if isinstance(v, dict))
    with open(dest, "w") as f:
        json.dump(results, f, indent=1)
    print(f"OFFLOAD_SELFTEST all pass {results['pass']}", flush=True)
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
