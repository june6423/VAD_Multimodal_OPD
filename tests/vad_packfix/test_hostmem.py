"""Tests of scripts/vad_packfix/hostmem.py: the host-memory fix changes no value, only the object sharing."""

import asyncio
import hashlib
import pickle
import types

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.vad_packfix import hostmem


def _sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _rows(n_prompts=3, n=8):
    """multi_modal_inputs of n rows for each prompt, each row computed again (as the processor does)."""
    rows = []
    for p in range(n_prompts):
        for _ in range(n):
            g = torch.Generator().manual_seed(p)
            rows.append({"pixel_values": torch.randn(50 + p, 1536, generator=g),
                         "image_grid_thw": torch.tensor([[1, 5, 10 + p]]),
                         "images_seqlens": torch.tensor([50 + p])})
    return rows


def _digest(rows):
    return [{k: _sha(v) for k, v in r.items()} for r in rows]


def test_share_pixel_values_keeps_each_value():
    plain = _rows()
    fixed = _rows()
    cache = {"pixel_values": {}, "shared": 0, "unique": 0}
    for r in fixed:
        hostmem.share_pixel_values(cache, r)
    assert _digest(plain) == _digest(fixed)
    assert cache["unique"] == 3 and cache["shared"] == 21
    for p in range(3):
        group = fixed[8 * p: 8 * p + 8]
        assert all(r["pixel_values"] is group[0]["pixel_values"] for r in group)
    assert fixed[0]["pixel_values"] is not fixed[8]["pixel_values"]
    # Pickle round trip (Ray pickles non_tensor_batch): equal values, one object for each group, 1/8 of the size.
    arr = np.empty(len(fixed), dtype=object)
    arr[:] = fixed
    data = pickle.dumps(arr, protocol=5)
    back = pickle.loads(data)
    assert _digest(list(back)) == _digest(plain)
    assert all(back[j]["pixel_values"] is back[0]["pixel_values"] for j in range(8))
    arr_plain = np.empty(len(plain), dtype=object)
    arr_plain[:] = plain
    assert len(data) < len(pickle.dumps(arr_plain, protocol=5)) / 7


def test_share_pixel_values_does_not_merge_different_values():
    a = {"pixel_values": torch.zeros(4, 8), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    b = {"pixel_values": torch.zeros(4, 8), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    b["pixel_values"][0, 0] = 1.0
    cache = {"pixel_values": {}, "shared": 0, "unique": 0}
    assert not hostmem.share_pixel_values(cache, a)
    assert not hostmem.share_pixel_values(cache, b)
    assert b["pixel_values"][0, 0] == 1.0 and cache["unique"] == 2
    assert not hostmem.share_pixel_values(None, {"pixel_values": torch.zeros(1)})


def test_process_vision_info_decodes_once_per_message_list():
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopWorker

    saved = {(AgentLoopBase, "process_vision_info"): AgentLoopBase.process_vision_info,
             (AgentLoopWorker, "generate_sequences"): AgentLoopWorker.generate_sequences,
             (AgentLoopWorker, "_compute_multi_modal_inputs"): AgentLoopWorker._compute_multi_modal_inputs}
    decodes = []

    class Dataset:
        @classmethod
        async def process_vision_info(cls, messages, image_patch_size, config):
            decodes.append(1)
            return [Image.new("RGB", (8, 4), color=(len(decodes), 0, 0))], None

    stub = types.SimpleNamespace(processor=types.SimpleNamespace(image_processor=types.SimpleNamespace(patch_size=16)),
                                 dataset_cls=Dataset, dataset_config=None)
    try:
        hostmem._patch_agent_loop()
        msgs_a = [{"role": "user", "content": "a"}]
        msgs_b = [{"role": "user", "content": "b"}]

        async def run():
            token = hostmem._CACHE.set({"pixel_values": {}, "vision": {}, "shared": 0, "unique": 0, "vision_hits": 0})
            try:
                outs = [await AgentLoopBase.process_vision_info(stub, list(m)) for m in [msgs_a] * 8 + [msgs_b] * 8]
                return outs, hostmem._CACHE.get()["vision_hits"]
            finally:
                hostmem._CACHE.reset(token)

        outs, hits = asyncio.run(run())
        assert len(decodes) == 2 and hits == 14
        assert all(o["images"][0] is outs[0]["images"][0] for o in outs[:8])
        assert all(o["images"] is not outs[0]["images"] for o in outs[1:8])  # a new list for each row
        assert outs[8]["images"][0] is not outs[0]["images"][0]
        # Without a cache (outside generate_sequences) each call decodes, as before.
        asyncio.run(AgentLoopBase.process_vision_info(stub, list(msgs_a)))
        assert len(decodes) == 3
    finally:
        for (cls, name), fn in saved.items():
            setattr(cls, name, fn)


def test_check_digest_sees_sharing_as_equal_and_a_changed_value_as_different():
    from scripts.vad_packfix import hostmem_check
    from verl.protocol import DataProto

    def dp(rows):
        mmi = np.empty(len(rows), dtype=object)
        mmi[:] = rows
        return DataProto.from_dict(tensors={"input_ids": torch.arange(len(rows) * 4).view(len(rows), 4)},
                                   non_tensors={"multi_modal_inputs": mmi, "raw_prompt": np.array(["q"] * len(rows))})

    plain, fixed = _rows(2, 4), _rows(2, 4)
    cache = {"pixel_values": {}, "shared": 0, "unique": 0}
    for r in fixed:
        hostmem.share_pixel_values(cache, r)
    d_plain = hostmem_check.digest(dp(plain))
    assert d_plain == hostmem_check.digest(dp(fixed))
    assert d_plain == hostmem_check.digest(pickle.loads(pickle.dumps(dp(fixed), protocol=5)))
    changed = _rows(2, 4)
    changed[5]["pixel_values"][3, 3] += 1.0
    d_changed = hostmem_check.digest(dp(changed))
    diff_rows = [i for i, (a, b) in enumerate(zip(d_plain["non_tensor.multi_modal_inputs"],
                                                   d_changed["non_tensor.multi_modal_inputs"])) if a != b]
    assert diff_rows == [5]


def test_setup_keeps_the_verl_dispatch_attributes():
    import inspect

    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopWorker
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.workers import fsdp_workers
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

    saved = {(ActorRolloutRefWorker, "init_model"): ActorRolloutRefWorker.init_model,
             (AgentLoopBase, "process_vision_info"): AgentLoopBase.process_vision_info,
             (AgentLoopWorker, "generate_sequences"): AgentLoopWorker.generate_sequences,
             (AgentLoopWorker, "_compute_multi_modal_inputs"): AgentLoopWorker._compute_multi_modal_inputs}
    saved.update({(fsdp_workers, name): getattr(fsdp_workers, name) for name in (
        "offload_fsdp_model_to_cpu", "offload_fsdp_optimizer", "load_fsdp_model_to_gpu", "load_fsdp_optimizer")})
    before = getattr(ActorRolloutRefWorker.init_model, MAGIC_ATTR)
    try:
        hostmem.setup()
        assert getattr(ActorRolloutRefWorker.init_model, MAGIC_ATTR) == before
        assert AsyncActorRolloutRefWorker.init_model is ActorRolloutRefWorker.init_model
        assert inspect.iscoroutinefunction(AgentLoopWorker.generate_sequences)
        assert inspect.iscoroutinefunction(AgentLoopBase.process_vision_info)
        assert all(getattr(cls, name) is not fn for (cls, name), fn in saved.items())
        assert fsdp_workers.offload_fsdp_model_to_cpu is hostmem.offload_model_pageable
        assert fsdp_workers.offload_fsdp_optimizer is hostmem.offload_optimizer_pageable
    finally:
        for (cls, name), fn in saved.items():
            setattr(cls, name, fn)


def test_release_reference_removes_only_an_unused_reference():
    def worker(use_kl_loss, teacher_is_ref):
        ref = torch.nn.Linear(4, 4)
        w = types.SimpleNamespace(_is_actor=True, ref_module_fsdp=ref, ref_policy=object(),
                                  config=types.SimpleNamespace(actor={"use_kl_loss": use_kl_loss}),
                                  actor=types.SimpleNamespace(teacher_module=ref if teacher_is_ref else object()))
        return w, ref

    w, _ = worker(False, False)
    assert hostmem.release_reference(w) is not None
    assert w.ref_module_fsdp is None
    with pytest.raises(RuntimeError):
        w.ref_policy.compute_log_prob(None)
    for args in [(True, False), (False, True)]:
        w, ref = worker(*args)
        assert hostmem.release_reference(w) is None and w.ref_module_fsdp is ref


def test_release_reference_frees_the_module_and_empties_the_pinned_host_cache(monkeypatch):
    # TASK-VLM-VAD-PACKFIX-001-012: the RSS stayed 60.2 GiB. The reference parameters are pinned (CPUOffloadPolicy),
    # and the caching host allocator keeps freed pinned blocks until torch._C._host_emptyCache().
    import weakref

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append(1), raising=False)
    ref = torch.nn.Linear(4, 4)
    alive = weakref.ref(ref)
    w = types.SimpleNamespace(_is_actor=True, ref_module_fsdp=ref, ref_policy=types.SimpleNamespace(actor_module=ref),
                              config=types.SimpleNamespace(actor={"use_kl_loss": False}),
                              actor=types.SimpleNamespace(teacher_module=object()))
    del ref
    log = hostmem.release_reference(w)
    assert alive() is None
    assert calls == [1]
    assert [line.split()[0] for line in log] == ["before", "after_gc", "after_storage_free", "after_host_empty_cache",
                                                 "after_malloc_trim"]


def test_release_reference_frees_storages_that_a_hidden_holder_keeps_but_not_shared_ones():
    # TASK-VLM-VAD-PACKFIX-001-015: after the release, the 57 flat parameters (16.91 GiB) stayed alive without a
    # Python referrer. The release must free their storages, but never a storage of the actor or the teacher.
    ref = torch.nn.Linear(64, 64)
    teacher = torch.nn.Linear(64, 64)
    teacher.bias = ref.bias  # a shared storage
    hidden = [ref.weight]  # the holder that gc does not see on b200
    w = types.SimpleNamespace(_is_actor=True, ref_module_fsdp=ref, ref_policy=object(),
                              actor_module_fsdp=torch.nn.Linear(64, 64), teacher_module_fsdp=teacher,
                              config=types.SimpleNamespace(actor={"use_kl_loss": False}),
                              actor=types.SimpleNamespace(teacher_module=teacher))
    del ref
    log = hostmem.release_reference(w)
    assert hidden[0].untyped_storage().nbytes() == 0
    assert teacher.bias.untyped_storage().nbytes() == 64 * 4
    assert any(line.startswith("after_storage_free") and "freed_storages 1 " in line for line in log)


def test_release_reference_names_a_referrer_that_keeps_the_module():
    ref = torch.nn.Linear(4, 4)
    keeper = [ref]
    w = types.SimpleNamespace(_is_actor=True, ref_module_fsdp=ref, ref_policy=object(),
                              config=types.SimpleNamespace(actor={"use_kl_loss": False}),
                              actor=types.SimpleNamespace(teacher_module=object()))
    del ref
    log = hostmem.release_reference(w)
    assert any(line.startswith("module_still_referenced") and "list" in line for line in log)
    assert keeper


def test_check_waits_for_all_agent_workers(tmp_path):
    from scripts.vad_packfix import hostmem_check

    (tmp_path / "hostmem_check_1.json").write_text("{}")
    assert hostmem_check.wait_for_workers(str(tmp_path), 2, timeout=0.3, poll=0.1) == 1
    (tmp_path / "hostmem_check_2.json").write_text("{}")
    assert hostmem_check.wait_for_workers(str(tmp_path), 2, timeout=60, poll=0.1) == 2
