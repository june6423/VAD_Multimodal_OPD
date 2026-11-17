"""Tests of scripts/vad_view2/memfix.py (DIRECTIVE-VLM-VAD-VIEW-002-002)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "vad_view2"))
import hook  # noqa: E402
from scripts.vad_view2 import memfix  # noqa: E402

from verl.protocol import DataProto  # noqa: E402


def _batch(step: int, rows: int = 4) -> DataProto:
    return DataProto.from_dict(tensors={"input_ids": torch.full((rows, 3), step)},
                               non_tensors={"uid": np.array([f"u{step}-{i // 2}" for i in range(rows)], dtype=object)})


def _make_trainer():
    class Trainer:
        global_steps = 0

        def _maybe_build_self_distillation_batch(self, batch, reward_tensor, reward_extra_infos_dict=None):
            rows = len(batch)
            step = int(batch.batch["input_ids"][0, 0])
            shared = {"pixel_values": torch.full((5, 7), float(step)), "image_grid_thw": torch.tensor([[1, 2, 2]])}
            mm = np.empty(rows, dtype=object)
            for i in range(rows):  # rows 0 and 1 share one dict, the other rows have their own tensors
                mm[i] = shared if i < 2 else {"pixel_values": torch.full((5, 7), float(step + i))}
            out = DataProto.from_dict(tensors={"teacher_input_ids": batch.batch["input_ids"] + 100},
                                      non_tensors={"teacher_multi_modal_inputs": mm})
            return out, {"self_distillation/x": float(step)}

    return Trainer


def _fit(trainer, steps):
    """The order of RayPPOTrainer.fit: build, union into the step batch, keep the locals until the next build."""
    seen = []
    self_distillation_batch = None
    for step in steps:
        trainer.global_steps = step
        batch = _batch(step)
        data = trainer._maybe_build_self_distillation_batch(batch, None, {})
        self_distillation_batch, metrics = data
        batch = batch.union(self_distillation_batch)
        seen.append((batch.batch["teacher_input_ids"].clone(),
                     [d["pixel_values"].clone() for d in batch.non_tensor_batch["teacher_multi_modal_inputs"]],
                     metrics))
    return seen, self_distillation_batch


def test_memfix_keeps_the_values(tmp_path, monkeypatch):
    monkeypatch.setenv("VAD_VIEW_LOG_DIR", str(tmp_path))
    plain = _make_trainer()
    fixed = _make_trainer()
    memfix._prev.clear()
    memfix.install(fixed)
    ref, _ = _fit(plain(), [1, 2, 3])
    got, last = _fit(fixed(), [1, 2, 3])
    for (a_ids, a_pv, a_m), (b_ids, b_pv, b_m) in zip(ref, got):
        assert torch.equal(a_ids, b_ids)
        # The fixed trainer gives the bfloat16 values that the FSDP root forward makes from the float32 values.
        assert all(y.dtype == torch.bfloat16 and torch.equal(x.to(torch.bfloat16), y) for x, y in zip(a_pv, b_pv))
        assert a_m == b_m
    assert len(memfix._prev) == 1 and memfix._prev[0] is last and len(last.non_tensor_batch["teacher_multi_modal_inputs"]) == 4
    lines = [json.loads(x) for x in (tmp_path / "memfix.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [1, 2, 3]
    assert [x["released"] for x in lines] == [0, 1, 1]
    # 4 rows: one shared dict (pixel_values and image_grid_thw) and 2 own pixel_values tensors, 3 distinct dicts.
    assert all(x["teacher_tensors"] == 4 and x["teacher_tensors_fp32"] == 4 for x in lines)
    assert all(x["teacher_bytes_fp32"] == 3 * 5 * 7 * 4 + 3 * 8 for x in lines)
    assert all(x["teacher_bytes"] == 3 * 5 * 7 * 2 + 3 * 8 for x in lines)
    assert all(x["bf16_dicts"] == 3 for x in lines)


def test_bf16_skips_the_student_dicts():
    student_dict = {"pixel_values": torch.full((5, 7), 0.1)}
    teacher_dict = {"pixel_values": torch.full((5, 7), 0.1), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    half_dict = {"pixel_values": torch.full((5, 7), 0.1, dtype=torch.bfloat16)}
    student = np.array([student_dict, student_dict], dtype=object)
    teacher = np.array([student_dict, teacher_dict, teacher_dict, half_dict], dtype=object)
    dp = DataProto.from_dict(tensors={"x": torch.zeros(4, 1)},
                             non_tensors={"teacher_multi_modal_inputs": teacher,
                                          "uid": np.array(["a", "b", "c", "d"], dtype=object)})
    assert memfix.teacher_pixel_values_bf16(dp, student) == 2
    assert student_dict["pixel_values"].dtype == torch.float32
    assert teacher_dict["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(teacher_dict["pixel_values"], torch.full((5, 7), 0.1).to(torch.bfloat16))
    assert teacher_dict["image_grid_thw"].dtype == torch.int64
    assert half_dict["pixel_values"].dtype == torch.bfloat16


def test_patch_update_actor(capsys):
    class Worker:
        calls = 0

        def update_actor(self, data):
            Worker.calls += 1
            return {"out": data}

    memfix.patch_update_actor(Worker)
    wrapped = Worker.update_actor
    memfix.patch_update_actor(Worker)
    assert Worker.update_actor is wrapped
    assert Worker().update_actor(7) == {"out": 7} and Worker.calls == 1
    out = capsys.readouterr().out
    assert "VAD_VIEW2_MEMFIX worker pid=" in out and "update_actor rss_gib start" in out and "after_trim" in out


def test_worker_setup(monkeypatch):
    class Worker:
        def update_actor(self, data):
            return data

    class AgentWorker:
        async def generate_sequences(self, batch):
            return batch

    setup_calls = []
    hostmem = types.ModuleType("scripts.vad_packfix.hostmem")
    hostmem.setup = lambda: setup_calls.append(1)
    workers = types.ModuleType("verl.workers.fsdp_workers")
    workers.ActorRolloutRefWorker = Worker
    agent_loop = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop.AgentLoopWorker = AgentWorker
    import scripts.vad_packfix as packfix_pkg

    monkeypatch.setattr(packfix_pkg, "hostmem", hostmem, raising=False)
    monkeypatch.setitem(sys.modules, "scripts.vad_packfix.hostmem", hostmem)
    monkeypatch.setitem(sys.modules, "verl.workers.fsdp_workers", workers)
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop.agent_loop", agent_loop)
    memfix.worker_setup()
    assert setup_calls == [1] and Worker._vad_view2_memfix_worker is True and AgentWorker._vad_view2_agent_trim is True


def _generated(step: int) -> DataProto:
    """A generation output with the kinds of values that the training reads: tensors, object arrays, meta_info."""
    mm = np.empty(4, dtype=object)
    for i in range(4):
        mm[i] = {"pixel_values": torch.full((5, 7), float(step + i)), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    return DataProto.from_dict(tensors={"responses": torch.arange(12).reshape(4, 3) + step,
                                        "rollout_log_probs": torch.linspace(-2.0, 0.0, 12).reshape(4, 3)},
                               non_tensors={"multi_modal_inputs": mm,
                                            "uid": np.array(["a", "a", "b", "b"], dtype=object)},
                               meta_info={"timing": {"generate_sequences": 1.5}})


def test_patch_agent_loop_trim_keeps_the_value(capsys):
    """DIRECTIVE-VLM-VAD-VIEW-002-005 action 5: the trim does not change a value that the training reads."""
    import asyncio

    class AgentWorker:
        async def generate_sequences(self, batch):
            return _generated(int(batch.batch["input_ids"][0, 0]))

    batch = _batch(3)
    batch.meta_info["global_steps"] = 3
    ref = asyncio.run(AgentWorker().generate_sequences(batch))
    memfix.patch_agent_loop_trim(AgentWorker)
    wrapped = AgentWorker.generate_sequences
    memfix.patch_agent_loop_trim(AgentWorker)
    assert AgentWorker.generate_sequences is wrapped
    got = asyncio.run(AgentWorker().generate_sequences(batch))
    assert set(got.batch.keys()) == set(ref.batch.keys())
    assert all(torch.equal(got.batch[k], ref.batch[k]) for k in ref.batch.keys())
    assert list(got.non_tensor_batch["uid"]) == list(ref.non_tensor_batch["uid"])
    for a, b in zip(ref.non_tensor_batch["multi_modal_inputs"], got.non_tensor_batch["multi_modal_inputs"]):
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)
    assert got.meta_info == ref.meta_info
    assert batch.meta_info == {"global_steps": 3} and int(batch.batch["input_ids"][0, 0]) == 3
    out = capsys.readouterr().out
    assert "VAD_VIEW2_MEMFIX agent pid=" in out and "step=3 rows=4 rss_before_gib" in out and "rss_after_gib" in out


def test_memfix_empties_the_previous_step(tmp_path, monkeypatch):
    monkeypatch.setenv("VAD_VIEW_LOG_DIR", str(tmp_path))
    trainer_cls = _make_trainer()
    memfix._prev.clear()
    memfix.install(trainer_cls)
    t = trainer_cls()
    first, _ = t._maybe_build_self_distillation_batch(_batch(1), None, {})
    step1_batch = _batch(1).union(first)
    second, _ = t._maybe_build_self_distillation_batch(_batch(2), None, {})
    assert first.non_tensor_batch == {} and first.batch is None and first.meta_info == {}
    assert second.non_tensor_batch["teacher_multi_modal_inputs"][0]["pixel_values"][0, 0] == 2.0
    # The release removes references only. A batch that still holds the step-1 arrays keeps their values.
    assert step1_batch.non_tensor_batch["teacher_multi_modal_inputs"][2]["pixel_values"][0, 0] == 3.0


def test_install_once_and_hook_install(monkeypatch):
    trainer_cls = _make_trainer()
    fake = types.ModuleType("verl.trainer.ppo.ray_trainer")
    fake.RayPPOTrainer = trainer_cls
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", fake)
    hook._install_memfix()
    wrapped = trainer_cls._maybe_build_self_distillation_batch
    assert trainer_cls._vad_view2_memfix is True
    memfix.install(trainer_cls)
    assert trainer_cls._maybe_build_self_distillation_batch is wrapped
