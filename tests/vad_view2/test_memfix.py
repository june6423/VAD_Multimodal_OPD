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
        assert all(torch.equal(x, y) for x, y in zip(a_pv, b_pv))
        assert a_m == b_m
    assert len(memfix._prev) == 1 and memfix._prev[0] is last and len(last.non_tensor_batch["teacher_multi_modal_inputs"]) == 4
    lines = [json.loads(x) for x in (tmp_path / "memfix.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [1, 2, 3]
    assert [x["released"] for x in lines] == [0, 1, 1]
    # 4 rows: one shared dict (pixel_values and image_grid_thw) and 2 own pixel_values tensors.
    assert all(x["teacher_tensors"] == 4 for x in lines)
    assert all(x["teacher_bytes"] == 3 * 5 * 7 * 4 + 3 * 8 for x in lines)


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
