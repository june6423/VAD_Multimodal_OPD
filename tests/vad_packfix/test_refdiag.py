"""Tests of scripts/vad_packfix/refdiag.py (CPU, plain modules without FSDP)."""

import json
import types

import pytest
import torch

from scripts.vad_packfix import hostmem, refdiag


def _worker(share_with_teacher):
    ref = torch.nn.Linear(64, 64)
    teacher = torch.nn.Linear(64, 64)
    if share_with_teacher:
        teacher.weight = ref.weight
    w = types.SimpleNamespace(_is_actor=True, ref_module_fsdp=ref, ref_policy=object(),
                              actor_module_fsdp=torch.nn.Linear(64, 64), teacher_module_fsdp=teacher,
                              config=types.SimpleNamespace(actor={"use_kl_loss": False}),
                              actor=types.SimpleNamespace(teacher_module=teacher))
    return w


@pytest.mark.parametrize("share", [False, True])
def test_refdiag_finds_a_storage_shared_with_the_teacher(tmp_path, monkeypatch, share):
    monkeypatch.setattr(refdiag, "OUT", str(tmp_path))
    w = _worker(share)
    with pytest.raises(RuntimeError, match="REFDIAG"):
        refdiag._diagnose(w, hostmem.release_reference)
    (path,) = tmp_path.glob("refdiag_*.json")
    diag = json.loads(path.read_text())
    assert diag["reference"]["storages"] == 2
    assert [line.split()[0] for line in diag["release_log"]][:2] == ["before", "after_gc"]
    assert diag["reference_shared_with_teacher"]["storages"] == (1 if share else 0)
    assert diag["reference_shared_with_actor"]["storages"] == 0
    assert (diag["alive"] > 0) == share
    if share:
        assert diag["alive_kinds"] == ["param"] and diag["first_alive_referrers"]
