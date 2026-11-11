"""vgram.py on small fake vectors: names, Gram matrices, fold means, removal."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vad_view"))
sys.path.insert(0, str(ROOT / "scripts" / "vad_grad3"))

import vgram  # noqa: E402

NAMES = [f"model.layers.{i}.w" for i in range(248)]
SHAPE = (1, 3)
VIEWS = ["p1-z2.0", "p0.30-z3.0"]
FOLDS = [1, 3]


def test_vector_names():
    assert vgram.vector_name(Path("O__B8-f2@p0.30-z3.0__total.pt")) == "G__f2__p0.30-z3.0"
    assert vgram.vector_name(Path("H__skill__h3.pt")) == "H__skill__h3"
    assert vgram.vector_name(Path("partial_O__B8-f2@p1-z2.0__total.pt")) is None
    assert vgram.vector_name(Path("O__B8-f1@g2__total.tmp")) is not None  # only *.pt files are listed by inputs()


@pytest.mark.parametrize("block", [None, "100"])
def test_gram_fold_means_remove(tmp_path, monkeypatch, block):
    from safetensors.torch import save_file

    if block:
        monkeypatch.setenv("BLOCK", block)
    n = 248 * 3
    g = torch.Generator().manual_seed(5)
    (tmp_path / "G3_PARAMS.txt").write_text("\n".join(f"{x} ({SHAPE[0]}, {SHAPE[1]}) 3" for x in NAMES) + "\n")
    ck = tmp_path / "M0"
    (ck / "vectors").mkdir(parents=True)
    disp = tmp_path / "disp"
    disp.mkdir()
    vec = {}
    for f in FOLDS:
        for v in VIEWS:
            vec[f"G__f{f}__{v}"] = torch.randn(n, generator=g)
            torch.save(vec[f"G__f{f}__{v}"], ck / "vectors" / f"O__B8-f{f}@{v}__total.pt")
    for k in ("skill", "fmt"):
        for h in (1, 2):
            vec[f"H__{k}__h{h}"] = torch.randn(n, generator=g)
            torch.save(vec[f"H__{k}__h{h}"], ck / "vectors" / f"H__{k}__h{h}.pt")
    for d in vgram.DISPLACEMENTS + ("D__A2_0_80",):
        vec[d] = torch.randn(n, generator=g)
        torch.save(vec[d], disp / f"{d}.pt")
    torch.save(torch.randn(n, generator=g), ck / "vectors" / "partial_O__B8-f1@p1-z2.0__total.pt")
    vfile = {x: torch.rand(SHAPE, generator=g) * 1e-6 for x in NAMES}
    save_file(vfile, str(tmp_path / "V.safetensors"), metadata={"step": "80"})
    vgram.gram([str(ck), str(disp), str(tmp_path / "V.safetensors"), str(tmp_path / "G3_PARAMS.txt"), "M0"])
    res = json.loads((ck / "GRAM_M0.json").read_text())
    assert "D__A2_0_80" not in res["names"] and res["names"][-2:] == list(vgram.DISPLACEMENTS)
    assert sorted(res["names"]) == sorted(k for k in vec if k != "D__A2_0_80")
    x = torch.stack([vec[k] for k in res["names"]]).double()
    corr = math.sqrt(1 - 0.999 ** 80)
    d = torch.cat([(1.0 / (vfile[nm].sqrt() / corr + 1e-8)).reshape(-1) for nm in NAMES]).double()
    for m, want in (("raw", x @ x.T), ("M1", (x * d) @ x.T), ("M2", (x * d) @ (x * d).T)):
        assert np.allclose(np.asarray(res[m]), want.numpy(), rtol=1e-6), m
    for v in VIEWS:
        want = torch.stack([vec[f"G__f{f}__{v}"] for f in FOLDS]).double().mean(0).numpy()
        assert np.allclose(np.load(ck / f"MEAN__{v}.npy"), want, rtol=1e-6, atol=1e-7)
        assert res["fold_means"][v] == [f"G__f{f}__{v}" for f in FOLDS]
    want_v = torch.stack([vec[f"G__f{f}__{v}"] for f in FOLDS for v in VIEWS]).double().mean(0).numpy()
    assert np.allclose(np.load(ck / "MEAN__V.npy"), want_v, rtol=1e-6, atol=1e-7)
    assert not list(ck.glob("*.tmp"))
    vgram.remove([str(ck)])
    assert not list((ck / "vectors").glob("*")) and not list(ck.glob("MEAN__*"))
    rec = json.loads((ck / "VECTORS_REMOVED.json").read_text())
    assert rec["left"] == [] and "MEAN__V.npy" in rec["removed"] and "H__fmt__h2.pt" in rec["removed"]
    assert all((disp / f"{k}.pt").exists() for k in vgram.DISPLACEMENTS)
