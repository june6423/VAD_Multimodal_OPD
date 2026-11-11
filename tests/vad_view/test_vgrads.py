"""vgrads.geo_topk (Definition 13 of HANDOFF-VLM-VAD-VIEW-001-R3) and geo_views."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import vgrads  # noqa: E402

K = 100


def topk_logps(seed: int, shape=(2, 7)) -> torch.Tensor:
    """Log-probabilities of K entries of a full-vocabulary distribution (sum of the probabilities < 1)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(*shape, 1000, generator=g, dtype=torch.float64) * 3
    lp = torch.log_softmax(logits, -1)
    return lp.topk(K, dim=-1).values.float()


def centered_tail(x: torch.Tensor) -> torch.Tensor:
    y = vgrads.add_tail64(x)
    return y - y.mean(-1, keepdim=True)


def test_one_view_gives_the_input():
    x = topk_logps(1)
    out = vgrads.geo_topk([x])
    assert out.dtype == x.dtype and out.shape == x.shape
    assert torch.allclose(out.double(), x.double(), atol=1e-5)


def test_centered_tail_of_geo_is_mean_of_views():
    xs = [topk_logps(s) for s in (1, 2, 3)]
    geo = vgrads.geo_topk(xs)
    want = torch.stack([centered_tail(x) for x in xs]).mean(0)
    assert torch.allclose(centered_tail(geo), want, atol=1e-4)
    # fp64 inputs: the identity holds to rounding
    xs64 = [x.double() for x in xs]
    want64 = torch.stack([centered_tail(x) for x in xs64]).mean(0)
    assert torch.allclose(centered_tail(vgrads.geo_topk(xs64)), want64, atol=1e-10)


def test_geo_is_a_distribution_on_the_support():
    xs = [topk_logps(s) for s in (4, 5)]
    geo = vgrads.geo_topk([x.double() for x in xs])
    assert torch.all(geo.exp().sum(-1) < 1)
    assert torch.allclose(vgrads.add_tail64(geo).exp().sum(-1), torch.ones(geo.shape[:-1], dtype=torch.float64))


def test_geo_views(monkeypatch):
    monkeypatch.delenv("GEO_VIEWS", raising=False)
    names = vgrads.geo_views()
    assert len(names) == 8 and "p1-z2.0" in names and "p0.30-z1.5" not in names
    monkeypatch.setenv("GEO_VIEWS", "p1-z2.0,p0.30-z3.0")
    assert vgrads.geo_views() == ["p1-z2.0", "p0.30-z3.0"]
