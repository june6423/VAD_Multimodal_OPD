"""Toy tests of diag_layers.py (HANDOFF-VLM-VAD-GRAD-003-R3 Part B item 2).

The toy student has 3 residual layers under non-reentrant gradient checkpointing (as the A1 student), then a linear
head. The rest is the test_grads.py toy actor with the real O4 loss.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

import test_grads as tg

dl = tg.load("vad_grad3_diag_layers", tg.DIR / "diag_layers.py")
partsum = tg.load("vad_grad3_diag_partsum", tg.DIR / "diag_partsum.py")


class Block(torch.nn.Module):
    def __init__(self, attn: str):
        super().__init__()
        setattr(self, attn, torch.nn.Linear(tg.H, tg.H, bias=False))
        self.attn = attn

    def forward(self, h):
        return h + torch.tanh(getattr(self, self.attn)(h))


class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.layers = torch.nn.ModuleList([Block("linear_attn"), Block("linear_attn"), Block("self_attn")])
        self.head = torch.nn.Linear(tg.H, tg.V, bias=False)

    def forward(self, h):
        for layer in self.layers:
            h = checkpoint(layer, h, use_reentrant=False)
        return self.head(h)


def test_layer_places_and_probe_gradients_equal_autograd():
    """Each layer output is captured once (not again in the checkpoint recomputation), and the hook gradient equals
    torch.autograd.grad for that tensor."""
    net = ToyNet()
    places = dl.layer_places(net)
    assert [(p, k) for p, k, _ in places] == [("layer.0", "linear_attn"), ("layer.1", "linear_attn"),
                                             ("layer.2", "self_attn")]
    probe = dl.Probe(places)
    x = torch.randn(2, 5, tg.H)
    w = torch.randn(2, 5, tg.V)
    out = probe.forward(lambda: net(x))
    loss = (out * w).sum()
    params = list(net.parameters())
    g, got = probe.backward(loss, params, True)
    assert sorted(got) == ["layer.0", "layer.1", "layer.2"]
    for n in got:
        ref = torch.autograd.grad(loss, probe.tensors[n], retain_graph=True)[0]
        assert torch.allclose(got[n], ref)
    ref_p = torch.autograd.grad(loss, params)
    assert all(torch.equal(a, b) for a, b in zip(g, ref_p))


def test_rel_helpers():
    t = {"a": torch.ones(4), "z": torch.zeros(2)}
    r = dl.rel_places({"a": 3 * torch.ones(4) * 1.5}, t, 3.0)
    assert abs(r["a"] - 0.5) < 1e-12 and r["z"] is None
    names = ["m.layers.0.x.weight", "m.layers.1.x.weight"]
    p = dl.rel_params([torch.ones(2), 2 * torch.ones(2)], [torch.ones(2), torch.ones(2)], names)
    assert p["layer.0"] == 0.0 and abs(p["layer.1"] - 1.0) < 1e-12 and abs(p["all"] - 0.5 ** 0.5) < 1e-12


def test_diagnose_on_the_toy_actor(tmp_path):
    """On the toy (fp32, exact kernels) scale3 and the part sums equal the total at each place within fp32 rounding."""
    actor = tg.make_actor(tg.a1_actor_config())
    actor.actor_module = ToyNet()
    rows, data = tg.toy_rows()
    ctx = tg.ToyCtx(actor, data)
    ctx.params = list(actor.actor_module.parameters())
    names = [n for n, _ in actor.actor_module.named_parameters()]
    probe = dl.Probe(dl.layer_places(actor.actor_module))
    recs = tg.row_file(rows, "B8-f1")
    micro = {}
    for x in recs:  # the toy micro-batches have at most 2 rows (pick_micro needs 3): the largest one
        micro.setdefault(tg.grads.micro_key(x), []).append(x)
    mrows = sorted(max(micro.values(), key=len), key=lambda x: x["position"])
    assert len(mrows) >= 2
    r = dl.diagnose(ctx, probe, mrows, {x["traj"]: {"traj": x["traj"]} for x in recs}, names)
    # The O4 loss of the A1 config uses the top-k log-probabilities; log_probs is outside the backward graph here.
    assert r["missing_places"] == ["log_probs"]
    assert [p["place"] for p in r["places"]] == ["layer.0", "layer.1", "layer.2", "log_probs", "topk_logps"]
    assert "log_probs" not in r["scale3"]["places"] and "log_probs" not in r["total_sq_norm"]
    assert all(v < 1e-5 for v in r["scale3"]["places"].values())
    assert all(v < 1e-5 for v in r["scale3"]["params"].values())
    used = [s for s in r["parts"] if any(p in r["part_losses"] for p in s.split("+"))]
    assert used
    for s in used:
        assert r["parts"][s]["loss_rel_diff"] < 1e-6
        assert all(v < 1e-5 for v in r["parts"][s]["places"].values())
        assert r["parts"][s]["params"]["all"] < 1e-5
    assert len(dl.table(r)) == 1 + len(r["places"]) + 1
