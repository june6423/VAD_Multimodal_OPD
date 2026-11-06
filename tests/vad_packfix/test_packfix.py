"""Tests of the packed linear-attention fix (VLM-VAD-PACKFIX-001, Part A item 2).

A CPU toy Qwen3.5 text model with linear-attention layers only (the CPU has no varlen full attention) runs:
1. 3 rows packed into one batch-1 sequence (text position ids restart at 0 at each row),
2. each row alone.
With the fix, the packed outputs of each row equal its one-row outputs. Without the fix, rows 2 and 3 differ.
"""

import pytest
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from verl.models.transformers import qwen3_5_packfix as packfix

LENGTHS = (7, 3, 11)


def toy_model(n_layers=3):
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * n_layers,
    )
    model = qwen3_5.Qwen3_5TextModel(config).to(torch.float64).eval()
    for p in model.parameters():  # random weights so that the state term is not near zero
        torch.nn.init.normal_(p, std=0.2)
    return model


def rows():
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, 64, (n,), generator=gen) for n in LENGTHS]


def packed_inputs(ids):
    input_ids = torch.cat(ids)[None]
    position_ids = torch.cat([torch.arange(len(r)) for r in ids])[None]
    return input_ids, position_ids


def forward(model, input_ids, position_ids):
    return model(input_ids=input_ids, position_ids=position_ids, use_cache=False).last_hidden_state


def one_row_outputs(model, ids):
    with torch.no_grad():
        return [forward(model, r[None], torch.arange(len(r))[None])[0] for r in ids]


def split(out, ids):
    return list(torch.split(out, [len(r) for r in ids]))


@pytest.fixture
def patched():
    saved = dict(packfix._ORIGINAL)
    packfix._ORIGINAL.clear()
    layer, gdn = qwen3_5.Qwen3_5DecoderLayer.forward, qwen3_5.Qwen3_5GatedDeltaNet.forward
    assert packfix.apply_packfix()
    yield
    qwen3_5.Qwen3_5DecoderLayer.forward, qwen3_5.Qwen3_5GatedDeltaNet.forward = layer, gdn
    packfix._ORIGINAL.clear()
    packfix._ORIGINAL.update(saved)


def test_cu_seqlens_from_position_ids():
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    assert packfix.packed_cu_seqlens(pos).tolist() == [0, 3, 5, 9]
    assert packfix.packed_cu_seqlens(torch.arange(5)[None]) is None
    assert packfix.packed_cu_seqlens(None) is None
    assert packfix.packed_cu_seqlens(torch.zeros(2, 4, dtype=torch.long)) is None


def test_unfixed_packed_forward_leaks_state():
    model, ids = toy_model(), rows()
    input_ids, position_ids = packed_inputs(ids)
    with torch.no_grad():
        packed = split(forward(model, input_ids, position_ids)[0], ids)
    single = one_row_outputs(model, ids)
    assert torch.equal(packed[0], single[0])
    assert not torch.allclose(packed[1], single[1], atol=1e-6)
    assert not torch.allclose(packed[2], single[2], atol=1e-6)


def test_fixed_packed_forward_equals_one_row_forwards(patched):
    model, ids = toy_model(), rows()
    input_ids, position_ids = packed_inputs(ids)
    with torch.no_grad():
        packed = split(forward(model, input_ids, position_ids)[0], ids)
    single = one_row_outputs(model, ids)
    for p, s in zip(packed, single):
        torch.testing.assert_close(p, s, rtol=1e-10, atol=1e-10)


def test_fixed_gradients_equal_one_row_gradients(patched):
    model, ids = toy_model(n_layers=2), rows()
    input_ids, position_ids = packed_inputs(ids)
    model.train()
    model.zero_grad()
    forward(model, input_ids, position_ids).square().sum().backward()
    packed_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for r in ids:
        forward(model, r[None], torch.arange(len(r))[None]).square().sum().backward()
    # The gate g passes through float32 (".float()" in the model), so A_log and dt_bias sum their token terms in
    # float32 in another order: tolerance 1e-5.
    for gp, p in zip(packed_grads, model.parameters()):
        torch.testing.assert_close(gp, p.grad, rtol=1e-5, atol=1e-8)


def test_one_row_runs_the_original_code(patched):
    model, ids = toy_model(), rows()
    calls = []
    original = packfix._ORIGINAL["gdn"]

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    packfix._ORIGINAL["gdn"] = spy
    try:
        one_row_outputs(model, ids[:1])
    finally:
        packfix._ORIGINAL["gdn"] = original
    assert len(calls) == 3


def test_apply_is_idempotent(patched):
    assert not packfix.apply_packfix()
    assert qwen3_5.Qwen3_5GatedDeltaNet.forward is packfix.gated_delta_net_forward
