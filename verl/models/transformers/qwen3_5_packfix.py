# Packed-sequence fix of the Qwen3.5 linear-attention layers (VLM-VAD-PACKFIX-001, Definition 3).
#
# In the packed forward of verl (use_remove_padding: batch 1, rows concatenated, the text position ids restart at
# 0 at each row), Qwen3_5GatedDeltaNet of transformers 5.5.0 calls causal_conv1d_fn with seq_idx=None and
# chunk_gated_delta_rule without cu_seqlens. The conv window and the recurrent state then cross the row
# boundaries. This patch gives each packed row its own linear-attention state:
#   1. Row starts are the tokens with text position id 0 (the rule of the flash-attention varlen path).
#   2. The original Qwen3_5GatedDeltaNet.forward runs once for each row, and the row outputs are concatenated.
# Each row thus runs the same kernels as a one-row forward. The varlen kernels (seq_idx, cu_seqlens) are not used:
# in bfloat16 the varlen chunk_gated_delta_rule result of a row changes with the other rows of the pack
# (TASK-VLM-VAD-PACKFIX-001-002).
# A forward with one row, with a cache, or without position ids runs the original code.
# apply_monkey_patch turns the patch on when the environment variable VAD_PACKFIX is "1".

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5

_ORIGINAL = {}


def packed_cu_seqlens(position_ids):
    """Row boundaries [0, s_1, ..., L] of a packed batch-1 sequence, or None when it holds one row."""
    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    pos = position_ids[0]
    starts = torch.nonzero(pos == 0, as_tuple=False).flatten()
    if starts.numel() == 0 or starts[0].item() != 0:
        starts = torch.cat([starts.new_zeros(1), starts])
    if starts.numel() == 1:
        return None
    return torch.cat([starts, starts.new_tensor([pos.shape[0]])]).to(torch.long)


def decoder_layer_forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                          past_key_values=None, **kwargs):
    """Qwen3_5DecoderLayer.forward that hands the packed row boundaries to its linear-attention module."""
    cu_seqlens = None
    if self.layer_type == "linear_attention" and past_key_values is None:
        cu_seqlens = packed_cu_seqlens(position_ids)
    if cu_seqlens is None:
        return _ORIGINAL["layer"](self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    self.linear_attn._packfix_cu_seqlens = cu_seqlens
    try:
        return _ORIGINAL["layer"](self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    finally:
        self.linear_attn._packfix_cu_seqlens = None


def gated_delta_net_forward(self, hidden_states, cache_params=None, attention_mask=None):
    """Qwen3_5GatedDeltaNet.forward with one linear-attention state for each packed row."""
    cu_seqlens = getattr(self, "_packfix_cu_seqlens", None)
    if cu_seqlens is None or cache_params is not None:
        return _ORIGINAL["gdn"](self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)

    bounds = cu_seqlens.tolist()
    return torch.cat(
        [
            _ORIGINAL["gdn"](
                self,
                hidden_states[:, s:e],
                attention_mask=None if attention_mask is None else attention_mask[:, s:e],
            )
            for s, e in zip(bounds[:-1], bounds[1:])
        ],
        dim=1,
    )


def apply_packfix():
    """Replace the two forwards once. Return True when this call applied the patch."""
    if _ORIGINAL:
        return False
    _ORIGINAL["layer"] = qwen3_5.Qwen3_5DecoderLayer.forward
    _ORIGINAL["gdn"] = qwen3_5.Qwen3_5GatedDeltaNet.forward
    qwen3_5.Qwen3_5DecoderLayer.forward = decoder_layer_forward
    qwen3_5.Qwen3_5GatedDeltaNet.forward = gated_delta_net_forward
    return True
