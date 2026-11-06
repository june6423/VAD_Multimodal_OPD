# Packed-sequence fix of the Qwen3.5 linear-attention layers (VLM-VAD-PACKFIX-001, Definition 3).
#
# In the packed forward of verl (use_remove_padding: batch 1, rows concatenated, the text position ids restart at
# 0 at each row), Qwen3_5GatedDeltaNet of transformers 5.5.0 calls causal_conv1d_fn with seq_idx=None and
# chunk_gated_delta_rule without cu_seqlens. The conv window and the recurrent state then cross the row
# boundaries. This patch gives each packed row its own linear-attention state:
#   1. Row starts are the tokens with text position id 0 (the rule of the flash-attention varlen path).
#   2. causal_conv1d_fn gets seq_idx (the row index of each token).
#   3. chunk_gated_delta_rule gets cu_seqlens (the row boundaries).
# Without the fast kernels (CPU), the same boundaries split the torch fallback into one call per row.
# A forward with one row, with a cache, or without position ids runs the original code.
# apply_monkey_patch turns the patch on when the environment variable VAD_PACKFIX is "1".

import torch
import torch.nn.functional as F
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

    hidden_states = qwen3_5.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    bounds = cu_seqlens.tolist()
    segments = list(zip(bounds[:-1], bounds[1:]))

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if self.causal_conv1d_fn is not None:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_idx = torch.repeat_interleave(
            torch.arange(len(segments), device=mixed_qkv.device, dtype=torch.int32), lengths.to(mixed_qkv.device)
        ).unsqueeze(0)
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
    else:
        mixed_qkv = torch.cat(
            [F.silu(self.conv1d(mixed_qkv[:, :, s:e])[:, :, : e - s]) for s, e in segments], dim=-1
        )

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if qwen3_5.chunk_gated_delta_rule is not None and self.chunk_gated_delta_rule is qwen3_5.chunk_gated_delta_rule:
        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens.to(query.device),
        )
    else:
        core_attn_out = torch.cat(
            [
                self.chunk_gated_delta_rule(
                    query[:, s:e],
                    key[:, s:e],
                    value[:, s:e],
                    g=g[:, s:e],
                    beta=beta[:, s:e],
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )[0]
                for s, e in segments
            ],
            dim=1,
        )

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


def apply_packfix():
    """Replace the two forwards once. Return True when this call applied the patch."""
    if _ORIGINAL:
        return False
    _ORIGINAL["layer"] = qwen3_5.Qwen3_5DecoderLayer.forward
    _ORIGINAL["gdn"] = qwen3_5.Qwen3_5GatedDeltaNet.forward
    qwen3_5.Qwen3_5DecoderLayer.forward = decoder_layer_forward
    qwen3_5.Qwen3_5GatedDeltaNet.forward = gated_delta_net_forward
    return True
