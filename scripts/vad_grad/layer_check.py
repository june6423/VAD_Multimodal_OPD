"""G1 layer check (HANDOFF-VLM-VAD-GRAD-001-R1): one Qwen3.5-4B decoder layer of each type, forward and backward.

The text model is built from the Qwen3.5-4B config with random weights (bf16) and only the first layers up to the
first full-attention layer (3 GatedDeltaNet layers and 1 full-attention layer). One random sequence goes through it.
The check passes when the output, the loss and the gradient of every parameter of the first GatedDeltaNet layer
and of the full-attention layer are finite, and the GatedDeltaNet fast path (causal-conv1d and fla kernels, as in A1
training) is available on CUDA.
Usage: python layer_check.py <model dir with config.json> [device cuda|cpu] [attn flash_attention_2|sdpa] [tokens]
Prints one line "layer_check <json>" and exits 0 only when the check passes.
"""

from __future__ import annotations

import json
import sys

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq


def main() -> None:
    model_dir = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    attn = sys.argv[3] if len(sys.argv) > 3 else "flash_attention_2"
    tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    text = AutoConfig.from_pretrained(model_dir).text_config
    types = list(text.layer_types)
    n_layers = types.index("full_attention") + 1
    text.num_hidden_layers, text.layer_types = n_layers, types[:n_layers]
    torch.manual_seed(0)
    model = mq.Qwen3_5TextModel._from_config(text, attn_implementation=attn, dtype=torch.bfloat16).to(device)
    ids = torch.randint(0, text.vocab_size, (1, tokens), device=device)
    hidden = model(input_ids=ids).last_hidden_state
    loss = hidden.float().pow(2).mean()
    loss.backward()
    layers = {}
    for idx in (0, n_layers - 1):
        params = list(model.layers[idx].named_parameters())
        layers[f"{idx}:{types[idx]}"] = {
            "parameters": len(params),
            "all_have_grad": all(p.grad is not None for _, p in params),
            "grads_finite": all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in params),
            "grad_norm": float(torch.sqrt(sum(p.grad.float().pow(2).sum() for _, p in params if p.grad is not None))),
        }
    result = {"device": device, "attn": attn, "tokens": tokens, "layers_built": n_layers, "layer_types": types[:n_layers],
              "gated_delta_fast_path": bool(mq.is_fast_path_available), "output_finite": bool(torch.isfinite(hidden).all()),
              "loss": float(loss), "checked": layers}
    ok = result["output_finite"] and all(v["all_have_grad"] and v["grads_finite"] for v in layers.values())
    if device == "cuda":
        ok = ok and result["gated_delta_fast_path"]
    result["pass"] = ok
    print("layer_check", json.dumps(result))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
