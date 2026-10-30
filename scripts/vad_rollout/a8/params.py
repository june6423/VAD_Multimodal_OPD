"""A8 gradient set (HANDOFF-VLM-VAD-ROLLOUT-001-R9, Part A item 2).

All 2-D weight matrices of the language-model decoder layers: attention, linear-attention, and MLP projections.
Excluded: embeddings, lm_head, norms, conv1d weights, and the vision tower.
Usage: python params.py <model or config dir>  (builds the model on the meta device and prints the list)
"""

from __future__ import annotations

import sys

import torch

DECODER_MARKER = ".language_model.layers."
EXCLUDED = ("visual", "embed", "lm_head", "norm", "conv1d")


def is_grad_param(name: str, param: torch.Tensor) -> bool:
    return param.dim() == 2 and DECODER_MARKER in name and not any(word in name for word in EXCLUDED)


def grad_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if is_grad_param(name, p)]


def main() -> None:
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(sys.argv[1])
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    selected = grad_params(model)
    for name, p in selected:
        print(name, tuple(p.shape), p.numel())
    others = [(n, tuple(p.shape)) for n, p in model.named_parameters() if not is_grad_param(n, p)]
    excluded_2d_layers = [n for n, s in others if len(s) == 2 and DECODER_MARKER in n]
    print("selected tensors", len(selected), "parameters", sum(p.numel() for _, p in selected))
    print("model tensors", len(list(model.parameters())), "parameters", sum(p.numel() for p in model.parameters()))
    print("2-D decoder-layer tensors excluded by name:", excluded_2d_layers)


if __name__ == "__main__":
    main()
