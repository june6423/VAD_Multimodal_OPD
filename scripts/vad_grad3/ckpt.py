"""Read the A1 and A2 FSDP checkpoints on CPU (HANDOFF-VLM-VAD-GRAD-001-R1 Part A items 2 and 3).

Model shards (model_world_size_6_rank_<r>.pt, SHARDED_STATE_DICT): one DTensor per parameter, Shard(0). The full fp32
tensor is the concatenation of the 6 local tensors along dim 0.
Optimizer shards (optim_world_size_6_rank_<r>.pt): torch optimizer state of the FSDP1 flat parameters
(use_orig_params False). Flat parameter i is FSDP unit i: the root, then each module whose class is in
model._no_split_modules, in named_modules order. A unit holds the parameters of its module that no nested unit holds,
in named_parameters order, without padding between them; the flat tensor is padded to a multiple of 6 and rank r
holds chunk r. units() gives the layout; flat_state() and unit_slices() rebuild the per-parameter state.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

WORLD = 6


def open_shards(actor_dir: str, kind: str) -> list[dict]:
    return [torch.load(Path(actor_dir) / f"{kind}_world_size_{WORLD}_rank_{r}.pt", map_location="cpu",
                       weights_only=False, mmap=True) for r in range(WORLD)]


def model_tensor(shards: list[dict], name: str) -> torch.Tensor:
    parts = [s[name].to_local() if hasattr(s[name], "to_local") else s[name] for s in shards]
    full = torch.cat([p for p in parts if p.numel() or p.dim() == 0], dim=0) if parts[0].dim() else parts[0]
    shape = tuple(shards[0][name].shape)
    if tuple(full.shape) != shape:
        raise ValueError(f"{name}: rebuilt shape {tuple(full.shape)} differs from {shape}")
    return full.float()


def units(hf_config_dir: str) -> list[tuple[str, list[tuple[str, torch.Size]]]]:
    """[(unit module name, [(parameter name, shape), ...]), ...] in flat-parameter order."""
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(hf_config_dir)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg)
    wrap = set(model._no_split_modules or [])
    mods = [("", model)] + [(n, m) for n, m in model.named_modules() if type(m).__name__ in wrap]
    nested = [n for n, _ in mods[1:]]
    out, seen = [], set()
    for name, mod in mods:
        params = []
        for pn, p in mod.named_parameters(prefix=name):
            inner = any(pn.startswith(u + ".") for u in nested if u != name and (name == "" or u.startswith(name + ".")))
            if not inner and id(p) not in seen:
                seen.add(id(p))
                params.append((pn, p.shape))
        out.append((name, params))
    return out


def flat_state(shards: list[dict], index: int, key: str, numel: int) -> torch.Tensor:
    """The unpadded flat tensor of one optimizer state entry of flat parameter `index`."""
    flat = torch.cat([s["state"][index][key].reshape(-1) for s in shards])
    if not numel <= flat.numel() < numel + WORLD:
        raise ValueError(f"flat parameter {index}: {flat.numel()} elements for {numel} parameters")
    return flat[:numel]


def unit_slices(flat: torch.Tensor, params: list[tuple[str, torch.Size]]) -> dict[str, torch.Tensor]:
    out, start = {}, 0
    for name, shape in params:
        n = int(torch.Size(shape).numel())
        out[name] = flat[start:start + n].reshape(shape)
        start += n
    return out


def optimizer_step(shards: list[dict]) -> set[float]:
    return {float(s["state"][i]["step"]) for s in shards for i in s["state"]}


def base_tensors(model_dir: str):
    """name -> loader of the Base (bf16) tensor as fp32, from model.safetensors or its index."""
    from safetensors import safe_open

    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    handles = [safe_open(str(d / f), framework="pt") for f in files]
    where = {k: h for h in handles for k in h.keys()}
    return where, (lambda name: where[name].get_tensor(name).float())


def grad_set(g3_params: str) -> list[str]:
    """The 248 tensor names of TASK-VLM-VAD-ROLLOUT-001-047 G3_PARAMS.txt (lines '<name> (<shape>) <numel>')."""
    names = [line.split(" ", 1)[0] for line in Path(g3_params).read_text().splitlines()
             if line.startswith("model.") and "(" in line]
    if len(names) != 248:
        raise ValueError(f"G3_PARAMS.txt lists {len(names)} tensors, not 248")
    return names


def group_of(name: str) -> tuple[str, str]:
    """(layer, kind) of a parameter name: kind without the layer index, layer 'layers.<i>' or 'other'."""
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        return f"layers.{parts[i + 1]}", ".".join(parts[i + 2:])
    return "other", name
