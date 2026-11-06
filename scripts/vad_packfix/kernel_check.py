"""Diagnosis of the G2 failure of TASK-VLM-VAD-PACKFIX-001-001 (row-1 values changed with the fix).

Question: is the difference a fault of the row boundaries, or a numerical difference of the varlen kernels
(chunk_gated_delta_rule with cu_seqlens) against the kernels without cu_seqlens?

Part 1 (one linear-attention layer, layer 0 weights of Qwen3.5-4B, 3 rows of real text embeddings):
  max |out_varlen_packed[row] - out_original_one_row[row]| and max |out_varlen_packed[row] - out_varlen_one_row[row]|,
  in bfloat16 and in float32.
Part 2 (full model, text rows, bfloat16, flash_attention_2): the log-probability sum L of each row for
  original one-row forward (U), fixed packed forward (P), unfixed packed forward (P0), and one-row forward through
  the fixed code with cu_seqlens [0, n] (V).
Usage: python kernel_check.py <model dir> <output json>
"""

import json
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5

from verl.models.transformers import qwen3_5_packfix as packfix

model_dir, out_path = sys.argv[1], sys.argv[2]
LENGTHS = (600, 64, 900)
dev = "cuda"
result = {"lengths": LENGTHS, "fast_path": bool(qwen3_5.is_fast_path_available)}

tok = AutoTokenizer.from_pretrained(model_dir)
text = open(f"{model_dir}/README.md").read()
ids_all = tok(text, return_tensors="pt").input_ids[0]
starts = (0, 700, 800)
rows = [ids_all[s : s + n] for s, n in zip(starts, LENGTHS)]
result["readme_tokens"] = int(ids_all.numel())
assert all(r.numel() == n for r, n in zip(rows, LENGTHS))


def bounds(lengths):
    b = [0]
    for n in lengths:
        b.append(b[-1] + n)
    return b


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


# Part 1: one layer
config = AutoConfig.from_pretrained(model_dir)
text_config = config.text_config
prefix = "model.language_model.layers.0.linear_attn."
weights, embed = {}, None
index = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
for name, file in index.items():
    if name.startswith(prefix) or name == "model.language_model.embed_tokens.weight":
        with safe_open(f"{model_dir}/{file}", framework="pt") as f:
            t = f.get_tensor(name)
        if name.startswith(prefix):
            weights[name[len(prefix) :]] = t
        else:
            embed = t
packfix.apply_packfix()
part1 = {}
for dtype in (torch.bfloat16, torch.float32):
    layer = qwen3_5.Qwen3_5GatedDeltaNet(text_config, layer_idx=0).to(dev, dtype)
    layer.load_state_dict(weights)
    h = [embed[r].to(dev, dtype) for r in rows]
    h = [x / x.float().pow(2).mean(-1, keepdim=True).sqrt().to(dtype) for x in h]
    b = bounds(LENGTHS)
    with torch.no_grad():
        layer._packfix_cu_seqlens = torch.tensor(b, device=dev)
        packed = packfix.gated_delta_net_forward(layer, torch.cat(h)[None])[0]
        varlen_one = []
        for x in h:
            layer._packfix_cu_seqlens = torch.tensor([0, x.shape[0]], device=dev)
            varlen_one.append(packfix.gated_delta_net_forward(layer, x[None])[0])
        layer._packfix_cu_seqlens = None
        orig_one = [packfix._ORIGINAL["gdn"](layer, x[None])[0] for x in h]
        orig_packed = packfix._ORIGINAL["gdn"](layer, torch.cat(h)[None])[0]
    rows_out = []
    for i in range(3):
        p = packed[b[i] : b[i + 1]]
        rows_out.append(
            {
                "row": i + 1,
                "scale": float(orig_one[i].float().abs().max()),
                "varlen_packed_vs_original_one_row": maxdiff(p, orig_one[i]),
                "varlen_packed_vs_varlen_one_row": maxdiff(p, varlen_one[i]),
                "varlen_one_row_vs_original_one_row": maxdiff(varlen_one[i], orig_one[i]),
                "unfixed_packed_vs_original_one_row": maxdiff(orig_packed[b[i] : b[i + 1]], orig_one[i]),
            }
        )
    part1[str(dtype).replace("torch.", "")] = rows_out
    del layer
result["part1_layer0"] = part1
print(json.dumps(part1, indent=1), flush=True)

# Part 2: full model
model = AutoModelForImageTextToText.from_pretrained(
    model_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).to(dev).eval()


def row_logprob_sums(input_ids, position_ids, lengths):
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(dev), position_ids=position_ids.to(dev), use_cache=False).logits[0]
    lp = F.log_softmax(logits.float(), -1)
    sums, b = [], bounds(lengths)
    ids = input_ids[0].to(dev)
    for i in range(len(lengths)):
        s, e = b[i], b[i + 1]
        sums.append(float(lp[s : e - 1].gather(-1, ids[s + 1 : e, None]).sum()))
    return sums


packed_ids = torch.cat(rows)[None]
packed_pos = torch.cat([torch.arange(n) for n in LENGTHS])[None]
P = row_logprob_sums(packed_ids, packed_pos, LENGTHS)  # patch is on: fixed packed
U = [row_logprob_sums(r[None], torch.arange(r.numel())[None], [r.numel()])[0] for r in rows]  # one row: original code
real = packfix.packed_cu_seqlens
packfix.packed_cu_seqlens = lambda pos: torch.tensor([0, pos.shape[-1]], device=pos.device)
V = [row_logprob_sums(r[None], torch.arange(r.numel())[None], [r.numel()])[0] for r in rows]  # one row: varlen code
packfix.packed_cu_seqlens = lambda pos: None
P0 = row_logprob_sums(packed_ids, packed_pos, LENGTHS)  # unfixed packed
packfix.packed_cu_seqlens = real


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


part2 = []
for i in range(3):
    part2.append(
        {
            "row": i + 1,
            "tokens": LENGTHS[i],
            "L_U": U[i],
            "L_V": V[i],
            "L_P": P[i],
            "L_P0": P0[i],
            "rel_P_vs_U": rel(P[i], U[i]),
            "rel_P_vs_V": rel(P[i], V[i]),
            "rel_V_vs_U": rel(V[i], U[i]),
            "rel_P0_vs_U": rel(P0[i], U[i]),
        }
    )
result["part2_full_model_bf16"] = part2
print(json.dumps(part2, indent=1), flush=True)
json.dump(result, open(out_path, "w"), indent=1)
