"""Stage 0b: weight displacement of A1 and A2 (HANDOFF-VLM-VAD-GRAD-001-R1 Part A item 2; DIRECTIVE-002 D5).

theta_0 = Base Qwen3.5-4B. theta of A1 and A2 at steps 80 and 130 = the fp32 FSDP model shards (ckpt.model_tensor).
Check (D5.1): each rebuilt fp32 tensor rounded to bf16 equals the tensor of the merged model bit for bit. One
difference stops the stage (exit 1, STOP.txt) after all tensors are checked.
Per tensor, fp64: d_X_s = theta_X_s - theta_0, and m_X = theta_X_130 - theta_X_80 (X = A1, A2; s = 80, 130):
<d_A1_s, d_A1_s>, <d_A2_s, d_A2_s>, <d_A1_s, d_A2_s>, and the same for m. Tables over the 248 tensors of G3_PARAMS.txt
and over all language-model tensors (model.language_model.* and lm_head.*), by tensor kind, by layer, and in total.
Usage: python displacement.py <base model dir> <G3_PARAMS.txt> <out dir> \
         <A1 actor 80> <A1 actor 130> <A1 merged 80> <A1 merged 130> <A2 actor 80> <A2 actor 130> <A2 merged 80> <A2 merged 130>
Writes DISPLACEMENT_DOTS.json (per tensor) and DISPLACEMENT.json (tables, bit check, base name check).
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ckpt  # noqa: E402

KEYS = ("a1_80", "a1_130", "a2_80", "a2_130")


def dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()))


def tensor_dots(t0, th) -> dict:
    d = {k: th[k] - t0 for k in KEYS}
    m1, m2 = th["a1_130"] - th["a1_80"], th["a2_130"] - th["a2_80"]
    out = {}
    for s in ("80", "130"):
        a, b = d[f"a1_{s}"], d[f"a2_{s}"]
        out.update({f"d{s}_a1a1": dot(a, a), f"d{s}_a2a2": dot(b, b), f"d{s}_a1a2": dot(a, b)})
    out.update({"m_a1a1": dot(m1, m1), "m_a2a2": dot(m2, m2), "m_a1a2": dot(m1, m2)})
    return out


def table(rows: dict[str, dict], names: list[str]) -> dict:
    """Norms and cosines by kind, by layer, and in total, from per-tensor dot products."""
    groups = defaultdict(lambda: defaultdict(float))
    for n in names:
        layer, kind = ckpt.group_of(n)
        for g in (f"kind:{kind}", f"layer:{layer}", "total"):
            for k, v in rows[n].items():
                if k != "numel":
                    groups[g][k] += v
    out = {}
    for g, s in groups.items():
        e = {}
        for p in ("d80", "d130", "m"):
            n1, n2 = math.sqrt(s[f"{p}_a1a1"]), math.sqrt(s[f"{p}_a2a2"])
            e[f"{p}_norm_a1"], e[f"{p}_norm_a2"] = n1, n2
            e[f"{p}_cos_a1_a2"] = s[f"{p}_a1a2"] / (n1 * n2) if n1 > 0 and n2 > 0 else float("nan")
        out[g] = e
    return out


def main():
    base_dir, g3_params, out = sys.argv[1:4]
    paths = sys.argv[4:12]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    shards = {k: ckpt.open_shards(paths[i], "model") for i, k in zip((0, 1, 4, 5), KEYS)}
    merged = {k: safe_open(str(Path(paths[i]) / "model.safetensors"), framework="pt") for i, k in zip((2, 3, 6, 7), KEYS)}
    where, base = ckpt.base_tensors(base_dir)
    grad248 = ckpt.grad_set(g3_params)
    names = [n for n in shards["a1_80"][0] if torch.is_floating_point(shards["a1_80"][0][n])]
    rows, bit = {}, {"checked": 0, "differ": [], "not_in_merged": []}
    missing_base = []
    for n in names:
        th = {k: ckpt.model_tensor(shards[k], n) for k in KEYS}
        for k in KEYS:
            if n in merged[k].keys():
                bit["checked"] += 1
                if not torch.equal(th[k].to(torch.bfloat16).view(torch.int16),
                                   merged[k].get_tensor(n).to(torch.bfloat16).view(torch.int16)):
                    bit["differ"].append(f"{k}:{n}")
            else:
                bit["not_in_merged"].append(f"{k}:{n}")
        if n not in where:
            missing_base.append(n)
            continue
        rows[n] = {"numel": th["a1_80"].numel(), **tensor_dots(base(n), th)}
        del th
    lm = [n for n in rows if n.startswith("model.language_model.") or n.startswith("lm_head.")]
    result = {"seconds": round(time.time() - start, 1), "tensors": len(names), "bit_check": {
                  "checked": bit["checked"], "differ_count": len(bit["differ"]), "differ": bit["differ"][:50],
                  "not_in_merged": bit["not_in_merged"]},
              "missing_in_base": missing_base, "grad248_present": sum(1 for n in grad248 if n in rows),
              "tables": {"grad248": table(rows, [n for n in grad248 if n in rows]), "language_model": table(rows, lm)}}
    (out / "DISPLACEMENT_DOTS.json").write_text(json.dumps(rows))
    (out / "DISPLACEMENT.json").write_text(json.dumps(result, indent=1))
    if bit["differ"] or result["grad248_present"] != 248:
        (out / "STOP.txt").write_text(f"bit check differences {len(bit['differ'])}, grad248 present "
                                      f"{result['grad248_present']}\n")
        raise SystemExit("STOP stage 0b: see DISPLACEMENT.json")


if __name__ == "__main__":
    main()
