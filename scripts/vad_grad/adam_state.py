"""Stage 0c: Adam state of A1 and A2 (HANDOFF-VLM-VAD-GRAD-001-R1 Part A item 3; DIRECTIVE-002 D6, D8).

Reads exp_avg (m) and exp_avg_sq (v) of the FSDP1 flat parameters (ckpt.units, ckpt.flat_state) at steps 80 and 130.
Checks: the size of each unit equals the unpadded flat length (ckpt.flat_state); the optimizer step t equals the
global step (D6.1, else STOP); the jump of mean log10 v at each tensor boundary inside a unit, against jumps at
interior positions (D8). For the 248 tensors of G3_PARAMS.txt:
  v_hat = v / (1 - 0.999^t), u = (m / (1 - 0.9^t)) / (sqrt(v_hat) + 1e-8)   (torch AdamW update direction)
  per tensor fp64: <u_A1, u_A1>, <u_A2, u_A2>, <u_A1, u_A2>, <u_X, d_X>, <d_X, d_X> with d_X = theta_X_s - theta_0
  the fraction of elements with 1e-8 >= 0.1 sqrt(v_hat) (D6.2)
Saves v of each arm and step as V_<arm>_<step>.safetensors (fp32, 248 tensors, metadata step) for moments.py.
Usage: python adam_state.py <hf config dir> <base model dir> <G3_PARAMS.txt> <out dir> <step>=<A1 actor dir>=<A2 actor dir> ...
Writes ADAM_DOTS.json (per tensor), ADAM.json (tables, checks).
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ckpt  # noqa: E402

EPS, B1, B2 = 1e-8, 0.9, 0.999
WINDOW = 4096


def dot(a, b) -> float:
    return float(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()))


def log_v_jumps(flat_v: torch.Tensor, params, rng: random.Random) -> tuple[list[float], list[float]]:
    """|mean log10 v| differences across each tensor boundary, and at random interior positions (control)."""
    lv = torch.log10(flat_v.double().clamp_min(1e-30))
    bounds, control, start = [], [], 0
    sizes = [int(torch.Size(s).numel()) for _, s in params]
    for i, n in enumerate(sizes):
        if n > 2 * WINDOW:
            p = start + rng.randrange(WINDOW, n - WINDOW)
            control.append(abs(float(lv[p - WINDOW:p].mean() - lv[p:p + WINDOW].mean())))
        end = start + n
        if i + 1 < len(sizes) and n >= 8 and sizes[i + 1] >= 8:
            k = min(WINDOW, n, sizes[i + 1])
            bounds.append(abs(float(lv[end - k:end].mean() - lv[end:end + k].mean())))
        start = end
    return bounds, control


def table(rows: dict, names: list[str]) -> dict:
    groups = defaultdict(lambda: defaultdict(float))
    for n in names:
        layer, kind = ckpt.group_of(n)
        for g in (f"kind:{kind}", f"layer:{layer}", "total"):
            for k, v in rows[n].items():
                groups[g][k] += v
    out = {}
    for g, s in groups.items():
        cos = lambda ab, aa, bb: ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else float("nan")  # noqa: E731
        out[g] = {"u_norm_a1": math.sqrt(s["u1u1"]), "u_norm_a2": math.sqrt(s["u2u2"]),
                  "cos_u_a1_a2": cos(s["u1u2"], s["u1u1"], s["u2u2"]),
                  "cos_u_d_a1": cos(s["u1d1"], s["u1u1"], s["d1d1"]), "cos_u_d_a2": cos(s["u2d2"], s["u2u2"], s["d2d2"]),
                  "eps_fraction_a1": s["eps1"] / s["n"], "eps_fraction_a2": s["eps2"] / s["n"]}
    return out


def main():
    hf_dir, base_dir, g3_params, out = sys.argv[1:5]
    specs = [s.split("=") for s in sys.argv[5:]]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layout = ckpt.units(hf_dir)
    grad248 = set(ckpt.grad_set(g3_params))
    where, base = ckpt.base_tensors(base_dir)
    rng = random.Random(0)
    result, all_rows, stop = {"units": len(layout), "steps": {}}, {}, []
    for step, a1_dir, a2_dir in specs:
        start = time.time()
        step = int(step)
        opt = {"a1": ckpt.open_shards(a1_dir, "optim"), "a2": ckpt.open_shards(a2_dir, "optim")}
        mdl = {"a1": ckpt.open_shards(a1_dir, "model"), "a2": ckpt.open_shards(a2_dir, "model")}
        info = {}
        for arm in ("a1", "a2"):
            t = ckpt.optimizer_step(opt[arm])
            info[f"optimizer_step_{arm}"] = sorted(t)
            if t != {float(step)}:
                stop.append(f"{arm} step {step}: optimizer step values {sorted(t)}")
            if len(opt[arm][0]["state"]) != len(layout):
                raise SystemExit(f"STOP: {len(opt[arm][0]['state'])} flat parameters, layout has {len(layout)} units")
        c1, c2 = 1 - B1 ** step, 1 - B2 ** step
        rows, vsave = {}, {"a1": {}, "a2": {}}
        jumps = {"a1": ([], []), "a2": ([], [])}
        for i, (unit, params) in enumerate(layout):
            numel = sum(int(torch.Size(s).numel()) for _, s in params)
            sl = {}
            for arm in ("a1", "a2"):
                fv = ckpt.flat_state(opt[arm], i, "exp_avg_sq", numel)
                fm = ckpt.flat_state(opt[arm], i, "exp_avg", numel)
                b, c = log_v_jumps(fv, params, rng)
                jumps[arm][0].extend(b)
                jumps[arm][1].extend(c)
                sl[arm] = (ckpt.unit_slices(fm, params), ckpt.unit_slices(fv, params))
            for name, _ in params:
                if name not in grad248:
                    continue
                t0 = base(name)
                u, d, eps_n = {}, {}, {}
                for arm in ("a1", "a2"):
                    m, v = sl[arm][0][name].float(), sl[arm][1][name].float()
                    vhat = v / c2
                    u[arm] = (m / c1) / (vhat.sqrt() + EPS)
                    d[arm] = ckpt.model_tensor(mdl[arm], name) - t0
                    eps_n[arm] = float((EPS >= 0.1 * vhat.sqrt()).sum())
                    vsave[arm][name] = v.contiguous()
                rows[name] = {"n": float(v.numel()), "u1u1": dot(u["a1"], u["a1"]), "u2u2": dot(u["a2"], u["a2"]),
                              "u1u2": dot(u["a1"], u["a2"]), "u1d1": dot(u["a1"], d["a1"]), "u2d2": dot(u["a2"], d["a2"]),
                              "d1d1": dot(d["a1"], d["a1"]), "d2d2": dot(d["a2"], d["a2"]),
                              "eps1": eps_n["a1"], "eps2": eps_n["a2"]}
        for arm in ("a1", "a2"):
            save_file(vsave[arm], str(out / f"V_{arm.upper()}_{step}.safetensors"), metadata={"step": str(step)})
        info["jumps"] = {arm: {"boundaries": len(b), "boundary_median": sorted(b)[len(b) // 2] if b else None,
                               "control": len(c), "control_median": sorted(c)[len(c) // 2] if c else None,
                               "control_p95": sorted(c)[int(0.95 * len(c))] if c else None,
                               "boundaries_above_control_p95": sum(1 for x in b if c and x > sorted(c)[int(0.95 * len(c))])}
                         for arm, (b, c) in jumps.items()}
        info["tensors"] = len(rows)
        info["seconds"] = round(time.time() - start, 1)
        info["tables"] = table(rows, sorted(rows))
        result["steps"][str(step)] = info
        all_rows[str(step)] = rows
    (out / "ADAM_DOTS.json").write_text(json.dumps(all_rows))
    (out / "ADAM.json").write_text(json.dumps(result, indent=1))
    if stop or any(i["tensors"] != 248 for i in result["steps"].values()):
        (out / "STOP.txt").write_text("\n".join(stop) + "\n")
        raise SystemExit(f"STOP stage 0c: {stop}")


if __name__ == "__main__":
    main()
