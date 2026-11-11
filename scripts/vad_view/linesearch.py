"""Line search of HANDOFF-VLM-VAD-VIEW-001-R3 Part A item 7 with DIRECTIVE-VLM-VAD-VIEW-001-001 (fp32).

Directions at one checkpoint: u = -D Gbar for each fold mean MEAN__<view>, for MEAN__V and for MEAN__geo (vgram.py;
10 directions with 8 views), D the GRAD-003
metric (grads.metric_d), scaled to the raw norm 1. Step norm s = ||Delta_A1(0,80)|| / 80 (raw diagonal of
D__A1_0_80 in GRAM_<ck>.json). Point (u, alpha): theta = theta0 + alpha s u on the 248 tensors of G3_PARAMS.txt,
alpha in ALPHAS (1, 4, 16).
Model (directive item 1): the checkpoint loaded with AutoModelForImageTextToText in fp32, attn_implementation "sdpa",
no verl monkey patch; every parameter is rounded to bf16 and back (theta0 = the bf16 parameters of the official
forward), then the update is computed and applied in fp32.
Measure: for each held-out item of sets A and B, the official evaluation input (heldout.EvalInputs: the model's own
chat template with enable_thinking False, the full image, the processor defaults), the fp32 log softmax of the logits
at the last prompt position (the first response token); letter log probabilities, log R, m, letter argmax correct.
Points: a0-r1 and a0-r2 (alpha 0 two times, the numerical floor, directive item 3), then <direction>@a<alpha>.
For each point <out>/points/<point>.meta.json holds the bf16 changed fraction (directive item 5: the fraction of
the elements of the 248 tensors whose bf16 cast differs from theta0), and <point>.jsonl one line per item (resume: a
done item is skipped, a point with all items is skipped).
LINESEARCH_<ck>.json: for each point and set, the mean log R, the mean m, the accuracy, the change from a0-r1, the
floor |a0-r2 - a0-r1| of each quantity and a mark when |change| < 2 x floor (directive item 7); at alpha 0 the
difference fp32 (a0-r1) minus bf16 (HELDOUT_VALUES.jsonl of heldout.py grad, the A1 actor forward) on the same items
(directive item 4); for each direction the norm ||D Gbar|| and its value from the M2 Gram file (consistency check).

Speed-up (DIRECTIVE-VLM-VAD-VIEW-001-002 items 3 and 5): the items run in chunks. Each item of a chunk is processed
once (the processor inputs), and its vision output (model.model.get_image_features at theta0; the vision tower is not
in G3_PARAMS.txt, so it is the same at every point) is computed once and kept on the GPU. Then every point runs on the
chunk: the forward gets the cached vision output in place of a new vision forward (VisionCache), so the language
model forward and the values do not change. A chunk closes when the cached vision outputs reach VCACHE_GB.
a0-r2 (the second alpha-0 pass) runs only with LS_FLOOR_PASS=1 (the pilot); it processes each item again and runs
the vision tower again (no cache), so its floor also checks the cache. Without a0-r2 the floor values are null.
VISION.jsonl: one line per item and chunk (seconds of the input wait and the vision forward, image tokens, bytes).

Usage: python linesearch.py <repo> <model dir> <G3_PARAMS.txt> <V file> <ck dir> <HELDOUT.json> <bench dir>
         <HELDOUT_VALUES.jsonl> <checkpoint> <out dir>
Env: DIRECTIONS (all, or comma list of MEAN names such as MEAN__V), ALPHAS (1,4,16), LS_PER_SET (0 = all items; else
the first n items of set A and of set B), STOP_AFTER (0; exit 75 once after this many new item lines), PREFETCH (4),
LS_FLOOR_PASS (0), VCACHE_GB (40).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP_EXIT = 75
STEPS = 80
QUANTITIES = ("mean_log_R", "mean_m", "accuracy")


def point_names(directions: list[str], alphas: list[float], floor_pass: bool = True) -> list[tuple[str, str | None, float]]:
    pts = [("a0-r1", None, 0.0)] + ([("a0-r2", None, 0.0)] if floor_pass else [])
    return pts + [(f"{d}@a{a:g}", d, a) for d in directions for a in alphas]


class VisionCache:
    """model.model.get_image_features with a cache: for the item `current` it returns the vision output kept by
    put() (the same tensors the call gives at any point, because the vision tower is not changed); with current None
    it calls the model function."""

    def __init__(self, model):
        self.inner = model.model
        self.orig = model.model.get_image_features
        self.cache: dict = {}
        self.bytes = 0
        self.current = None
        self.inner.get_image_features = self.get

    def get(self, pixel_values, image_grid_thw=None, **kwargs):
        if self.current is None:
            return self.orig(pixel_values, image_grid_thw, **kwargs)
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        import torch

        grid, embeds = self.cache[self.current]
        if not torch.equal(grid, image_grid_thw):
            raise RuntimeError(f"vision cache: item {self.current} has another image_grid_thw")
        return BaseModelOutputWithPooling(pooler_output=embeds)

    def put(self, index: int, pixel_values, image_grid_thw) -> int:
        """Run the vision tower once for the item and keep its output; returns the bytes kept."""
        out = self.orig(pixel_values, image_grid_thw, return_dict=True)
        embeds = tuple(out.pooler_output)
        n = sum(e.numel() * e.element_size() for e in embeds)
        self.cache[index] = (image_grid_thw, embeds)
        self.bytes += n
        return n

    def clear(self):
        self.cache, self.bytes, self.current = {}, 0, None


def items_of(sel: dict, per_set: int) -> list[dict]:
    out = []
    for s in ("A", "B"):
        its = [it for it in sel["items"] if it["set"] == s]
        out += its[:per_set] if per_set else its
    return out


def gram_norm2(gram: dict, members: list[str]) -> float:
    """||D mean(members)||^2 from the M2 Gram."""
    idx = [gram["names"].index(n) for n in members]
    m2 = gram["M2"]
    return sum(m2[i][j] for i in idx for j in idx) / len(idx) ** 2


def direction_members(gram: dict, direction: str) -> list[str]:
    fm = gram["fold_means"]
    view = direction[len("MEAN__"):]
    if view == "geo":
        return gram["geo_fold_mean"]
    return [n for v in sorted(fm) for n in fm[v]] if view == "V" else fm[view]


def set_means(lines: list[dict]) -> dict:
    out = {}
    for s in ("A", "B"):
        xs = [x for x in lines if x["set"] == s]
        if xs:
            out[s] = {"n": len(xs), "mean_log_R": sum(x["log_R"] for x in xs) / len(xs),
                      "mean_m": sum(x["m"] for x in xs) / len(xs),
                      "accuracy": sum(bool(x["correct"]) for x in xs) / len(xs)}
    return out


def summarize(points: list[tuple[str, str | None, float]], lines: dict[str, list[dict]], metas: dict[str, dict],
              bf16: list[dict]) -> dict:
    """The LINESEARCH table from the item lines of each point (lines[point]) and the bf16 values."""
    means = {p: set_means(lines[p]) for p, _, _ in points}
    base = means["a0-r1"]
    by1 = {x["index"]: x for x in lines["a0-r1"]}
    floor = item_floor = None
    if "a0-r2" in means:
        rep = means["a0-r2"]
        floor = {s: {q: abs(rep[s][q] - base[s][q]) for q in QUANTITIES} for s in base}
        by2 = {x["index"]: x for x in lines["a0-r2"]}
        item_floor = {q: max(abs(by1[i][q] - by2[i][q]) for i in by1) for q in ("log_R", "m")}
    bfi = {x["index"]: x for x in bf16 if x["index"] in by1}
    if set(bfi) != set(by1):
        raise SystemExit(f"STOP: HELDOUT_VALUES.jsonl lacks {len(set(by1) - set(bfi))} line-search items")
    bf_means = set_means(list(bfi.values()))
    fp32_vs_bf16 = {
        "means": {s: {q: base[s][q] - bf_means[s][q] for q in QUANTITIES} for s in base},
        "item_max_abs": {q: max(abs(by1[i][q] - bfi[i][q]) for i in by1) for q in ("log_R", "m")},
        "argmax_disagree": sum(bool(by1[i]["correct"]) != bool(bfi[i]["correct"]) for i in by1),
        "items": len(by1)}
    table = []
    for p, d, a in points:
        row = {"point": p, "direction": d, "alpha": a, "bf16_changed_fraction": metas.get(p, {}).get("bf16_changed_fraction"),
               "sets": {}}
        for s, v in means[p].items():
            ch = {q: v[q] - base[s][q] for q in QUANTITIES}
            row["sets"][s] = {**v, "change": ch,
                              "below_2x_floor": {q: abs(ch[q]) < 2 * floor[s][q] for q in QUANTITIES}
                              if floor is not None else None}
        table.append(row)
    return {"floor": floor, "item_floor": item_floor, "fp32_minus_bf16_alpha0": fp32_vs_bf16, "table": table}


def done_items(path: Path) -> list[dict]:
    return [json.loads(x) for x in open(path)] if path.exists() else []


def load_model(model_dir: str):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_dir, attn_implementation="sdpa")
    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.float32, config=config,
                                                        attn_implementation="sdpa")
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16).to(torch.float32)
    return model.cuda()


def main():
    (repo, model_dir, g3_params, v_file, ck_dir, heldout_json, bench_dir, bf16_values, checkpoint,
     out) = sys.argv[1:]
    sys.path.insert(0, str(HERE))
    import heldout

    heldout.setup(repo)
    import ckpt
    import grads as g3
    import numpy as np
    import params as gparams
    import torch

    out, ck_dir = Path(out), Path(ck_dir)
    (out / "points").mkdir(parents=True, exist_ok=True)
    gram = json.loads((ck_dir / f"GRAM_{checkpoint}.json").read_text())
    names = gram["names"]
    i_d = names.index("D__A1_0_80")
    s = math.sqrt(gram["raw"][i_d][i_d]) / STEPS
    found = sorted(p.name[:-len(".npy")] for p in ck_dir.glob("MEAN__*.npy"))
    sel_dirs = os.environ.get("DIRECTIONS", "all")
    directions = found if sel_dirs == "all" else sel_dirs.split(",")
    if not set(directions) <= set(found):
        raise SystemExit(f"STOP: directions {sorted(set(directions) - set(found))} have no MEAN file in {ck_dir}")
    alphas = [float(a) for a in os.environ.get("ALPHAS", "1,4,16").split(",")]
    floor_pass = os.environ.get("LS_FLOOR_PASS", "0") == "1"
    points = point_names(directions, alphas, floor_pass)
    sel = json.loads(Path(heldout_json).read_text())
    items = items_of(sel, int(os.environ.get("LS_PER_SET", "0")))
    bf16 = done_items(Path(bf16_values))
    tokens_bf16 = {x["index"]: x["prompt_tokens"] for x in bf16}
    stop_after = int(os.environ.get("STOP_AFTER", "0"))
    log = {"checkpoint": checkpoint, "model_dir": model_dir, "start": g3.now(), "step_norm_s": s,
           "delta_norm": s * STEPS, "directions": directions, "alphas": alphas, "items": len(items),
           "floor_pass": floor_pass, "vcache_gb": float(os.environ.get("VCACHE_GB", "40"))}

    t0 = time.time()
    model = load_model(model_dir)
    named = gparams.grad_params(model)
    if [n for n, _ in named] != ckpt.grad_set(g3_params):
        raise SystemExit("STOP: the model tensors differ from G3_PARAMS.txt (names or order)")
    params = [p for _, p in named]
    theta0 = [p.detach().clone() for p in params]
    log.update(load_seconds=round(time.time() - t0, 1), model_class=type(model).__name__,
               attn=model.config._attn_implementation, elements=sum(p.numel() for p in params))
    ev = heldout.EvalInputs(model_dir, bench_dir)
    ids = {it["index"]: ev.letter_ids(it["letters"]) for it in items}
    d_flat = None
    current = None  # (direction, u tensors)
    count = [0]
    pool = ThreadPoolExecutor(1)
    prefetch = int(os.environ.get("PREFETCH", "4"))

    def prepared(todo):
        q = deque()
        it = iter(todo)
        for x in it:
            q.append((x, pool.submit(ev.processor_inputs, x)))
            if len(q) >= prefetch:
                break
        while q:
            x, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, pool.submit(ev.processor_inputs, nxt)))
            yield x, fut.result()

    def direction(dname: str):
        nonlocal d_flat
        if d_flat is None:
            d_flat, d_info = g3.metric_d(v_file, ckpt.grad_set(g3_params))
            log["metric"] = d_info
        mean = np.load(ck_dir / f"{dname}.npy", mmap_mode="r")
        if mean.shape[0] != d_flat.numel():
            raise SystemExit(f"STOP: {dname} has {mean.shape[0]} elements, D {d_flat.numel()}")
        u, off, sq = [], 0, 0.0
        for p in params:
            n = p.numel()
            x = -(d_flat[off:off + n].cuda() * torch.from_numpy(np.asarray(mean[off:off + n])).cuda())
            sq += float(x.double().pow(2).sum())
            u.append(x.view(p.shape))
            off += n
        norm = math.sqrt(sq)
        if not math.isfinite(norm) or norm == 0:
            raise SystemExit(f"STOP: ||D Gbar|| of {dname} is {norm} (stop condition 3)")
        for x in u:
            x.div_(norm)
        expect = math.sqrt(gram_norm2(gram, direction_members(gram, dname)))
        g3.write_json(out / "points" / f"{dname}.direction.json",
                      {"direction": dname, "norm_D_Gbar": norm, "norm_from_M2_gram": expect,
                       "rel_diff": abs(norm - expect) / expect})
        return u

    vc = VisionCache(model)
    budget = float(os.environ.get("VCACHE_GB", "40")) * 1e9
    placeholder = torch.empty(0)  # pixel_values: the cached vision output replaces the vision forward
    done = {p: {x["index"] for x in done_items(out / "points" / f"{p}.jsonl")} for p, _, _ in points}
    log["chunks"] = []

    def set_point(p_name: str, dname: str | None, alpha: float):
        nonlocal current
        with torch.no_grad():
            if dname is None:
                for p, t in zip(params, theta0):
                    p.copy_(t)
                c = 0.0
            else:
                if current is None or current[0] != dname:
                    current = None
                    torch.cuda.empty_cache()
                    current = (dname, direction(dname))
                c = alpha * s
                for p, t, x in zip(params, theta0, current[1]):
                    torch.add(t, x, alpha=c, out=p)
            changed = sum(int((p.to(torch.bfloat16) != t.to(torch.bfloat16)).sum()) for p, t in zip(params, theta0))
        meta = {"point": p_name, "direction": dname, "alpha": alpha, "step": c,
                "bf16_changed_fraction": changed / log["elements"], "bf16_changed": changed}
        g3.write_json(out / "points" / f"{p_name}.meta.json", meta)

    def measure(f, p_name: str, it: dict, inp: dict, cached: bool):
        t1 = time.time()
        n_tok = int(inp["input_ids"].shape[1])
        if tokens_bf16.get(it["index"]) not in (None, n_tok):
            raise SystemExit(f"STOP: item {it['index']}: {n_tok} prompt tokens, HELDOUT_VALUES "
                             f"{tokens_bf16[it['index']]}")
        with torch.no_grad():
            if cached:
                vc.current = it["index"]
                args = {k: (placeholder if k == "pixel_values" else v.cuda()) for k, v in inp.items()}
            else:
                vc.current = None
                args = {k: v.cuda() for k, v in inp.items()}
            logits = model(**args, use_cache=False, logits_to_keep=1).logits[0, -1]
            vc.current = None
            lp = torch.log_softmax(logits.double(), -1)
            lt = lp[ids[it["index"]]]
            log_r, log_m = heldout.letter_values(lt, it["letters"].index(it["answer"]))
        vals = [float(x) for x in lt]
        if not all(math.isfinite(x) for x in vals):
            raise SystemExit(f"STOP: non-finite letter log probability for item {it['index']} at {p_name}")
        best = max(range(len(vals)), key=lambda i: (vals[i], -i))
        f.write(json.dumps({"index": it["index"], "set": it["set"], "letter_logp": vals,
                            "log_R": float(log_r), "m": math.exp(float(log_m)),
                            "correct": it["letters"][best] == it["answer"], "prompt_tokens": n_tok,
                            "vision_cache": cached, "seconds": round(time.time() - t1, 3)}) + "\n")
        f.flush()
        done[p_name].add(it["index"])
        count[0] += 1
        if stop_after and count[0] == stop_after:
            log.update(stopped_after=stop_after, end=g3.now())
            g3.append_jsonl(out / "runs.jsonl", log)
            print(f"forced stop after {stop_after} lines", flush=True)
            sys.exit(STOP_EXIT)

    def run_chunk(chunk: list[tuple[dict, dict]], k: int):
        t2 = time.time()
        for p_name, dname, alpha in points:
            todo = [(it, inp) for it, inp in chunk if it["index"] not in done[p_name]]
            if not todo:
                continue
            set_point(p_name, dname, alpha)
            with open(out / "points" / f"{p_name}.jsonl", "a") as f:
                if p_name == "a0-r2":  # no cache: process the items again and run the vision tower again
                    for it, inp in prepared([it for it, _ in todo]):
                        measure(f, p_name, it, inp, cached=False)
                else:
                    for it, inp in todo:
                        measure(f, p_name, it, inp, cached=True)
        log["chunks"].append({"chunk": k, "items": len(chunk), "vision_bytes": vc.bytes,
                              "points_seconds": round(time.time() - t2, 1)})
        vc.clear()
        torch.cuda.empty_cache()

    todo_any = [it for it in items if any(it["index"] not in done[p] for p, _, _ in points)]
    chunk, k = [], 0
    with open(out / "VISION.jsonl", "a") as vf:
        t_wait = time.time()
        for it, inp in prepared(todo_any):
            t3 = time.time()
            with torch.no_grad():
                n = vc.put(it["index"], inp["pixel_values"].cuda(), inp["image_grid_thw"].cuda())
            t4 = time.time()
            vf.write(json.dumps({"index": it["index"], "chunk": k, "wait_seconds": round(t3 - t_wait, 3),
                                 "vision_seconds": round(t4 - t3, 3), "prompt_tokens": int(inp["input_ids"].shape[1]),
                                 "image_tokens": int(inp["image_grid_thw"].prod(-1).sum()) // 4,
                                 "bytes": n}) + "\n")
            vf.flush()
            chunk.append((it, {key: v for key, v in inp.items() if key != "pixel_values"} | {"pixel_values": None}))
            if vc.bytes >= budget:
                run_chunk(chunk, k)
                chunk, k = [], k + 1
            t_wait = time.time()
        if chunk:
            run_chunk(chunk, k)
    pool.shutdown()
    lines = {p: done_items(out / "points" / f"{p}.jsonl") for p, _, _ in points}
    metas = {p: json.loads((out / "points" / f"{p}.meta.json").read_text()) for p, _, _ in points}
    res = {"checkpoint": checkpoint, "step_norm_s": s, "alphas": alphas, "directions": directions,
           "items": {"A": sum(it["set"] == "A" for it in items), "B": sum(it["set"] == "B" for it in items)},
           "direction_norms": {d: json.loads((out / "points" / f"{d}.direction.json").read_text())
                               for d in directions},
           **summarize(points, lines, metas, bf16)}
    g3.write_json(out / f"LINESEARCH_{checkpoint}.json", res)
    log.update(end=g3.now(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    g3.append_jsonl(out / "runs.jsonl", log)
    print(json.dumps({"checkpoint": checkpoint, "floor": res["floor"], "fp32_minus_bf16": res["fp32_minus_bf16_alpha0"]}))


if __name__ == "__main__":
    main()
