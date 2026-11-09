"""Gradient vectors of Definition 9 (HANDOFF-VLM-VAD-GRAD-003-R2 Part A items 3 to 5).

Batch loss (Definition 7, the A1 aggregation of batches.py): for each micro-batch m of each rank, the official O4 loss
O4_m of the packed micro-batch (one Objectives.losses(["O4"]) call: the student forward with gradient packs the rows with
use_remove_padding, then the x-minus and x-plus teacher forwards), times B_m / mini_cfg (dp_actor.py 1444, 1829). The
step gradient is the mean over the 6 ranks (FSDP) of the sum over the micro-batches of the rank. The gradient covers the
248 tensors of params.grad_params in the order of G3_PARAMS.txt, bf16 parameters, each micro-batch gradient added to an
fp32 accumulator with the factor 1 / 6.
Parts (Definition 9.1, 9.2): part X = the O4 loss of the same packed forward outputs with the response mask of the rows
outside X set to 0, times T_X / T_m (token mean, core_algos.py 1057-1060, 1207-1209) and B_m / mini_cfg, its own
backward on the retained graph. Row losses l_i (GRADROWS): the O4 loss of row i on its slice of the packed outputs
(Replay). Each micro-batch checks |sum_i (T_i/T_m) l_i - O4_m| <= 1e-4 |O4_m| (else STOP).
Rounding level (DIRECTIVE-VLM-VAD-GRAD-003-005): after the total backward, the backward of 3 x the total target on the
retained graph, divided by 3, goes to the accumulator total_x3 (partial files, no vector file). Each key writes
rounding_rel = ||total_x3 - total|| / ||total|| and part_sum_rel = ||sum of the parts - total|| / ||total|| (C+I and
L1+L2+L3, null for a key without parts) to its progress file and to runs.jsonl.
Vectors of a batch key (mode O = VAD_PACKFIX unset, Fx = VAD_PACKFIX=1, set by run_grad3.sh for the process):
  O, B8-f<f>: total, C, I, L1, L2, L3.  O, B1-f<f>-j<j>: total, C, I.  Other keys (R8-*, Fx, <batch>@rep): total.
  <batch>@rep: the rows of <batch> again (the numerical repeat of Definition 9.3).
Files: vectors/<mode>__<key>__<part>.pt (flat fp32), GRADROWS_<mode>__<key>.jsonl (rows with l_i), progress_<mode>__<key>.json
  (after each micro-batch). Partial accumulators (vectors/partial_*.pt) with their micro-batch list at most every
  COMMIT_SECONDS, at a forced stop and at the batch end. A restart loads them and skips those micro-batches; a key with
  its GRADROWS file is skipped.
Gram (Part A item 4): GRAM_<checkpoint>.json holds, for all vector files and the 4 displacement vectors, the fp64 inner
  products raw <x, y>, M1 <x, D y> and M2 <x, D^2 y>, D = 1 / (sqrt(v) / sqrt(1 - 0.999^t) + 1e-8) (GRAD-001 moments.py
  91-99), and the separate squared norms. Then GRAM_<checkpoint>.sha256, then the vector files are removed (VECTORS_REMOVED.json).

Usage: python grads.py run <repo> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <rows dir> <G3_PARAMS.txt> <out dir> <mode> <key>...
       python grads.py displacement <base model dir> <G3_PARAMS.txt> <out dir> <A1 80> <A1 130> <A2 80> <A2 130>  (actor dirs)
       python grads.py gram <out dir> <displacement dir> <V file> <G3_PARAMS.txt> <checkpoint>
Env: STOP_AFTER_MICRO (0; stop once after this many micro-batches, exit 75), COMMIT_SECONDS (1800), BLOCK (8388608), READERS (16; gram read threads).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STOP_EXIT = 75
RANKS = 6
CHECK_REL = 1e-4
DISPLACEMENTS = ("D__A1_0_80", "D__A1_80_130", "D__A2_0_80", "D__A2_80_130")
X3 = "total_x3"  # accumulator of the scale-3 gradients; no vector file


class Stop(Exception):
    """Forced stop after STOP_AFTER_MICRO micro-batches (pilot)."""


def parts_of(mode: str, key: str) -> list[str]:
    if mode == "O" and "@" not in key and key.startswith("B8-"):
        return ["total", "C", "I", "L1", "L2", "L3"]
    if mode == "O" and "@" not in key and key.startswith("B1-"):
        return ["total", "C", "I"]
    return ["total"]


def members(part: str, rows: list[dict]) -> list[int]:
    """Row places of a part: C = label 1, I = label 0, L1 to L3 = length bin."""
    if part in ("C", "I"):
        return [i for i, r in enumerate(rows) if r["label"] == (1 if part == "C" else 0)]
    return [i for i, r in enumerate(rows) if r["len_bin"] == part]


def make_classes(obj):
    class Capture(obj.Objectives):
        """Objectives.losses that keeps its forward outputs: student, x-minus teacher, x-plus teacher."""

        outputs: list

        def _forward(self, inputs, grad, **kwargs):
            out = super()._forward(inputs, grad, **kwargs)
            self.outputs.append(out)
            return out

        def run(self, mi):
            self.outputs = []
            loss = self.losses(mi, ["O4"])["O4"]
            if len(self.outputs) != 3:
                raise RuntimeError(f"O4 made {len(self.outputs)} forwards, not 3")
            student, minus, plus = self.outputs
            self.outputs = []
            return loss, student, minus, plus

    class Replay(obj.Objectives):
        """Objectives.losses on given forward outputs (student, x-minus, x-plus): the O4 loss of one row."""

        queue: list

        def _forward(self, inputs, grad, **kwargs):
            return self.queue.pop(0)

        def o4(self, mi, student, minus, plus):
            self.queue = [student, minus, plus]
            return self.losses(mi, ["O4"])["O4"]

    return Capture, Replay


def row_slice(out: dict, i: int) -> dict:
    """Row i of the forward outputs; keeps the graph of the student outputs."""
    return {k: v[i:i + 1] for k, v in out.items() if hasattr(v, "dim")}


def loss_tokens(mi: dict):
    """Tokens of the O4 loss mask of each row (core_algos.py 1207-1209)."""
    mask = mi["response_mask"].float()
    sd = mi.get("self_distillation_mask")
    if sd is not None:
        mask = mask * sd.float().unsqueeze(1)
    return mask.sum(-1)


class Accum:
    """fp32 gradient sums of the parts of one batch key, with the micro-batches in them."""

    def __init__(self, params, device, names: list[str]):
        import torch

        self.parts = {n: [torch.zeros(p.shape, dtype=torch.float32, device=device) for p in params] for n in names}
        self.done: list = []

    def add(self, name: str, grads, alpha: float):
        for a, g in zip(self.parts[name], grads):
            a.add_(g.float(), alpha=alpha)

    def flat(self, name: str):
        import torch

        return torch.cat([a.reshape(-1) for a in self.parts[name]]).cpu()

    def sq_norm(self, name: str) -> float:
        return float(sum(a.double().pow(2).sum() for a in self.parts[name]))

    def save(self, path: Path, name: str):
        import torch

        tmp = path.with_suffix(".tmp")
        torch.save({"done": self.done, "flat": self.flat(name)}, tmp)
        os.replace(tmp, path)

    def load(self, path: Path, name: str) -> list:
        import torch

        state = torch.load(path, map_location="cpu", mmap=True)
        off = 0
        for a in self.parts[name]:
            n = a.numel()
            a.copy_(state["flat"][off:off + n].view(a.shape))
            off += n
        return state["done"]


def micro_key(r: dict) -> tuple:
    return (r["rank"], r["mini"], r["micro"])


def stop_file(out: Path, text: str):
    (out / "STOP.txt").write_text(text + "\n")
    raise SystemExit(f"STOP: {text}")


def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def run_batch(ctx, mode: str, key: str, rows: list[dict], rollouts: dict, out: Path, opts: dict, counter: dict) -> dict:
    import torch

    tag = f"{mode}__{key}"
    vec_dir = out / "vectors"
    vec_dir.mkdir(exist_ok=True)
    if (out / f"GRADROWS_{tag}.jsonl").exists():
        return {"key": tag, "skipped": "GRADROWS exists"}
    names = parts_of(mode, key)
    accs = names + [X3]
    mini = {8: 128, 1: 16}[rows[0]["n"]]
    micro = {}
    for r in rows:
        micro.setdefault(micro_key(r), []).append(r)
    acc = Accum(ctx.params, ctx.device, accs)
    prog_path = out / f"progress_{tag}.json"
    progress = json.loads(prog_path.read_text()) if prog_path.exists() else {"rows": [], "micro": [], "committed": []}
    partial = {n: vec_dir / f"partial_{tag}__{n}.pt" for n in accs}
    if all(p.exists() for p in partial.values()):
        dones = [acc.load(partial[n], n) for n in accs]
        if any(d != dones[0] for d in dones):
            stop_file(out, f"partial vectors of {tag} hold different micro-batches")
        acc.done = dones[0]
    committed = {tuple(k) for k in acc.done}
    progress["rows"] = [r for r in progress["rows"] if micro_key(r) in committed]
    progress["micro"] = [m for m in progress["micro"] if tuple(m["key"]) in committed]
    progress.setdefault("resumes", []).append({"committed": len(committed), "at": now()})
    last_commit = time.time()

    def commit():
        t = time.time()
        for n in accs:
            acc.save(partial[n], n)
        progress["committed"] = [list(k) for k in acc.done]
        progress["seconds_commit_max"] = max(progress.get("seconds_commit_max", 0.0), round(time.time() - t, 1))
        write_json(prog_path, progress)

    Capture, Replay = ctx.classes
    cap, rep = Capture(ctx.actor, ctx.params), Replay(ctx.actor, ctx.params)
    for k in micro:
        if k in committed:
            continue
        mrows = sorted(micro[k], key=lambda r: r["position"])
        t0 = time.time()
        mi, rows_mi = ctx.inputs([rollouts[r["traj"]] for r in mrows])
        t_rows = loss_tokens(mi)
        t_m = float(t_rows.sum())
        if [float(x) for x in t_rows] != [float(r["loss_tokens"]) for r in mrows]:
            stop_file(out, f"{tag} micro-batch {k}: loss tokens differ from the row file")
        scale = len(mrows) / mini
        t1 = time.time()
        loss, student, minus, plus = cap.run(mi)
        need_parts = len(names) > 1
        rec_l = [None] * len(mrows)
        check = None
        if need_parts:
            minus_d = {k2: v.detach() for k2, v in minus.items() if hasattr(v, "dim")}
            plus_d = {k2: v.detach() for k2, v in plus.items() if hasattr(v, "dim")}
            l_rows = [rep.o4(rows_mi[i], row_slice(student, i), row_slice(minus_d, i), row_slice(plus_d, i))
                      for i in range(len(mrows))]
            w = [float(t_rows[i]) / t_m if t_m > 0 else 0.0 for i in range(len(mrows))]
            recon = sum(wi * float(li.detach()) for wi, li in zip(w, l_rows))
            check = abs(recon - float(loss.detach()))
            if check > CHECK_REL * max(abs(float(loss.detach())), 1e-12):
                stop_file(out, f"{tag} micro-batch {k}: row losses give {recon}, the packed loss is {float(loss)}")
            rec_l = [float(li.detach()) for li in l_rows]
        if not math.isfinite(float(loss.detach())):
            stop_file(out, f"non-finite loss in {tag} micro-batch {k} (stop condition 3)")
        t2 = time.time()
        grads = torch.autograd.grad(loss * scale, ctx.params, retain_graph=True)
        acc.add("total", grads, 1.0 / RANKS)
        del grads
        t3 = time.time()
        # Rounding level (DIRECTIVE-VLM-VAD-GRAD-003-005): the backward of 3 x the target on the retained graph, / 3.
        grads = torch.autograd.grad(loss * (3 * scale), ctx.params, retain_graph=need_parts)
        acc.add(X3, grads, 1.0 / (3 * RANKS))
        del grads
        t4 = time.time()
        parts = [n for n in names[1:] if members(n, mrows)]
        for j, n in enumerate(parts):
            # Definition 9.1: the O4 loss of the same packed forward outputs with the response mask of the rows
            # outside part n set to 0, times T_n / T_m (token mean) and B_m / mini.
            idx = members(n, mrows)
            mask = mi["response_mask"].clone()
            for i in range(len(mrows)):
                if i not in idx:
                    mask[i] = 0
            mmi = {**mi, "response_mask": mask}
            target = rep.o4(mmi, student, minus_d, plus_d) * (float(loss_tokens(mmi).sum()) / t_m * scale)
            grads = torch.autograd.grad(target, ctx.params, retain_graph=j < len(parts) - 1)
            acc.add(n, grads, 1.0 / RANKS)
            del grads, target, mmi
        t5 = time.time()
        recs = [{**r, "l_row": rec_l[i], "O4_micro": float(loss.detach()), "scale": scale} for i, r in enumerate(mrows)]
        del loss, student, minus, plus, mi, rows_mi
        if need_parts:
            del l_rows
        acc.done.append(list(k))
        progress["rows"] += recs
        progress["micro"].append({"key": list(k), "rows": len(mrows), "tokens": sum(r["total_tokens"] for r in mrows),
                                  "check_abs": check, "seconds_inputs": round(t1 - t0, 3),
                                  "seconds_forward": round(t2 - t1, 3),
                                  "seconds_backward": round(t3 - t2 + t5 - t4, 3), "seconds_rounding": round(t4 - t3, 3)})
        write_json(prog_path, progress)
        counter["micro"] += 1
        if opts["stop_after"] and counter["micro"] == opts["stop_after"]:
            commit()
            raise Stop()
        if time.time() - last_commit > opts["commit_seconds"]:
            commit()
            last_commit = time.time()
    t_end = time.time()
    norms = {n: acc.sq_norm(n) for n in names}
    if not all(math.isfinite(v) for v in norms.values()):
        stop_file(out, f"non-finite vector norm in {tag}: {norms} (stop condition 3)")
    for n in names:
        tmp = vec_dir / f"{tag}__{n}.tmp"
        torch.save(acc.flat(n), tmp)
        os.replace(tmp, vec_dir / f"{tag}__{n}.pt")
    progress["sq_norm"] = norms
    progress["committed"] = [list(k) for k in acc.done]
    progress["part_sum_rel"] = part_sum_check(acc, names) if len(names) > 1 else None
    progress["rounding_rel"] = rel_to_total(acc, (X3,))
    write_json(prog_path, progress)
    with open(out / f"GRADROWS_{tag}.jsonl", "w") as f:
        for rec in progress["rows"]:
            f.write(json.dumps(rec) + "\n")
    for p in partial.values():
        p.unlink(missing_ok=True)
    return {"key": tag, "micro_batches": len(acc.done), "rows": len(progress["rows"]), "sq_norm": norms,
            "part_sum_rel": progress["part_sum_rel"], "rounding_rel": progress["rounding_rel"],
            "seconds_end": round(time.time() - t_end, 1)}


def rel_to_total(acc: Accum, group) -> float:
    """||sum of the group vectors - total|| / ||total||, fp64 per tensor."""
    total2 = acc.sq_norm("total")
    d2 = 0.0
    for t in range(len(acc.parts["total"])):
        s = sum(acc.parts[n][t].double() for n in group)
        d2 += float((s - acc.parts["total"][t].double()).pow(2).sum())
    return math.sqrt(d2 / total2) if total2 > 0 else float("nan")


def part_sum_check(acc: Accum, names: list[str]) -> dict:
    """part_sum_rel for each partition of the rows (C+I, L1+L2+L3)."""
    return {"+".join(g): rel_to_total(acc, g) for g in (("C", "I"), ("L1", "L2", "L3")) if all(n in names for n in g)}


class Context:
    def __init__(self, repo: str, command_txt: str, model_dir: str, g3_params: str):
        import ckpt
        import objectives as obj
        import params as gparams
        import trainer_io

        self.classes = make_classes(obj)
        self.cfg = trainer_io.compose_a1_config(repo, command_txt)
        self.tokenizer, self.processor = trainer_io.tokenizer_processor(self.cfg)
        self.dataset = trainer_io.load_dataset(self.cfg, self.tokenizer, self.processor)
        trainer_io.init_single_rank()
        student = trainer_io.load_model(model_dir, self.cfg, train=True)
        teacher = trainer_io.load_model(self.cfg.actor_rollout_ref.model.path, self.cfg, train=False)
        self.actor = trainer_io.make_actor(self.cfg, student, teacher)
        named = gparams.grad_params(student)
        if [n for n, _ in named] != ckpt.grad_set(g3_params):
            raise SystemExit("STOP: the gradient tensors differ from G3_PARAMS.txt (names or order)")
        self.params = [p for _, p in named]
        self.device = "cuda"
        self._prompts = {}

    def prompt(self, index: int) -> dict:
        import trainer_io

        if index not in self._prompts:
            self._prompts[index] = trainer_io.prompt_inputs(self.dataset, index, self.processor, self.cfg)
        return self._prompts[index]

    def batch(self, ros: list[dict]):
        """DataProto of the rollout rows (the A1 agent-loop postprocess and the A1 teacher views), on the GPU."""
        import trainer_io

        items = []
        for ro in ros:
            item = self.prompt(ro["dataset_index"])
            if hashlib.sha256(json.dumps(item["prompt_ids"]).encode()).hexdigest() != ro["prompt_sha256"]:
                raise RuntimeError(f"prompt of trajectory {ro['traj']} differs from the rollout prompt")
            items.append({**item, "response_ids": ro["response_ids"], "uid": str(ro["prompt_pos"])})
        b = trainer_io.assemble_batch(items, self.tokenizer, self.processor, self.cfg)
        b = b.union(trainer_io.teacher_views(b, self.cfg, self.tokenizer, self.processor))
        return trainer_io.to_device(b, self.device)

    def inputs(self, ros: list[dict]) -> tuple[dict, list[dict]]:
        """Model inputs of a micro-batch (rows in position order) and of each of its rows (batch[[i]])."""
        b = self.batch(ros)
        pad = self.tokenizer.pad_token_id
        mi = {**b.batch, **b.non_tensor_batch, "pad_token_id": pad}
        rows_mi = [{**b[[i]].batch, **b[[i]].non_tensor_batch, "pad_token_id": pad} for i in range(len(ros))]
        return mi, rows_mi


def setup_paths(repo: str):
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(HERE))


def run(argv: list[str]):
    import torch

    repo, command_txt, model_dir, rollouts_path, rows_dir, g3_params, out, mode, *keys = argv
    setup_paths(repo)
    import provenance

    if mode not in ("O", "Fx") or (mode == "Fx") != (os.environ.get("VAD_PACKFIX") == "1"):
        raise SystemExit(f"STOP: mode {mode} with VAD_PACKFIX={os.environ.get('VAD_PACKFIX')}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"stop_after": int(os.environ.get("STOP_AFTER_MICRO", "0")),
            "commit_seconds": float(os.environ.get("COMMIT_SECONDS", "1800"))}
    todo = [k for k in keys if not (out / f"GRADROWS_{mode}__{k}.jsonl").exists()]
    log = {"model_dir": model_dir, "mode": mode, "keys": keys, "todo": todo, "start": now(),
           "verl": provenance.verl_origin(repo), **opts}
    if not todo:
        print(f"{mode} {keys}: all GRADROWS files exist")
        return
    t0 = time.time()
    ctx = Context(repo, command_txt, model_dir, g3_params)
    log.update(load_seconds=round(time.time() - t0, 1), grad_tensors=len(ctx.params),
               grad_elements=sum(p.numel() for p in ctx.params))
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    torch.cuda.reset_peak_memory_stats()
    counter = {"micro": 0}
    log["results"] = []
    try:
        for k in todo:
            rows = [json.loads(line) for line in open(Path(rows_dir) / f"ROWS_{k.split('@')[0]}.jsonl")]
            log["results"].append(run_batch(ctx, mode, k, rows, rollouts, out, opts, counter))
            print(json.dumps(log["results"][-1]), flush=True)
            ctx._prompts = {}
    except Stop:
        log.update(stopped_after_micro=opts["stop_after"], end=now(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        append_jsonl(out / "runs.jsonl", log)
        print(f"forced stop after {opts['stop_after']} micro-batches")
        sys.exit(STOP_EXIT)
    log.update(end=now(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    append_jsonl(out / "runs.jsonl", log)


def displacement(argv: list[str]):
    """The 4 displacement vectors of Definition 8.2 over the 248 tensors, flat fp32 in G3_PARAMS.txt order:
    theta_after - theta_before, theta of A1 and A2 from the fp32 FSDP model shards (ckpt.model_tensor), theta_0 = Base."""
    import torch

    base_dir, g3_params, out, a1_80, a1_130, a2_80, a2_130 = argv
    sys.path.insert(0, str(HERE))
    import ckpt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ckpt.grad_set(g3_params)
    _, base = ckpt.base_tensors(base_dir)
    shards = {k: ckpt.open_shards(d, "model") for k, d in
              (("a1_80", a1_80), ("a1_130", a1_130), ("a2_80", a2_80), ("a2_130", a2_130))}
    pairs = {"D__A1_0_80": (None, "a1_80"), "D__A1_80_130": ("a1_80", "a1_130"),
             "D__A2_0_80": (None, "a2_80"), "D__A2_80_130": ("a2_80", "a2_130")}
    info = {}
    for name, (before, after) in pairs.items():
        if (out / f"{name}.pt").exists():
            continue
        t0 = time.time()
        parts = []
        for n in names:
            th_a = ckpt.model_tensor(shards[after], n)
            th_b = base(n) if before is None else ckpt.model_tensor(shards[before], n)
            parts.append((th_a - th_b).reshape(-1))
        flat = torch.cat(parts)
        del parts
        tmp = out / f"{name}.tmp"
        torch.save(flat, tmp)
        os.replace(tmp, out / f"{name}.pt")
        info[name] = {"elements": flat.numel(), "sq_norm": float(flat.double().pow(2).sum()),
                      "seconds": round(time.time() - t0, 1)}
        del flat
    old = json.loads((out / "DISPLACEMENT_VECTORS.json").read_text()) if (out / "DISPLACEMENT_VECTORS.json").exists() else {}
    write_json(out / "DISPLACEMENT_VECTORS.json", {**old, **info})
    print(json.dumps(info))


def metric_d(v_file: str, names: list[str]):
    """Flat fp32 D (GRAD-001 moments.py 91-99) in the order of names."""
    import torch
    from safetensors import safe_open

    with safe_open(v_file, framework="pt") as f:
        step = int(f.metadata()["step"])
        corr = math.sqrt(1.0 - 0.999 ** step)
        d = torch.cat([(1.0 / (f.get_tensor(n).float().sqrt() / corr + 1e-8)).reshape(-1) for n in names])
    return d, {"adam_file": v_file, "step": step, "beta2": 0.999, "eps": 1e-8}


def gram(argv: list[str]):
    """GRAM_<checkpoint>.json: raw, M1 and M2 fp64 Gram matrices of the vector files and the displacements."""
    import torch

    out, disp_dir, v_file, g3_params, checkpoint = argv
    out = Path(out)
    sys.path.insert(0, str(HERE))
    import ckpt

    files = sorted((out / "vectors").glob("*__*__*.pt")) + [Path(disp_dir) / f"{n}.pt" for n in DISPLACEMENTS]
    names = [f.stem for f in files]
    vecs = [torch.load(f, mmap=True) for f in files]
    n = vecs[0].numel()
    if any(v.numel() != n or v.dtype != torch.float32 for v in vecs):
        raise SystemExit("STOP: vector files differ in length or are not fp32")
    d, d_info = metric_d(v_file, ckpt.grad_set(g3_params))
    if d.numel() != n:
        raise SystemExit(f"STOP: D has {d.numel()} elements, the vectors {n}")
    block = int(os.environ.get("BLOCK", str(1 << 23)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    k = len(vecs)
    # Large pread calls on READERS threads, not mmap slices: mmap page faults read Lustre at 0.108 GB/s (IO_CHECK of
    # TASK-VLM-VAD-GRAD-003-001), 64 MB sequential reads at 5.4 GB/s.
    fds = [os.open(f, os.O_RDONLY) for f in files]
    offs = [data_offset(f, v) for f, v in zip(files, vecs)]
    for j, v in enumerate(vecs):
        for s in (0, max(0, n - 1024)):
            if not torch.equal(read_into(fds[j], offs[j], s, np.empty(min(1024, n - s), np.float32)), v[s:s + 1024]):
                raise SystemExit(f"STOP: the pread of {files[j].name} differs from torch.load")
    del vecs
    buf = np.empty((k, min(block, n)), np.float32)
    g = {m: torch.zeros(k, k, dtype=torch.float64, device=dev) for m in ("raw", "M1", "M2")}
    with ThreadPoolExecutor(int(os.environ.get("READERS", "16"))) as ex:
        for s in range(0, n, block):
            m_s = min(block, n - s)
            list(ex.map(lambda j: read_into(fds[j], offs[j], s, buf[j, :m_s]), range(k)))
            x = torch.from_numpy(buf[:, :m_s]).to(dev, torch.float64)
            db = d[s:s + block].to(dev, torch.float64)
            xd = x * db
            g["raw"] += x @ x.T
            g["M1"] += xd @ x.T
            g["M2"] += xd @ xd.T
            del x, xd, db
    for fd in fds:
        os.close(fd)
    sep = {}
    for p in sorted(out.glob("progress_*.json")):
        state = json.loads(p.read_text())
        for part, v in state.get("sq_norm", {}).items():
            sep[f"{p.stem[len('progress_'):]}__{part}"] = v
    res = {"checkpoint": checkpoint, "names": names, "elements": n, "block": block, "metric": d_info,
           **{m: g[m].cpu().tolist() for m in g}, "sq_norm_separate": sep, "seconds": round(time.time() - t0, 1)}
    path = out / f"GRAM_{checkpoint}.json"
    path.write_text(json.dumps(res))
    (out / f"GRAM_{checkpoint}.sha256").write_text(f"{sha256_file(path)}  {path.name}\n")
    removed = []
    for f in files[:len(files) - len(DISPLACEMENTS)]:
        f.unlink()
        removed.append(f.name)
    write_json(out / "VECTORS_REMOVED.json", {"at": now(), "gram": path.name,
                                             "gram_sha256": sha256_file(path), "removed": removed})
    print(f"gram {checkpoint}: {k} vectors of {n} elements, {res['seconds']} s, {len(removed)} vector files removed")


def data_offset(path: Path, t) -> int:
    """Byte offset of element 0 of t = torch.load(path, mmap=True) in the torch.save zip file (one stored record)."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        recs = [i for i in z.infolist() if "/data/" in i.filename]
    if len(recs) != 1 or recs[0].compress_type != zipfile.ZIP_STORED:
        raise SystemExit(f"STOP: {path} does not hold one stored data record")
    with open(path, "rb") as f:
        f.seek(recs[0].header_offset)
        head = f.read(30)
    if head[:4] != b"PK\x03\x04":
        raise SystemExit(f"STOP: {path}: no local file header at {recs[0].header_offset}")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    return recs[0].header_offset + 30 + name_len + extra_len + t.storage_offset() * t.element_size()


def read_into(fd: int, off: int, s: int, out):
    """Read fp32 elements s .. s + len(out) of a vector file into the numpy array out; return it as a tensor."""
    import torch

    mv = memoryview(out).cast("B")
    got = 0
    while got < len(mv):
        r = os.preadv(fd, [mv[got:]], off + 4 * s + got)
        if r == 0:
            raise SystemExit("STOP: a vector file ends early")
        got += r
    return torch.from_numpy(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def append_jsonl(path: Path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2:])
    elif cmd == "displacement":
        displacement(sys.argv[2:])
    elif cmd == "gram":
        gram(sys.argv[2:])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
