"""Arm P (packed forward) and arm U (one forward per row) on the A1 micro-batches (HANDOFF-VLM-VAD-PACK-001-R1 Part A
items 2 to 5, Definitions 2, 3, 7 to 12).

Inputs of a micro-batch: trainer_io.assemble_batch of its rows (the rollout response ids, the A1 agent-loop prompt),
the A1 teacher views (trainer_io.teacher_views), on the GPU. Both arms use these tensors (Definition 7: shared inputs):
  arm P: the whole micro-batch in one Objectives.losses(["O4"]) call (DataParallelPPOActor._forward_micro_batch packs
    the rows with use_remove_padding; the student, the x-minus teacher and the x-plus teacher forwards are packed).
    Batch loss term: O4_m x B_m / B_mini (dp_actor.py 1444, 1829), B_mini = rows of the rank mini-batch.
  arm U: row i of the same tensors (batch[[i]]) in its own Objectives.losses(["O4"]) call. Batch loss term:
    O4_i x (B_m / B_mini) x T_i / T_m, T = the loss-mask tokens (response_mask x self_distillation_mask). The O4 per-token
    loss uses only the tensors of its row and agg_loss token-mean divides by T_m (core_algos.py 1207-1209, 1305-1334,
    1634-1636, 1057-1060), so the sum over the rows equals the arm P aggregation with the arm U forwards.
  Each arm: old_log_probs = log_probs.detach() of its own student forward (on_policy, dp_actor.py 1485-1486), so the
  importance ratio is 1. The O4 loss is that of scripts/vad_grad/moments.py (A8 Objectives, rollout_is_weights None).
Gradient g_X(b) (Definition 8): the mean over the ranks (FSDP gradient averaging) of the sum over the micro-batches of
  the rank of the gradient of its batch loss term, over the 248 language-model tensors (a8 params.grad_params), bf16
  parameters, each micro-batch gradient added to an fp32 accumulator with the factor 1 / ranks.
Row values (Definition 12): L = sum of the student response log-probabilities; d_t = logp_P - logp_U (student);
  KL_t = KL(U || P) over the full vocabulary from the response logits of the student and of the x-plus teacher forwards
  (the teacher of A1, the "teacher_*" view); O4 of each arm for the row (arm P: the O4 code on the row slice of the
  packed outputs).
Progress: progress_<batch>.json after each micro-batch. The accumulators are saved (vectors/partial_<batch>_<arm>.pt)
  with the list of the micro-batches in them at most every COMMIT_SECONDS, at a forced stop and at the batch end. A
  restart loads them and skips those micro-batches; it computes the other micro-batches again.
At the batch end: vectors/<batch>_P.pt and _U.pt (flat fp32), the squared norm of each vector (fp64, on the GPU) and
  PACKROWS_<batch>.jsonl.
gram: GRAM_PACK.json, the fp64 inner products of all vector files (blocks of BLOCK elements), then GRAM_PACK.sha256, then
  the vector files are removed.

Usage: python pack.py run <repo root> <COMMAND_B200.txt> <model dir> <rollouts.jsonl> <split dir> <out dir> <batch>...
       python pack.py gram <out dir>
Env: STOP_AFTER_MICRO (0; the pilot stops once after this many micro-batches, exit 75), PILOT_MICRO (a json list of
  [batch, rank, mini, micro]; only these micro-batches), ONE_ROW_CHECK (0; 1 scores each row again from a one-row
  batch built alone and records L_1row and O4_1row), COMMIT_SECONDS (1800), BLOCK (67108864).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP_EXIT = 75
FIELDS_SHARED = ("input_ids", "responses", "teacher_input_ids", "teacher_minus_input_ids")


class Stop(Exception):
    """Forced stop after STOP_AFTER_MICRO micro-batches (pilot)."""


def _objectives():
    sys.path.insert(0, str(HERE.parent / "vad_rollout" / "a8"))
    import objectives as obj

    return obj


def make_capture(obj):
    class Capture(obj.Objectives):
        """Objectives.losses that keeps the outputs of its forwards: student, x-minus teacher, x-plus teacher.

        The student and the x-plus teacher forwards also return the response logits (full vocabulary) for KL_t.
        """

        outputs: list

        def _forward(self, inputs, grad, **kwargs):
            if len(self.outputs) in (0, 2):
                kwargs["return_response_logits"] = True
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
            return loss, student, plus, minus

    class Replay(obj.Objectives):
        """Objectives.losses on stored forward outputs (student, x-minus, x-plus): the O4 of one row of arm P."""

        queue: list

        def _forward(self, inputs, grad, **kwargs):
            return self.queue.pop(0)

        def o4(self, mi, student, minus, plus):
            self.queue = [student, minus, plus]
            return self.losses(mi, ["O4"])["O4"]

    return Capture, Replay


def row_of(out: dict, i: int) -> dict:
    return {k: v[i:i + 1].detach() for k, v in out.items() if hasattr(v, "dim")}


def loss_tokens(mi: dict):
    """Tokens of the O4 loss mask of each row (core_algos.py 1207-1209)."""
    mask = mi["response_mask"].float()
    sd = mi.get("self_distillation_mask")
    if sd is not None:
        mask = mask * sd.float().unsqueeze(1)
    return mask.sum(-1)


def kl_sum(logits_u, logits_p, mask) -> float:
    """Sum over the masked response tokens of KL(U || P) over the full vocabulary (fp32 log-softmax)."""
    import torch

    idx = mask[0].nonzero().squeeze(-1)
    lu = torch.log_softmax(logits_u[0, idx].float(), dim=-1)
    lp = torch.log_softmax(logits_p[0, idx].float(), dim=-1)
    return float((lu.exp() * (lu - lp)).sum())


def row_values(sP: dict, plusP: dict, i: int, sU: dict, plusU: dict, mask) -> dict:
    """L_P, L_U, |d_t| and the KL sums of row i (mask: the response mask of the row, shape (1, R))."""
    lp_p = sP["log_probs"][i:i + 1].detach().float()
    lp_u = sU["log_probs"].detach().float()
    m = mask.float()
    d = ((lp_p - lp_u) * m)[0][mask[0].bool()].abs()
    return {"L_P": float((lp_p * m).sum()), "L_U": float((lp_u * m).sum()),
            "d_abs": [float(f"{x:.6g}") for x in d.tolist()],
            "d_mean": float(d.mean()) if d.numel() else 0.0, "d_max": float(d.max()) if d.numel() else 0.0,
            "kl_student_sum": kl_sum(sU["response_logits"], sP["response_logits"][i:i + 1], mask),
            "kl_teacher_sum": kl_sum(plusU["response_logits"], plusP["response_logits"][i:i + 1], mask),
            "kl_tokens": int(mask.sum())}


def shared_digest(mi: dict) -> dict:
    """sha256 of the shared input tensors of a micro-batch (Part A item 2)."""
    out = {}
    for k in FIELDS_SHARED:
        if k in mi:
            out[k] = hashlib.sha256(mi[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return out


class Accum:
    """fp32 gradient sums of one batch for the two arms, with the micro-batches in them."""

    def __init__(self, params, device):
        import torch

        self.parts = {a: [torch.zeros(p.shape, dtype=torch.float32, device=device) for p in params] for a in "PU"}
        self.done: list = []

    def add(self, arm: str, grads, alpha: float):
        for a, g in zip(self.parts[arm], grads):
            a.add_(g.float(), alpha=alpha)

    def flat(self, arm: str):
        import torch

        return torch.cat([a.reshape(-1) for a in self.parts[arm]]).cpu()

    def sq_norm(self, arm: str) -> float:
        return float(sum(a.double().pow(2).sum() for a in self.parts[arm]))

    def save(self, path: Path, arm: str):
        import torch

        tmp = path.with_suffix(".tmp")
        torch.save({"done": self.done, "flat": self.flat(arm)}, tmp)
        os.replace(tmp, path)

    def load(self, path: Path, arm: str) -> list:
        import torch

        state = torch.load(path, map_location="cpu")
        off = 0
        for a in self.parts[arm]:
            n = a.numel()
            a.copy_(state["flat"][off:off + n].view(a.shape))
            off += n
        return state["done"]


def check_finite(rec: dict, where: str, out: Path):
    for k, v in rec.items():
        if isinstance(v, float) and not math.isfinite(v):
            (out / "STOP.txt").write_text(f"non-finite {k} at {where}\n")
            raise SystemExit(f"STOP (stop condition 2): non-finite {k} at {where}")


def micro_key(r: dict) -> tuple:
    return (r["rank"], r["mini"], r["micro"])


def run_batch(ctx, batch: str, split_rows: list[dict], rollouts: dict, out: Path, opts: dict, counter: dict) -> dict:
    import torch

    import trainer_values as tv

    arm_name = split_rows[0]["arm"]
    ranks = tv.VALUES[arm_name]["ranks"] // tv.VALUES[arm_name]["ulysses"]
    b_mini = tv.rank_mini_batch_size(arm_name)
    vec_dir = out / "vectors"
    vec_dir.mkdir(exist_ok=True)
    if (out / f"PACKROWS_{batch}.jsonl").exists():
        return {"batch": batch, "skipped": "PACKROWS exists"}
    micro = {}
    for r in split_rows:
        micro.setdefault(micro_key(r), []).append(r)
    keys = [k for k in micro if opts["pilot"] is None or [batch, *k] in opts["pilot"]]
    if not keys:
        return {"batch": batch, "skipped": "no selected micro-batch"}
    acc = Accum(ctx.params, ctx.device)
    prog_path = out / f"progress_{batch}.json"
    progress = json.loads(prog_path.read_text()) if prog_path.exists() else {"rows": [], "micro": [], "committed": []}
    partial = {a: vec_dir / f"partial_{batch}_{a}.pt" for a in "PU"}
    if all(p.exists() for p in partial.values()):
        done_p, done_u = (acc.load(partial[a], a) for a in "PU")
        if done_p != done_u:
            raise SystemExit(f"STOP: partial vectors of {batch} hold different micro-batches")
        acc.done = done_p
    committed = {tuple(k) for k in acc.done}
    progress["rows"] = [r for r in progress["rows"] if micro_key(r) in committed]
    progress["micro"] = [m for m in progress["micro"] if tuple(m["key"]) in committed]
    last_commit = time.time()
    commit_times = []

    def commit():
        t = time.time()
        for a in "PU":
            acc.save(partial[a], a)
        progress["committed"] = [list(k) for k in acc.done]
        write_json(prog_path, progress)
        commit_times.append(time.time() - t)
        progress["seconds_commit_max"] = max(commit_times)
        write_json(prog_path, progress)

    Capture, Replay = ctx.classes
    cap = Capture(ctx.actor, ctx.params)
    rep = Replay(ctx.actor, ctx.params)
    for key in keys:
        if key in committed:
            continue
        rows = sorted(micro[key], key=lambda r: r["position"])
        t0 = time.time()
        mi, rows_mi = ctx.inputs([rollouts[r["traj"]] for r in rows])
        digest = shared_digest(mi)
        t_rows = loss_tokens(mi)
        t_m = float(t_rows.sum())
        scale = len(rows) / b_mini
        t1 = time.time()
        # arm P
        loss_p, sP, plusP, minusP = cap.run(mi)
        grads = torch.autograd.grad(loss_p * scale, ctx.params)
        acc.add("P", grads, 1.0 / ranks)
        del grads
        sP, plusP, minusP = ({k: v.detach() for k, v in o.items() if hasattr(v, "dim")} for o in (sP, plusP, minusP))
        with torch.no_grad():
            o4_p = [float(rep.o4(rows_mi[i], row_of(sP, i), row_of(minusP, i), row_of(plusP, i))) for i in range(len(rows))]
        t2 = time.time()
        # arm U
        recs = []
        for i, r in enumerate(rows):
            loss_u, sU, plusU, _ = cap.run(rows_mi[i])
            w = scale * float(t_rows[i]) / t_m if t_m > 0 else 0.0
            grads = torch.autograd.grad(loss_u * w, ctx.params)
            acc.add("U", grads, 1.0 / ranks)
            del grads
            with torch.no_grad():
                vals = row_values(sP, plusP, i, sU, plusU, rows_mi[i]["response_mask"])
            rec = {**{k: r[k] for k in ("batch", "arm", "set", "rank", "mini", "micro", "position", "micro_rows",
                                        "traj", "prompt", "sibling", "label", "total_tokens", "response_tokens")},
                   "loss_tokens": float(t_rows[i]), "weight_P": scale, "weight_U": w,
                   "O4_P": o4_p[i], "O4_U": float(loss_u.detach()), **vals}
            if opts["one_row_check"]:
                rec.update(one_row(ctx, cap, rollouts[r["traj"]]))
            check_finite(rec, f"{batch} trajectory {r['traj']}", out)
            recs.append(rec)
            del sU, plusU, loss_u
        t3 = time.time()
        del sP, plusP, minusP, loss_p, mi, rows_mi
        acc.done.append(list(key))
        progress["rows"] += recs
        progress["micro"].append({"key": list(key), "rows": len(rows), "tokens": sum(r["total_tokens"] for r in rows),
                                  "inputs_sha256": digest, "seconds_inputs": round(t1 - t0, 3),
                                  "seconds_P": round(t2 - t1, 3), "seconds_U": round(t3 - t2, 3)})
        write_json(prog_path, progress)
        counter["micro"] += 1
        if opts["stop_after"] and counter["micro"] == opts["stop_after"]:
            commit()
            raise Stop()
        if time.time() - last_commit > opts["commit_seconds"]:
            commit()
            last_commit = time.time()
    # batch end
    t_end = time.time()
    norms = {a: acc.sq_norm(a) for a in "PU"}
    for a in "PU":
        tmp = vec_dir / f"{batch}_{a}.tmp"
        torch.save(acc.flat(a), tmp)
        os.replace(tmp, vec_dir / f"{batch}_{a}.pt")
    progress["sq_norm"] = norms
    progress["committed"] = [list(k) for k in acc.done]
    write_json(prog_path, progress)
    with open(out / f"PACKROWS_{batch}.jsonl", "w") as f:
        for rec in progress["rows"]:
            f.write(json.dumps(rec) + "\n")
    for p in partial.values():
        p.unlink(missing_ok=True)
    return {"batch": batch, "micro_batches": len(acc.done), "rows": len(progress["rows"]), "sq_norm": norms,
            "seconds_end": round(time.time() - t_end, 1), "seconds_commit_max": progress.get("seconds_commit_max", 0.0)}


def one_row(ctx, cap, ro: dict) -> dict:
    """L and O4 of one trajectory from a one-row batch built alone (the GRAD-001 input path), no gradient."""
    import torch

    mi = ctx.inputs([ro])[0]
    with torch.no_grad():
        loss, s, _, _ = cap.run(mi)
    return {"L_1row": float((s["log_probs"].float() * mi["response_mask"]).sum()), "O4_1row": float(loss)}


def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


class Context:
    def __init__(self, repo: str, command_txt: str, model_dir: str):
        import trainer_io
        import params as gparams

        obj = _objectives()
        self.classes = make_capture(obj)
        self.cfg = trainer_io.compose_a1_config(repo, command_txt)
        self.tokenizer, self.processor = trainer_io.tokenizer_processor(self.cfg)
        self.dataset = trainer_io.load_dataset(self.cfg, self.tokenizer, self.processor)
        trainer_io.init_single_rank()
        student = trainer_io.load_model(model_dir, self.cfg, train=True)
        teacher = trainer_io.load_model(self.cfg.actor_rollout_ref.model.path, self.cfg, train=False)
        self.actor = trainer_io.make_actor(self.cfg, student, teacher)
        self.params = [p for _, p in gparams.grad_params(student)]
        self.device = "cuda"
        self._prompts = {}

    def prompt(self, index: int) -> dict:
        import trainer_io

        if index not in self._prompts:
            self._prompts[index] = trainer_io.prompt_inputs(self.dataset, index, self.processor, self.cfg)
        return self._prompts[index]

    def inputs(self, ros: list[dict]) -> tuple[dict, list[dict]]:
        """Model inputs of a micro-batch (the rollout rows in position order) and of each of its rows (batch[[i]])."""
        import trainer_io

        items = []
        for ro in ros:
            item = self.prompt(ro["dataset_index"])
            if hashlib.sha256(json.dumps(item["prompt_ids"]).encode()).hexdigest() != ro["prompt_sha256"]:
                raise RuntimeError(f"prompt of trajectory {ro['traj']} differs from the rollout prompt")
            items.append({**item, "response_ids": ro["response_ids"], "uid": str(ro["prompt_pos"])})
        b = trainer_io.assemble_batch(items, self.tokenizer, self.processor, self.cfg)
        b = b.union(trainer_io.teacher_views(b, self.cfg, self.tokenizer, self.processor))
        b = trainer_io.to_device(b, self.device)
        pad = self.tokenizer.pad_token_id
        mi = {**b.batch, **b.non_tensor_batch, "pad_token_id": pad}
        rows_mi = [{**b[[i]].batch, **b[[i]].non_tensor_batch, "pad_token_id": pad} for i in range(len(ros))]
        return mi, rows_mi


def run(argv: list[str]):
    import torch

    repo, command_txt, model_dir, rollouts_path, split_dir, out, *batches = argv
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(HERE.parent / "vad_rollout" / "a8"))
    sys.path.insert(0, str(HERE.parent / "vad_grad"))
    sys.path.insert(0, str(HERE))
    import provenance

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"stop_after": int(os.environ.get("STOP_AFTER_MICRO", "0")),
            "pilot": json.loads(Path(os.environ["PILOT_MICRO"]).read_text()) if os.environ.get("PILOT_MICRO") else None,
            "one_row_check": os.environ.get("ONE_ROW_CHECK", "0") == "1",
            "commit_seconds": float(os.environ.get("COMMIT_SECONDS", "1800"))}
    todo = [b for b in batches if not (out / f"PACKROWS_{b}.jsonl").exists()]
    log = {"model_dir": model_dir, "batches": batches, "start": now(), "verl": provenance.verl_origin(repo), **opts}
    if not todo:
        print(f"{batches}: all PACKROWS files exist")
        return
    t0 = time.time()
    ctx = Context(repo, command_txt, model_dir)
    log["load_seconds"] = round(time.time() - t0, 1)
    log["grad_tensors"] = len(ctx.params)
    log["grad_elements"] = sum(p.numel() for p in ctx.params)
    rollouts = {r["traj"]: r for r in map(json.loads, open(rollouts_path))}
    torch.cuda.reset_peak_memory_stats()
    counter = {"micro": 0}
    log["results"] = []
    try:
        for b in todo:
            split_rows = [json.loads(line) for line in open(Path(split_dir) / f"ROWS_{b}.jsonl")]
            log["results"].append(run_batch(ctx, b, split_rows, rollouts, out, opts, counter))
            ctx._prompts = {}
    except Stop:
        log.update(stopped_after_micro=opts["stop_after"], end=now(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        append_jsonl(out / "runs.jsonl", log)
        print(f"forced stop after {opts['stop_after']} micro-batches")
        sys.exit(STOP_EXIT)
    log.update(end=now(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    append_jsonl(out / "runs.jsonl", log)
    print(json.dumps(log["results"]))


def gram(out: Path):
    """GRAM_PACK.json: fp64 inner products of the vector files, block by block on the GPU; then GRAM_PACK.sha256 and the
    removal of the vector files."""
    import torch

    files = sorted((out / "vectors").glob("*_[PU].pt"))
    names = [f.stem for f in files]
    vecs = [torch.load(f, mmap=True) for f in files]
    n = vecs[0].numel()
    if any(v.numel() != n for v in vecs):
        raise SystemExit("STOP: vector files differ in length")
    block = int(os.environ.get("BLOCK", str(1 << 26)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    g = torch.zeros(len(vecs), len(vecs), dtype=torch.float64, device=dev)
    for s in range(0, n, block):
        x = torch.stack([v[s:s + block].to(dev, torch.float64) for v in vecs])
        g += x @ x.T
        del x
    sep = {}
    for p in sorted(out.glob("progress_*.json")):
        state = json.loads(p.read_text())
        for a, v in state.get("sq_norm", {}).items():
            sep[f"{p.stem[len('progress_'):]}_{a}"] = v
    res = {"names": names, "elements": n, "block": block, "gram": g.cpu().tolist(), "sq_norm_separate": sep,
           "seconds": round(time.time() - t0, 1), "vector_sha256": {f.stem: sha256_file(f) for f in files}}
    path = out / "GRAM_PACK.json"
    path.write_text(json.dumps(res, indent=1))
    (out / "GRAM_PACK.sha256").write_text(f"{sha256_file(path)}  GRAM_PACK.json\n")
    for f in files:
        f.unlink()
    print(f"gram: {len(names)} vectors of {n} elements, {res['seconds']} s, vector files removed")


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
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    elif sys.argv[1] == "gram":
        gram(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command {sys.argv[1]}")


if __name__ == "__main__":
    main()
