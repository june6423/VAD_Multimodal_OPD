"""Held-out sets and held-out gradients of HANDOFF-VLM-VAD-VIEW-001-R2 Definitions 7 and 9.2.

select: the items of MME_RealWorld.json (MME-RealWorld EN, the file of the official evaluation) in the order of
  random.Random(20261111).shuffle of the item list; an item is kept when its query has option lines "(X) " with
  distinct letters A, B, ... in order and its answer is one of them, else it is replaced by the next item (recorded).
  Set A = the first 500 kept items in 4 parts of 125, set B = the next 500. HELDOUT.json lists the 1000 items (later
  Avg6 reports exclude them).
subset: a pilot HELDOUT.json: the first <parts> parts of set A with <n> items each and the first <m> items of set B.
grad: at one checkpoint, for each item, the official evaluation input: the model's own chat template (not the A1
  custom template) with enable_thinking False, one user message with the full-size image and the query, the processor
  defaults. The forward is the A1 student forward (DataParallelPPOActor._forward_micro_batch, bf16, one row), at the
  first response token. Letter log probabilities l_X (X = option letters, fresh token ids), log m = logsumexp(l),
  log R = l_gt - log m. g_skill(h) = gradient of the mean of log R over part h, g_fmt(h) = the same for log m, over
  the 248 tensors of G3_PARAMS.txt, fp32 accumulators. Set B: forward only. HELDOUT_VALUES.jsonl: one line per item
  (log R, m, letter argmax correct). Vector files vectors/H__skill__h<h>.pt and H__fmt__h<h>.pt (flat fp32). A part
  with both vector files is skipped (resume).

Usage: python heldout.py select <MME_RealWorld.json> <HELDOUT.json>
       python heldout.py subset <HELDOUT.json> <parts> <n> <m> <PILOT_HELDOUT.json>
       python heldout.py grad <repo> <COMMAND_B200.txt> <model dir> <G3_PARAMS.txt> <HELDOUT.json> <bench dir> <out dir>
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261111
PARTS, PART_SIZE, B_SIZE = 4, 125, 500
OPTION = re.compile(r"(?m)^\(([A-F])\)\s")


def option_letters(query: str) -> list[str] | None:
    found = OPTION.findall(query)
    if not found or found != [chr(ord("A") + i) for i in range(len(found))]:
        return None
    return found


def select(items: list[dict]) -> dict:
    order = list(range(len(items)))
    random.Random(SEED).shuffle(order)
    kept, replaced = [], []
    for i in order:
        it = items[i]
        lets = option_letters(it["query"])
        if lets is None or it["response"] not in lets:
            replaced.append({"index": it["index"], "reason": "no parsable option letters"})
            continue
        if len(it["images"]) != 1:
            raise SystemExit(f"STOP: item {it['index']} has {len(it['images'])} images")
        kept.append({"index": it["index"], "image": it["images"][0], "query": it["query"], "answer": it["response"],
                     "letters": lets})
        if len(kept) == PARTS * PART_SIZE + B_SIZE:
            break
    a = kept[:PARTS * PART_SIZE]
    for k, it in enumerate(a):
        it.update(set="A", part=k // PART_SIZE + 1)
    for it in kept[PARTS * PART_SIZE:]:
        it.update(set="B", part=0)
    return {"seed": SEED, "items": kept, "replaced": replaced, "parts": PARTS, "part_size": PART_SIZE,
            "b_size": B_SIZE}


def subset(sel: dict, parts: int, n: int, m: int) -> dict:
    a = [it for it in sel["items"] if it["set"] == "A" and it["part"] <= parts]
    a = [it for p in range(1, parts + 1) for it in [x for x in a if x["part"] == p][:n]]
    b = [it for it in sel["items"] if it["set"] == "B"][:m]
    return {**sel, "items": a + b, "parts": parts, "part_size": n, "b_size": m, "subset_of": "HELDOUT.json"}


def setup(repo: str):
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(HERE.parent / "vad_grad3"))
    sys.path.insert(0, str(HERE))


class EvalInputs:
    """The official evaluation input of an item (the model's own chat template, enable_thinking False)."""

    def __init__(self, model_dir: str, bench_dir: str):
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
        from verl.utils import hf_processor

        self.processor = hf_processor(model_dir, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        self.stub = AgentLoopWorker.__new__(AgentLoopWorker)
        self.stub.tokenizer, self.stub.processor = self.tokenizer, self.processor
        self.bench = Path(bench_dir)
        self.eos = self.tokenizer.eos_token_id

    def text_image(self, it: dict):
        from PIL import Image

        with Image.open(self.bench / it["image"]) as im:
            image = im.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": it["query"]}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)
        return text, image

    def prompt(self, it: dict):
        text, image = self.text_image(it)
        ids = self.processor(text=[text], images=[image], return_tensors="pt")["input_ids"][0].tolist()
        return ids, image

    def processor_inputs(self, it: dict) -> dict:
        """The processor outputs of the item (input_ids, attention_mask, pixel_values, image_grid_thw, ...)."""
        text, image = self.text_image(it)
        return dict(self.processor(text=[text], images=[image], return_tensors="pt"))

    def actor_inputs(self, it: dict, device: str = "cuda") -> tuple[dict, int]:
        """Inputs of DataParallelPPOActor._forward_micro_batch (the prompt and the response [eos]), prompt tokens."""
        import numpy as np
        import trainer_io

        ids, image = self.prompt(it)
        t, mm = trainer_io._one_trajectory(self.stub, self.tokenizer, ids, [self.eos], [image], len(ids), 1)
        mi = {k: v.to(device) for k, v in t.items()}
        arr = np.empty(1, dtype=object)
        arr[0] = {k: v.to(device) if hasattr(v, "to") else v for k, v in mm.items()}
        mi["multi_modal_inputs"] = arr
        return mi, len(ids)

    def letter_ids(self, letters: list[str]) -> list[int]:
        import letters as lt

        return lt.fresh_ids(self.tokenizer, letters)


def letter_values(lt, gt_i: int):
    """(log R, log m) of the letter log probabilities lt (a tensor)."""
    import torch

    log_m = torch.logsumexp(lt, 0)
    return lt[gt_i] - log_m, log_m


def grad(argv: list[str]):
    repo, command_txt, model_dir, g3_params, heldout_json, bench_dir, out = argv
    setup(repo)
    import grads as g3
    import torch

    out = Path(out)
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(heldout_json).read_text())
    ctx = g3.Context(repo, command_txt, model_dir, g3_params)
    ev = EvalInputs(model_dir, bench_dir)
    actor = ctx.actor
    values_path = out / "HELDOUT_VALUES.jsonl"
    done = {json.loads(x)["index"] for x in open(values_path)} if values_path.exists() else set()

    def student_letters(it, need_grad: bool):
        mi, n_tok = ev.actor_inputs(it)
        with torch.set_grad_enabled(need_grad):
            lp = actor._forward_micro_batch(mi, temperature=1.0, calculate_entropy=False,
                                            return_all_logps=True)["all_logps"][0, 0]
            lt = lp[ev.letter_ids(it["letters"])]
        return lt, n_tok

    def record(f, it, lt, n_tok, t0):
        log_r, log_m = letter_values(lt.detach().double(), it["letters"].index(it["answer"]))
        vals = [float(x) for x in lt.detach()]
        if not all(math.isfinite(x) for x in vals + [float(log_r)]):
            raise SystemExit(f"STOP: non-finite letter log probability for item {it['index']} (stop condition 3)")
        best = max(range(len(vals)), key=lambda i: (vals[i], -i))
        f.write(json.dumps({"index": it["index"], "set": it["set"], "part": it["part"], "letter_logp": vals,
                            "log_R": float(log_r), "m": math.exp(float(log_m)),
                            "correct": it["letters"][best] == it["answer"], "prompt_tokens": n_tok,
                            "seconds": round(time.time() - t0, 3)}) + "\n")
        f.flush()

    parts = sorted({it["part"] for it in sel["items"] if it["set"] == "A"})
    with open(values_path, "a") as f:
        for h in parts:
            files = {k: out / "vectors" / f"H__{k}__h{h}.pt" for k in ("skill", "fmt")}
            if all(p.exists() for p in files.values()):
                continue
            items = [it for it in sel["items"] if it["set"] == "A" and it["part"] == h]
            acc = g3.Accum(ctx.params, ctx.device, ["skill", "fmt"])
            for it in items:
                t0 = time.time()
                lt, n_tok = student_letters(it, True)
                log_r, log_m = letter_values(lt, it["letters"].index(it["answer"]))
                gr = torch.autograd.grad(log_r / len(items), ctx.params, retain_graph=True)
                acc.add("skill", gr, 1.0)
                del gr
                gm = torch.autograd.grad(log_m / len(items), ctx.params)
                acc.add("fmt", gm, 1.0)
                del gm
                if it["index"] not in done:
                    record(f, it, lt, n_tok, t0)
                    done.add(it["index"])
                del lt, log_r, log_m
            for k, p in files.items():
                n2 = acc.sq_norm(k)
                if not math.isfinite(n2):
                    raise SystemExit(f"STOP: non-finite norm of H__{k}__h{h} (stop condition 3)")
                tmp = p.with_suffix(".tmp")
                torch.save(acc.flat(k), tmp)
                os.replace(tmp, p)
            del acc
        for it in sel["items"]:
            if it["set"] == "B" and it["index"] not in done:
                t0 = time.time()
                lt, n_tok = student_letters(it, False)
                record(f, it, lt, n_tok, t0)
                done.add(it["index"])


def main():
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "select":
        res = select(json.loads(Path(argv[0]).read_text()))
        Path(argv[1]).write_text(json.dumps(res, indent=1))
        print(json.dumps({"kept": len(res["items"]), "replaced": len(res["replaced"])}))
    elif cmd == "subset":
        sel = json.loads(Path(argv[0]).read_text())
        Path(argv[4]).write_text(json.dumps(subset(sel, int(argv[1]), int(argv[2]), int(argv[3])), indent=1))
    elif cmd == "grad":
        grad(argv)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
