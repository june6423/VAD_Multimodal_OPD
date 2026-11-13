"""Held-out set B values of HANDOFF-VLM-VAD-VIEW-002-R1 Definition 6 on one checkpoint.

The code is the alpha-0 point of the VIEW-001 line search (exp/VLM-VAD-VIEW-001 scripts/vad_view/linesearch.py
load_model and measure, heldout.py EvalInputs and letter_values, letters.py fresh_ids), copied here:
  model: the merged checkpoint loaded with AutoModelForImageTextToText in fp32, attn_implementation "sdpa", no verl
    monkey patch, every parameter rounded to bf16 and back;
  input: the official evaluation input of the item (the model's own chat template with enable_thinking False, one user
    message with the full-size image and the query, the processor defaults of verl.utils.hf_processor);
  values: the fp32 log softmax of the logits at the last prompt position (the first response token); l_X = the log
    probability of each option letter X (one-token ids, tokenizer.encode(X)); log m = logsumexp(l); log R = l_gt - log m;
    correct = the letter of the largest l_X (the first letter on a tie) is the answer.
Items: the set B items of HELDOUT.json (VIEW-001 Definition 7, 500 items), in file order; env HELDOUT_N (0 = all) keeps
the first n (the pilot uses 10).
Writes <out>/HELDOUT_B.jsonl (one line per item; a restart skips the items that have a line) and <out>/HELDOUT_B.json
(n, mean log R, mean m, accuracy). A non-finite value stops the script (exit 1).
Usage: python heldout.py <repo> <merged model dir> <HELDOUT.json> <bench dir> <out dir>
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path


def fresh_ids(tokenizer, letters: list[str]) -> list[int]:
    ids = []
    for letter in letters:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f"STOP: letter {letter} is not one token: {enc}")
        ids.append(enc[0])
    return ids


def letter_values(lt, gt_i: int):
    """(log R, log m) of the letter log probabilities lt (a tensor)."""
    import torch

    log_m = torch.logsumexp(lt, 0)
    return lt[gt_i] - log_m, log_m


def summary(lines: list[dict]) -> dict:
    n = len(lines)
    return {"n": n, "mean_log_R": sum(x["log_R"] for x in lines) / n, "mean_m": sum(x["m"] for x in lines) / n,
            "accuracy": sum(bool(x["correct"]) for x in lines) / n}


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
    repo, model_dir, heldout_json, bench_dir, out = sys.argv[1:]
    sys.path.insert(0, str(Path(repo).resolve()))
    import torch
    from PIL import Image
    from verl.utils import hf_processor

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads(Path(heldout_json).read_text())["items"] if it["set"] == "B"]
    n_keep = int(os.environ.get("HELDOUT_N", "0"))
    items = items[:n_keep] if n_keep else items
    path = out / "HELDOUT_B.jsonl"
    done = {json.loads(x)["index"] for x in open(path)} if path.exists() else set()
    t0 = time.time()
    model = load_model(model_dir)
    processor = hf_processor(model_dir, use_fast=True)
    print(f"model loaded in {time.time() - t0:.1f} s, {len(items)} items, {len(done)} done", flush=True)
    with open(path, "a") as f:
        for it in items:
            if it["index"] in done:
                continue
            t1 = time.time()
            with Image.open(Path(bench_dir) / it["image"]) as im:
                image = im.convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": it["query"]}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False)
            inp = dict(processor(text=[text], images=[image], return_tensors="pt"))
            ids = fresh_ids(processor.tokenizer, it["letters"])
            with torch.no_grad():
                logits = model(**{k: v.cuda() for k, v in inp.items()}, use_cache=False, logits_to_keep=1).logits[0, -1]
                lt = torch.log_softmax(logits.double(), -1)[ids]
                log_r, log_m = letter_values(lt, it["letters"].index(it["answer"]))
            vals = [float(x) for x in lt]
            if not all(math.isfinite(x) for x in vals):
                raise SystemExit(f"STOP: non-finite letter log probability for item {it['index']}")
            best = max(range(len(vals)), key=lambda i: (vals[i], -i))
            f.write(json.dumps({"index": it["index"], "letter_logp": vals, "log_R": float(log_r),
                                "m": math.exp(float(log_m)), "correct": it["letters"][best] == it["answer"],
                                "prompt_tokens": int(inp["input_ids"].shape[1]),
                                "seconds": round(time.time() - t1, 3)}) + "\n")
            f.flush()
    lines = [json.loads(x) for x in open(path)]
    want = {it["index"] for it in items}
    lines = [x for x in lines if x["index"] in want]
    if len(lines) != len(items):
        raise SystemExit(f"STOP: {len(lines)} lines for {len(items)} items")
    res = {**summary(lines), "model_dir": model_dir, "heldout": heldout_json, "seconds": round(time.time() - t0, 1)}
    (out / "HELDOUT_B.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
