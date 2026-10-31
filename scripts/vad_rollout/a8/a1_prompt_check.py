"""A1 prompt check for the A8 generation (HANDOFF-VLM-VAD-ROLLOUT-001-R9 G4, DT answer after TASK-047). Reads only.

Usage: python a1_prompt_check.py <repo root> <A1 COMMAND.txt> <A1 output root> <out dir> <dataset row> [...]

For each dataset row, it renders the student prompt as generate.py does (trainer_io.prompt_inputs) and finds the A1
rollout dump entries (<A1 root>/rollouts/<step>.jsonl) with the same prompt text. The dump keeps the prompt decoded with
skip_special_tokens=True (ray_trainer.py 531), so this text has no chat-template special tokens and no image tokens.
The prompt token counts (image tokens included) are compared per step with STEP_METRICS.json (prompt_length mean, max,
min) over the dataset rows that the dump text maps to. Step 1 gives the Base distribution (initial weights).
Writes A1_PROMPT_CHECK.json and A1_PROMPT_CHECK.txt in <out dir>.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402
import trainer_io  # noqa: E402

PAD = "<|image_pad|>"


def compress_pads(text: str) -> str:
    return re.sub(rf"(?:{re.escape(PAD)})+", lambda m: f"{PAD}x{len(m.group(0)) // len(PAD)}", text)


def skip_text(tokenizer, processor, messages, cfg) -> str:
    """The prompt text as the A1 dump keeps it, without the image expansion (image tokens are special)."""
    raw = processor.apply_chat_template(messages, tools=[], add_generation_prompt=True, tokenize=False,
                                        **(cfg.data.get("apply_chat_template_kwargs") or {}))
    return tokenizer.decode(tokenizer(raw, add_special_tokens=False)["input_ids"], skip_special_tokens=True)


def message_summary(messages) -> list[dict]:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(c.get("text", "") if c.get("type") == "text" else f"<{c.get('type')}>" for c in content)
        out.append({"role": m["role"], "content": content})
    return out


def first_difference(a: str, b: str) -> dict | None:
    if a == b:
        return None
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"index": i, "a8": a[max(0, i - 80):i + 80], "a1": b[max(0, i - 80):i + 80], "len_a8": len(a), "len_a1": len(b)}


def tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    repo, command_txt, a1_root, out = sys.argv[1:5]
    rows = [int(r) for r in sys.argv[5:]]
    out = Path(out)
    cfg = trainer_io.compose_a1_config(repo, command_txt)
    tokenizer, processor = trainer_io.tokenizer_processor(cfg)
    dataset = trainer_io.load_dataset(cfg, tokenizer, processor)
    template_file = cfg.actor_rollout_ref.model.get("custom_chat_template_file")
    report = {
        "settings": {
            "custom_chat_template_file": template_file,
            "custom_chat_template_sha256": hashlib.sha256(Path(template_file).read_bytes()).hexdigest() if template_file else None,
            "apply_chat_template_kwargs": dict(cfg.data.get("apply_chat_template_kwargs") or {}),
            "image_patch_size": processor.image_processor.patch_size,
            "image_processor_size": dict(getattr(processor.image_processor, "size", {}) or {}),
            "image_processor_min_pixels": getattr(processor.image_processor, "min_pixels", None),
            "image_processor_max_pixels": getattr(processor.image_processor, "max_pixels", None),
            "data_max_prompt_length": cfg.data.max_prompt_length,
            "rollout_prompt_length": cfg.actor_rollout_ref.rollout.prompt_length,
        },
        "rows": {}, "steps": {}}

    # Text index of every dataset row: dump text -> dataset rows.
    index = defaultdict(list)
    for i in range(len(dataset)):
        index[skip_text(tokenizer, processor, dataset[i]["raw_prompt"], cfg)].append(i)
    report["index"] = {"rows": len(dataset), "keys": len(index), "keys_with_two_or_more_rows": sum(len(v) > 1 for v in index.values())}

    dumps = {}
    for path in glob.glob(str(Path(a1_root) / "rollouts" / "*.jsonl")):
        dumps[int(Path(path).stem)] = [json.loads(line) for line in open(path) if line.strip()]
    steps_metrics = json.loads((Path(a1_root) / "STEP_METRICS.json").read_text())

    check_steps = {1}
    for r in rows:
        item = trainer_io.prompt_inputs(dataset, r, processor, cfg)
        ids = item["prompt_ids"]
        key = skip_text(tokenizer, processor, item["row"]["raw_prompt"], cfg)
        prompt_text = tokenizer.decode(ids, skip_special_tokens=True)
        gt = item["row"]["reward_model"]["ground_truth"]
        entry = {"gt": gt, "messages": message_summary(item["row"]["raw_prompt"]),
                 "rendered_prompt": compress_pads(tokenizer.decode(ids, skip_special_tokens=False)),
                 "prompt_tokens": len(ids), "image_tokens": ids.count(tokenizer.convert_tokens_to_ids(PAD)),
                 "image_sizes": [list(im.size) for im in item["images"]],
                 "skip_text_equals_decoded_ids": key == prompt_text, "rows_with_same_text": index[key],
                 "appearances": []}
        for step in sorted(dumps):
            hits = [e for e in dumps[step] if e["input"] == key]
            if not hits:
                continue
            check_steps.add(step)
            entry["appearances"].append({"step": step, "dump_input": hits[0]["input"], "responses": [
                {"uid": e.get("uid"), "output": e["output"], "response_tokens": tokens(tokenizer, e["output"]),
                 "words": len(e["output"].split()), "a1_score": e.get("score"), "gts": e.get("gts"),
                 **generate.grade(prompt_text, gt, e["output"])} for e in hits]})
        if not entry["appearances"]:  # near matches: same ground truth and the same last user text
            tail = entry["messages"][-1]["content"][-200:].replace("<image>", "").strip()
            near = [(s, e) for s in sorted(dumps) for e in dumps[s] if e.get("gts") == gt and tail and tail in e["input"]]
            entry["near_matches"] = [{"step": s, "difference": first_difference(key, e["input"])} for s, e in near[:4]]
        report["rows"][str(r)] = entry

    # Prompt token counts per step against STEP_METRICS.json.
    for step in sorted(check_steps):
        if step not in dumps:
            continue
        lengths, unmatched, ambiguous, cache = [], 0, 0, {}
        for e in dumps[step]:
            cands = index.get(e["input"], [])
            if not cands:
                unmatched += 1
                continue
            if e["input"] not in cache:
                cache[e["input"]] = {len(trainer_io.prompt_inputs(dataset, c, processor, cfg)["prompt_ids"]) for c in cands}
            if len(cache[e["input"]]) > 1:
                ambiguous += 1
                continue
            lengths.append(next(iter(cache[e["input"]])))
        m = steps_metrics.get(str(step), {})
        report["steps"][str(step)] = {
            "rollouts": len(dumps[step]), "matched": len(lengths), "unmatched": unmatched, "ambiguous": ambiguous,
            "a8_prompt_length": {"mean": sum(lengths) / len(lengths) if lengths else None,
                                 "max": max(lengths, default=None), "min": min(lengths, default=None)},
            "a1_prompt_length": {k: m.get(f"prompt_length/{k}") for k in ("mean", "max", "min")}}

    # Base distribution: step 1.
    if 1 in dumps:
        outs = dumps[1]
        lens = [tokens(tokenizer, e["output"]) for e in outs]
        by_uid = defaultdict(list)
        graded = []
        for e, n in zip(outs, lens):
            g = generate.grade(e["input"], e["gts"], e["output"])
            graded.append(g)
            by_uid[e.get("uid")].append(n)
        report["step1"] = {
            "rollouts": len(outs), "prompts": len(by_uid),
            "correct_a8_grader": sum(g["correct"] for g in graded), "no_letter": sum(g["letter"] is None for g in graded),
            "response_tokens_reencoded": {"mean": sum(lens) / len(lens), "max": max(lens), "min": min(lens),
                                          "le_4": sum(n <= 4 for n in lens), "5_32": sum(5 <= n <= 32 for n in lens),
                                          "33_256": sum(33 <= n <= 256 for n in lens), "gt_256": sum(n > 256 for n in lens)},
            "prompts_with_all_responses_le_4": sum(all(n <= 4 for n in v) for v in by_uid.values()),
            "a1_metrics": {k: steps_metrics.get("1", {}).get(k) for k in
                           ("response_length/mean", "response_length/max", "response_length/min", "prompt_length/mean")}}

    (out / "A1_PROMPT_CHECK.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    lines = [f"settings: {json.dumps(report['settings'])}", f"index: {json.dumps(report['index'])}"]
    for r, e in report["rows"].items():
        lines.append(f"row {r}: gt {e['gt']} prompt_tokens {e['prompt_tokens']} image_tokens {e['image_tokens']} "
                     f"image_sizes {e['image_sizes']} same_text_rows {e['rows_with_same_text']} "
                     f"appearances {[a['step'] for a in e['appearances']]} near {e.get('near_matches')}")
        for a in e["appearances"]:
            lines.append(f"  step {a['step']}: " + "; ".join(
                f"{x['response_tokens']} tok {x['words']} words correct {x['correct']} letter {x['letter']}" for x in a["responses"]))
    for s, v in report["steps"].items():
        lines.append(f"step {s} prompt length: {json.dumps(v)}")
    if "step1" in report:
        lines.append(f"step 1 base: {json.dumps(report['step1'])}")
    (out / "A1_PROMPT_CHECK.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
