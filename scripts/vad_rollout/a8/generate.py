"""A8 rollouts of one checkpoint (HANDOFF-VLM-VAD-ROLLOUT-001-R9, G4).

Prompts: the 375 dataset rows drawn with seed 42 from the A1 training parquet (sorted indices). Prompt tokens and
images follow the A1 agent loop (trainer_io.prompt_inputs). Sampling follows the A1 rollout (vllm_async_server.py
474-504): 8 separate requests per prompt, temperature 1.0, top_p 1.0, no top_k, max_tokens = min(1024, max_model_len
- prompt length), sampled-token logprobs, the four vision tokens as bad words, engine seed 0. Each rollout is graded
with the A1 rule grader (rule_judge_one, correct iff "Yes").

Usage: python generate.py <repo root> <A1 COMMAND.txt> <model dir> <out jsonl> <shard> <shards> [prompts] [siblings]
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trainer_io  # noqa: E402

DATA_SOURCE = "zwz_rl_vqa_bbox_teacher"
VISION_TOKENS = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
PROMPT_SEED = 42


def prompt_indices(dataset_size: int, prompts: int) -> list[int]:
    return sorted(random.Random(PROMPT_SEED).sample(range(dataset_size), prompts))


def grade(prompt_text: str, gt: str, response_text: str) -> dict:
    from verl.utils.reward_score.opd_benchmark_judge import extract_answer, extract_first_option, rule_judge_one

    result = rule_judge_one(DATA_SOURCE, prompt_text, gt, response_text)
    return {"correct": result.get("judge") == "Yes", "judge": result.get("judge"), "judge_source": result.get("judge_source"),
            "letter": extract_first_option(extract_answer(response_text))}


def main():
    repo, command_txt, model_dir, out_path = sys.argv[1:5]
    shard, shards = int(sys.argv[5]), int(sys.argv[6])
    prompts = int(sys.argv[7]) if len(sys.argv) > 7 else 375
    siblings = int(sys.argv[8]) if len(sys.argv) > 8 else 8
    from vllm import LLM, SamplingParams

    cfg = trainer_io.compose_a1_config(repo, command_txt)
    tokenizer, processor = trainer_io.tokenizer_processor(cfg)
    dataset = trainer_io.load_dataset(cfg, tokenizer, processor)
    positions = list(enumerate(prompt_indices(len(dataset), prompts)))[shard::shards]
    rollout = cfg.actor_rollout_ref.rollout
    llm = LLM(model=model_dir, tokenizer=cfg.actor_rollout_ref.model.path, tensor_parallel_size=1, seed=0,
              max_model_len=rollout.max_model_len, limit_mm_per_prompt={"image": 1}, trust_remote_code=False)
    requests, meta = [], []
    for pos, index in positions:
        item = trainer_io.prompt_inputs(dataset, index, processor, cfg)
        prompt = {"prompt_token_ids": item["prompt_ids"], "multi_modal_data": {"image": item["images"]}}
        params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=min(rollout.response_length,
                                rollout.max_model_len - len(item["prompt_ids"])), logprobs=0, bad_words=VISION_TOKENS)
        for s in range(siblings):
            requests.append((prompt, params))
            meta.append((pos, index, s, item))
    outputs = llm.generate([r[0] for r in requests], sampling_params=[r[1] for r in requests])
    with open(out_path, "w") as f:
        for (pos, index, s, item), output in zip(meta, outputs):
            completion = output.outputs[0]
            response_ids = list(completion.token_ids)
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            prompt_text = tokenizer.decode(item["prompt_ids"], skip_special_tokens=True)
            gt = item["row"]["reward_model"]["ground_truth"]
            logprobs = [next(iter(lp.values())).logprob if lp else None for lp in (completion.logprobs or [])]
            row = {"traj": pos * siblings + s, "prompt_pos": pos, "dataset_index": index, "sibling": s, "gt": gt,
                   "prompt_len": len(item["prompt_ids"]),
                   "prompt_sha256": hashlib.sha256(json.dumps(item["prompt_ids"]).encode()).hexdigest(),
                   "response_ids": response_ids, "response_text": text, "rollout_logprobs": logprobs,
                   "finish_reason": completion.finish_reason, **grade(prompt_text, gt, text)}
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
