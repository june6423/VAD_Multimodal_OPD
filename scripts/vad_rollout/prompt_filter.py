#!/usr/bin/env python3
"""VLM-VAD-ROLLOUT-001 input-length filter (researcher decision of 2026-10-29 after DIRECTIVE-VLM-VAD-ROLLOUT-001-003).

Usage: prompt_filter.py <model dir> <chat template file> <max model len> <workers> <summary json> \
    <in parquet> <out parquet> <skip jsonl> [<in parquet> <out parquet> <skip jsonl> ...]

Counts the rollout prompt tokens of every row as the agent loop does: RLHFDataset._build_messages, then
qwen_vl_utils.process_vision_info with the processor patch size (AgentLoopBase.process_vision_info), then
processor.apply_chat_template with the custom chat template and the processor call
(AgentLoopBase.apply_chat_template). A row with prompt tokens >= <max model len> has no room for one response
token (vllm_async_server.py generate), so it goes to <skip jsonl> and not to <out parquet>. The other rows keep
their order. <summary json> holds per file and data_source the rows, the maximum, the mean, the rows above 8192,
the skipped rows, and the SHA-256 of the input and output files.
"""

import hashlib
import json
import sys
from multiprocessing import Pool
from types import SimpleNamespace

import datasets
import pandas as pd
from qwen_vl_utils import process_vision_info

from verl.utils import hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset

_state = {}


def init_worker(model_dir: str, template_file: str) -> None:
    processor = hf_processor(model_dir, trust_remote_code=True)
    with open(template_file, encoding="utf-8") as f:
        processor.chat_template = f.read()
    _state["processor"] = processor
    _state["builder"] = SimpleNamespace(prompt_key="prompt", image_key="images", video_key="videos",
                                        processor=processor)


def prompt_tokens(row: dict) -> int:
    processor = _state["processor"]
    messages = RLHFDataset._build_messages(_state["builder"], dict(row))
    images, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size,
                                    return_video_metadata=True)
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(processor(text=[raw_prompt], images=images)["input_ids"][0])


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    model_dir, template_file, limit, workers, summary_path, *triples = sys.argv[1:]
    limit, workers = int(limit), int(workers)
    result = {"model_dir": model_dir, "chat_template_file": template_file, "max_model_len": limit,
              "skip_rule": "prompt tokens >= max_model_len", "files": {}}
    with Pool(workers, initializer=init_worker, initargs=(model_dir, template_file)) as pool:
        for in_path, out_path, skip_path in zip(triples[0::3], triples[1::3], triples[2::3], strict=True):
            frame = pd.read_parquet(in_path)
            rows = datasets.Dataset.from_pandas(frame, preserve_index=False).to_list()
            lengths = pool.map(prompt_tokens, rows, chunksize=16)
            keep = [i for i, n in enumerate(lengths) if n < limit]
            frame.iloc[keep].reset_index(drop=True).to_parquet(out_path, index=False)
            summary = {}
            with open(skip_path, "w", encoding="utf-8") as f:
                for i, (row, n) in enumerate(zip(rows, lengths, strict=True)):
                    source = row.get("data_source", "unknown")
                    item = summary.setdefault(source, {"rows": 0, "max": 0, "sum": 0, "above_8192": 0, "skipped": 0})
                    item["rows"] += 1
                    item["max"] = max(item["max"], n)
                    item["sum"] += n
                    item["above_8192"] += n > 8192
                    if n >= limit:
                        item["skipped"] += 1
                        extra = row.get("extra_info") or {}
                        f.write(json.dumps({"row": i, "data_source": source, "index": extra.get("index"),
                                            "sample_id": extra.get("sample_id"), "prompt_tokens": n}) + "\n")
            for item in summary.values():
                item["mean"] = item.pop("sum") / item["rows"]
            result["files"][in_path] = {"rows": len(rows), "kept": len(keep), "skipped": len(rows) - len(keep),
                                        "input_sha256": sha256(in_path), "output": out_path,
                                        "output_sha256": sha256(out_path), "skip_file": skip_path,
                                        "by_data_source": summary}
            print(in_path, json.dumps(result["files"][in_path], indent=1), flush=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
