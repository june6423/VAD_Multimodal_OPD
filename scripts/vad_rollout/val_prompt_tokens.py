#!/usr/bin/env python3
"""VLM-VAD-ROLLOUT-001 validation prompt lengths (DIRECTIVE-VLM-VAD-ROLLOUT-001-003, item 2).

Usage: val_prompt_tokens.py <model dir> <chat template file> <out json> <parquet> [<parquet> ...]

Counts the prompt tokens of every row as the rollout does: RLHFDataset._build_messages, then
qwen_vl_utils.process_vision_info with the processor patch size (AgentLoopBase.process_vision_info), then
processor.apply_chat_template with the custom chat template and the processor call
(AgentLoopBase.apply_chat_template). Prints per file and benchmark the row count, the maximum, the mean, and
the rows above 8192 and above 9216. Also checks that the rows of a later file equal the first rows of the
first file (the no8k parquet against the four-benchmark parquet).
"""

import json
import sys
from types import SimpleNamespace

import datasets
import pandas as pd
from qwen_vl_utils import process_vision_info

from verl.utils import hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset

LIMITS = (8192, 9216)


def main() -> None:
    model_dir, template_file, out_path, *parquets = sys.argv[1:]
    processor = hf_processor(model_dir, trust_remote_code=True)
    with open(template_file, encoding="utf-8") as f:
        processor.chat_template = f.read()
    patch_size = processor.image_processor.patch_size
    builder = SimpleNamespace(prompt_key="prompt", image_key="images", video_key="videos", processor=processor)

    result = {"model_dir": model_dir, "chat_template_file": template_file, "image_patch_size": patch_size,
              "files": {}}
    frames = {}
    for path in parquets:
        frame = pd.read_parquet(path)
        frames[path] = frame
        rows = datasets.Dataset.from_pandas(frame, preserve_index=False)
        lengths = {}
        top = []
        for index in range(len(rows)):
            row = dict(rows[index])
            messages = RLHFDataset._build_messages(builder, row)
            images, _ = process_vision_info(messages, image_patch_size=patch_size, return_video_metadata=True)
            raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            n = len(processor(text=[raw_prompt], images=images)["input_ids"][0])
            lengths.setdefault(row["data_source"], []).append(n)
            top.append((n, row["data_source"], row["extra_info"]["sample_id"]))
        summary = {}
        for source, values in lengths.items():
            summary[source] = {"rows": len(values), "max": max(values), "mean": sum(values) / len(values)}
            for limit in LIMITS:
                summary[source][f"above_{limit}"] = sum(v > limit for v in values)
        top.sort(reverse=True)
        result["files"][path] = {"rows": len(rows), "benchmarks": summary,
                                 "longest_10": [{"tokens": n, "benchmark": s, "sample_id": i} for n, s, i in top[:10]]}
        print(path, json.dumps(summary, indent=1), flush=True)

    first = parquets[0]
    for path in parquets[1:]:
        head = frames[first].head(len(frames[path])).reset_index(drop=True)
        equal = head.to_json(orient="records") == frames[path].reset_index(drop=True).to_json(orient="records")
        result["files"][path]["rows_equal_first_rows_of"] = {"file": first, "equal": equal}
        print(path, "rows equal the first rows of", first, equal)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
