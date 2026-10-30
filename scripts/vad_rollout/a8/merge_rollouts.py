"""Merge the generation shards of one checkpoint and summarize them (HANDOFF-VLM-VAD-ROLLOUT-001-R9, G4).

Usage: python merge_rollouts.py <merged jsonl> <summary json> <shard jsonl> [...]
The merged file is sorted by trajectory id. The summary gives the counts of correct and incorrect rollouts, the
prompt classes (all-correct, all-wrong, mixed with >= 2 of each, other), and the SHA-256 of the merged file.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402


def main():
    merged, summary_path, shards = sys.argv[1], sys.argv[2], sys.argv[3:]
    rows = sorted((json.loads(line) for path in shards for line in open(path)), key=lambda r: r["traj"])
    if len({r["traj"] for r in rows}) != len(rows):
        raise SystemExit("duplicate trajectory ids in the shards")
    with open(merged, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_pos"]].append(r["correct"])
    classes = defaultdict(int)
    for values in by_prompt.values():
        classes[stats.prompt_class(sum(values), len(values) - sum(values))] += 1
    summary = {
        "file": merged, "sha256": hashlib.sha256(Path(merged).read_bytes()).hexdigest(),
        "trajectories": len(rows), "prompts": len(by_prompt), "correct": sum(r["correct"] for r in rows),
        "incorrect": sum(not r["correct"] for r in rows), "prompt_classes": dict(classes),
        "no_letter": sum(not r["letter"] for r in rows),
        "finish_reasons": {k: sum(r["finish_reason"] == k for r in rows) for k in {r["finish_reason"] for r in rows}},
        "judge_sources": {k: sum(r["judge_source"] == k for r in rows) for k in {r["judge_source"] for r in rows}},
    }
    Path(summary_path).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
