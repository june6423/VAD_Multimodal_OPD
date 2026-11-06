"""Rollout accuracy of one arm (HANDOFF-VLM-VAD-PACKFIX-001-R6 Definition 8, DIRECTIVE-VLM-VAD-PACKFIX-001-007).

The A1 training sets a zero reward and calls no grader. This script scores the rollout dumps <arm>/rollouts/<s>.jsonl
after the training, in the judge session of the arm, with the A1 reward function
verl.utils.reward_score.opd_benchmark_judge.compute_score_batched and the A1 reward_kwargs (REWARD_KWARGS, from the
custom_reward_function.reward_kwargs of A1). Inputs of each row as in the batch reward manager: data_source,
the response text ("output"), the ground truth ("gts") and extra_info (hostmem.py item 9 adds data_source and
extra_info to the dump).
ROLLOUT_ACC.json gives, for each step: the row count, the mean score (rollout accuracy), the judge_source counts,
the count of judge calls (rows with judge_source "llm"), the wall seconds, and the per-row scores.
A step that already has a result is not scored again. The check fails (exit 1) when a step in 1..<steps> has no dump,
a row lacks an input, or a score is missing.
Usage: python rollout_acc.py <arm output root> <steps> <judge api base>
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

JUDGE_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
REWARD_KWARGS = {"judge_max_tokens": 2048, "parallel_workers": 8, "request_timeout": 600,
                 "min_request_interval": 0.0, "enable_thinking": False, "thinking_extra_body_style": "direct",
                 "show_progress": False}
FIELDS = ("data_source", "output", "gts", "extra_info")


def score_step(rows, compute):
    """compute(data_sources, solution_strs, ground_truths, extra_infos) -> list of result dicts with "score"."""
    t0 = time.perf_counter()
    results = compute([r["data_source"] for r in rows], [r["output"] for r in rows], [r["gts"] for r in rows],
                      [r["extra_info"] for r in rows])
    seconds = time.perf_counter() - t0
    scores = [float(r["score"]) for r in results]
    sources = Counter(str(r.get("judge_source")) for r in results)
    return {"rows": len(rows), "acc_mean": sum(scores) / len(scores) if scores else None,
            "judge_source": dict(sources), "judge_calls": sources.get("llm", 0), "seconds": round(seconds, 2),
            "row_score": scores}


def main(out, steps, compute):
    path = out / "ROLLOUT_ACC.json"
    result = json.loads(path.read_text()) if path.exists() else {"steps": {}}
    problems = []
    for s in range(1, steps + 1):
        if str(s) in result["steps"]:
            continue
        dump = out / "rollouts" / f"{s}.jsonl"
        if not dump.exists():
            problems.append([s, "no dump"])
            continue
        rows = [json.loads(line) for line in dump.open() if line.strip()]
        lacking = sorted({f for r in rows for f in FIELDS if f not in r})
        if lacking or not rows:
            problems.append([s, f"rows {len(rows)} lacking {lacking}"])
            continue
        result["steps"][str(s)] = score_step(rows, compute)
        print(f"rollout_acc step {s} rows {len(rows)} acc {result['steps'][str(s)]['acc_mean']} judge_calls "
              f"{result['steps'][str(s)]['judge_calls']} seconds {result['steps'][str(s)]['seconds']}", flush=True)
        path.write_text(json.dumps(result, indent=1))
    result["check"] = {"steps": steps, "problems": problems,
                       "pass": not problems and all(str(s) in result["steps"] for s in range(1, steps + 1))}
    result["reward_kwargs"] = {**REWARD_KWARGS, "judge_model": JUDGE_MODEL}
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["check"]))
    return 0 if result["check"]["pass"] else 1


if __name__ == "__main__":
    from verl.utils.reward_score.opd_benchmark_judge import compute_score_batched

    api_base = sys.argv[3]

    def compute(data_sources, solution_strs, ground_truths, extra_infos):
        return compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos,
                                     judge_api_base=api_base, judge_model=JUDGE_MODEL, **REWARD_KWARGS)

    sys.exit(main(Path(sys.argv[1]), int(sys.argv[2]), compute))
