"""Labels of the TASK-VLM-VAD-ROLLOUT-001-053 trajectories (HANDOFF-VLM-VAD-GRAD-001-R1 Part A item 4, gate G5).

Rule of the A1 training reward (DT proposal D2): opd_benchmark_judge.judge_batched of the A1 code (2627743a, unchanged
at a5ace7be) with the A1 reward kwargs (TASK-022 COMMAND.txt lines 132-142). It runs the rule grader on every row and
sends only the rule misses to the Mistral judge (temperature 0). The inputs are the ones of the A1 batch reward manager
(batch.py 62-74): data_source, reward_model.ground_truth and extra_info of the dataset row, and the decoded response.
- A rule-graded row must equal its TASK-053 label (rule_judge_one at generation). A difference stops the script.
- A judge output that starts with "[JUDGE_API_ERROR]" is a transport error: up to 3 more rounds for those rows only.
  A row that still has no label stops the script (fallback table: STATUS before the gradients).
- R8 rule: a judge output that canonical_label (repair_judges.py) maps keeps that label. Every other output is
  incorrect (source noncanonical_as_incorrect), with no retry. The raw output stays in judge_raw.
Usage: python label.py <repo root> <mapped A1 COMMAND.txt> <out dir> <judge api base> <name>=<rollouts.jsonl> [...]
Writes <out dir>/<name>/LABELS.jsonl (traj, label, source, judge_raw, rule_label) and <out dir>/LABELS_SUMMARY.json.
LABELS_SUMMARY.json also counts the permissive labels (permissive.py, DT reading tool) for the POST_RUN agreement table.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vad_rollout" / "a8"))
import trainer_io  # noqa: E402
from permissive import permissive_label  # noqa: E402

RULE_SOURCES = ("rule", "first letter", "yes/no")
ERROR = "[JUDGE_API_ERROR]"
RETRY_ROUNDS = 3


def a1_kwargs(api_base: str) -> dict:
    return {"judge_api_base": api_base, "judge_model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "judge_max_tokens": 2048, "parallel_workers": 8, "request_timeout": 600, "min_request_interval": 0.0,
            "enable_thinking": False, "thinking_extra_body_style": "direct", "show_progress": False}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_rows(rows: list[dict], inputs: list[tuple], judge, canonical) -> tuple[list[dict], dict]:
    """rows: TASK-053 rows. inputs: (data_source, response, gt, extra_info) per row. judge: judge_batched."""
    results = judge(*map(list, zip(*inputs)))
    rounds = 0
    for _ in range(RETRY_ROUNDS):
        bad = [i for i, r in enumerate(results) if str(r.get("judge", "")).startswith(ERROR)]
        if not bad:
            break
        rounds += 1
        again = judge(*map(list, zip(*[inputs[i] for i in bad])))
        for i, r in zip(bad, again):
            results[i] = r
    out, counts = [], Counter()
    for row, res in zip(rows, results):
        source, raw = res.get("judge_source"), res.get("judge")
        rec = {"traj": row["traj"], "rule_label": int(bool(row["correct"])), "judge_raw": None}
        if source in RULE_SOURCES:
            if not row["correct"] or row["judge_source"] != source:
                raise SystemExit(f"STOP: rule label of trajectory {row['traj']} differs from TASK-053 "
                                 f"({source} vs {row['judge_source']}, correct {row['correct']})")
            rec.update(label=1, source="rule")
        elif str(raw).startswith(ERROR):
            rec.update(label=None, source="transport_error", judge_raw=raw)
        else:
            if row["correct"]:
                raise SystemExit(f"STOP: trajectory {row['traj']} was rule-correct in TASK-053 but went to the judge")
            mapped = canonical(raw)
            rec.update(judge_raw=raw, label=int(mapped == "Yes"),
                       source="judge" if mapped else "noncanonical_as_incorrect")
            counts["judge_calls"] += 1
        counts[rec["source"]] += 1
        out.append(rec)
    counts["retry_rounds"] = rounds
    return out, dict(counts)


def prompt_classes(rows: list[dict], labels: dict[int, int | None]) -> dict:
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_pos"], []).append(labels[r["traj"]] or 0)
    out = Counter()
    for v in by_prompt.values():
        c, i = sum(v), len(v) - sum(v)
        out["mixed" if c >= 2 and i >= 2 else "all_correct" if i == 0 else "all_wrong" if c == 0 else "other"] += 1
    return dict(out)


def main():
    repo, command_txt, out, api_base = sys.argv[1:5]
    specs = [s.split("=", 1) for s in sys.argv[5:]]
    out = Path(out)
    sys.path.insert(0, str(Path(repo).resolve() / "evaluation" / "vision_opd_official"))
    from repair_judges import canonical_label

    from verl.utils.reward_score.opd_benchmark_judge import judge_batched

    cfg = trainer_io.compose_a1_config(repo, command_txt)
    tokenizer, processor = trainer_io.tokenizer_processor(cfg)
    frame = trainer_io.load_dataset(cfg, tokenizer, processor).dataframe
    kwargs = a1_kwargs(api_base)
    summary = {"judge": kwargs, "checkpoints": {}}
    stop = False
    for name, rollouts in specs:
        start = time.time()
        rows = [json.loads(line) for line in open(rollouts)]
        inputs = []
        for r in rows:
            ds = frame[r["dataset_index"]]
            gt = ds["reward_model"]["ground_truth"]
            if gt != r["gt"]:
                raise SystemExit(f"STOP: ground truth of trajectory {r['traj']} differs from the dataset row")
            inputs.append((ds["data_source"], r["response_text"], gt, ds["extra_info"]))
        recs, counts = label_rows(rows, inputs, lambda *a: judge_batched(*a, **kwargs), canonical_label)
        (out / name).mkdir(parents=True, exist_ok=True)
        path = out / name / "LABELS.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        judged = {r["traj"]: r["label"] for r in recs}
        strict = {r["traj"]: r["rule_label"] for r in recs}
        agree = Counter((strict[t], judged[t]) for t in judged)
        permissive = {r["traj"]: permissive_label(r["response_text"], r["gt"]) for r in rows}
        joint = Counter((strict[t], judged[t], permissive[t]) for t in judged)
        summary["checkpoints"][name] = {
            "rollouts": rollouts, "rollouts_sha256": sha256(rollouts), "labels_sha256": sha256(path),
            "trajectories": len(recs), "counts": counts, "seconds": round(time.time() - start, 1),
            "correct": {"judge_based": sum(v or 0 for v in judged.values()), "strict_rule_only": sum(strict.values()),
                        "permissive": sum(permissive.values())},
            "strict_vs_judge": {f"{a}->{b}": n for (a, b), n in sorted(agree.items(), key=str)},
            "strict_judge_permissive": {f"{a},{b},{c}": n for (a, b, c), n in sorted(joint.items(), key=str)},
            "prompt_classes": prompt_classes(rows, judged),
        }
        if counts.get("transport_error"):
            stop = True
    (out / "LABELS_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    if stop:
        raise SystemExit("STOP: rows without a label after the judge retries (transport errors)")


if __name__ == "__main__":
    main()
