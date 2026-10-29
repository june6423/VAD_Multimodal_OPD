#!/usr/bin/env python3
"""Paired comparison of two training arms for VLM-VAD-ROLLOUT-001 (HANDOFF-VLM-VAD-ROLLOUT-001-R6, Part A item 1, gate G6).

Usage: python3 scripts/vad_rollout/compare_arms.py <A1 eval root> <A2 eval root> <A1 train root> <A2 train root> <out dir>

Inputs:
  <eval root>/step_<s>/run/judge/<benchmark>/*_step<s>_seed42_answer.jsonl for s in 80, 130: the judge results of
  eval_checkpoint.sh (a JSON array; a row is correct when judge is "yes", as evaluation/vision_opd_official/cal_acc.py).
  <train root>/STEP_METRICS.json: the per-step metrics of train_arm.sh with val-core/<benchmark>/acc/mean@1.
Rows join by (benchmark, sample_uid). For each step and benchmark, and for the union of the six benchmarks, the table
gives the rows, the accuracy of each arm in percent, the difference A2 - A1, both-correct, only-A1, only-A2,
both-wrong, and the two-sided exact sign test p-value on the only-A1 and only-A2 counts. Avg6 is the mean of the six
benchmark accuracies. The curves give the validation accuracy of the four in-training benchmarks and their average
per validation step, and the maximum absolute difference of the average.
Outputs: <out dir>/compare.json and <out dir>/compare.md, with the SHA-256 of every input file. CPU only.
"""

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

STEPS = (80, 130)
BENCHMARKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
CURVE_BENCHMARKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sign_test_p(only_a: int, only_b: int) -> float:
    """Two-sided exact binomial test of only_a against only_b with p = 0.5."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def judge_file(eval_root: Path, step: int, benchmark: str) -> Path:
    files = sorted((eval_root / f"step_{step}" / "run" / "judge" / benchmark).glob(f"*_step{step}_seed42_answer.jsonl"))
    if len(files) != 1:
        raise SystemExit(f"expected one judge file for step {step} {benchmark} under {eval_root}, found {len(files)}")
    return files[0]


def load_correct(path: Path) -> dict:
    rows = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        key = item["sample_uid"]
        if key in rows:
            raise SystemExit(f"duplicate sample_uid {key} in {path}")
        rows[key] = str(item.get("judge", "")).strip().lower() == "yes"
    return rows


def paired(a: dict, b: dict) -> dict:
    if set(a) != set(b):
        raise SystemExit(f"row sets differ: {len(set(a) ^ set(b))} rows in one arm only")
    both = sum(a[k] and b[k] for k in a)
    only_a = sum(a[k] and not b[k] for k in a)
    only_b = sum(b[k] and not a[k] for k in a)
    n = len(a)
    acc_a = 100.0 * (both + only_a) / n
    acc_b = 100.0 * (both + only_b) / n
    return {"rows": n, "acc_A1": acc_a, "acc_A2": acc_b, "diff_A2_minus_A1": acc_b - acc_a, "both_correct": both,
            "only_A1": only_a, "only_A2": only_b, "both_wrong": n - both - only_a - only_b,
            "sign_test_p": sign_test_p(only_a, only_b)}


def curves(a1_metrics: dict, a2_metrics: dict) -> dict:
    def series(metrics):
        out = {}
        for step, values in metrics.items():
            accs = [values.get(f"val-core/{b}/acc/mean@1") for b in CURVE_BENCHMARKS]
            if all(v is not None for v in accs):
                out[int(step)] = {**dict(zip(CURVE_BENCHMARKS, accs)), "average": sum(accs) / len(accs)}
        return out

    s1, s2 = series(a1_metrics), series(a2_metrics)
    common = sorted(set(s1) & set(s2))
    diffs = {step: s2[step]["average"] - s1[step]["average"] for step in common}
    worst = max(common, key=lambda step: abs(diffs[step])) if common else None
    return {"A1": s1, "A2": s2, "steps_both": common, "average_diff_A2_minus_A1": diffs,
            "max_abs_average_diff": abs(diffs[worst]) if common else None, "max_abs_average_diff_step": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("a1_eval", "a2_eval", "a1_train", "a2_train", "out_dir"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {}
    table = {}
    for step in STEPS:
        union_a1, union_a2 = {}, {}
        per_benchmark = {}
        for benchmark in BENCHMARKS:
            files = [judge_file(args.a1_eval, step, benchmark), judge_file(args.a2_eval, step, benchmark)]
            inputs.update({str(p): sha256_file(p) for p in files})
            a1, a2 = load_correct(files[0]), load_correct(files[1])
            per_benchmark[benchmark] = paired(a1, a2)
            union_a1.update({(benchmark, k): v for k, v in a1.items()})
            union_a2.update({(benchmark, k): v for k, v in a2.items()})
        avg6_a1 = sum(per_benchmark[b]["acc_A1"] for b in BENCHMARKS) / len(BENCHMARKS)
        avg6_a2 = sum(per_benchmark[b]["acc_A2"] for b in BENCHMARKS) / len(BENCHMARKS)
        table[str(step)] = {"benchmarks": per_benchmark, "union": paired(union_a1, union_a2),
                            "avg6": {"A1": avg6_a1, "A2": avg6_a2, "diff_A2_minus_A1": avg6_a2 - avg6_a1}}

    metric_files = [args.a1_train / "STEP_METRICS.json", args.a2_train / "STEP_METRICS.json"]
    inputs.update({str(p): sha256_file(p) for p in metric_files})
    curve = curves(*(json.loads(p.read_text()) for p in metric_files))

    result = {"steps": table, "curves": curve, "inputs_sha256": inputs}
    (args.out_dir / "compare.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    lines = ["# A1 vs A2 paired comparison (Mistral judge)", ""]
    for step in STEPS:
        t = table[str(step)]
        lines += [f"## step {step}", "",
                  "| benchmark | rows | A1 % | A2 % | A2-A1 | both correct | only A1 | only A2 | both wrong | sign p |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in list(t["benchmarks"].items()) + [("union", t["union"])]:
            lines.append(f"| {name} | {r['rows']} | {r['acc_A1']:.2f} | {r['acc_A2']:.2f} | {r['diff_A2_minus_A1']:+.2f} "
                         f"| {r['both_correct']} | {r['only_A1']} | {r['only_A2']} | {r['both_wrong']} "
                         f"| {r['sign_test_p']:.4g} |")
        a = t["avg6"]
        lines += ["", f"Avg6: A1 {a['A1']:.2f}, A2 {a['A2']:.2f}, A2-A1 {a['diff_A2_minus_A1']:+.2f}", ""]
    lines += ["## validation curves (average of the four in-training benchmarks)", "",
              "| step | A1 | A2 | A2-A1 |", "|---|---|---|---|"]
    for step in curve["steps_both"]:
        lines.append(f"| {step} | {curve['A1'][step]['average']:.4f} | {curve['A2'][step]['average']:.4f} "
                     f"| {curve['average_diff_A2_minus_A1'][step]:+.4f} |")
    lines += ["", f"max |A2-A1| of the average: {curve['max_abs_average_diff']} at step {curve['max_abs_average_diff_step']}"]
    (args.out_dir / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
