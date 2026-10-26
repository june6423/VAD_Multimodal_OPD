"""Rollout redundancy tables for VLM-VAD-ROLLOUT-001 (HANDOFF-VLM-VAD-ROLLOUT-001-R1, Part A item 8, gate G6).

Usage: python3 scripts/vad_rollout/analyze_rollouts.py <rollout dir> <tokstats dir> <out dir>

Inputs: the rollout dump <step>.jsonl (fields input, output, gts, step, uid, row_index) and the token
statistics step_<step>_rank_<rank>.jsonl (uid, row_index, response_length, r_norm, rho, jsd, anchor_jsd).
Outputs: <out dir>/tables.json and <out dir>/tables.md, with the SHA-256 of every input file. CPU only.

Definitions. A prompt is the group of rows of one step file with one uid. n is the group size.
1. Answer letter: extract_first_option(rule_judge_one(...)["extracted_answer"]) of
   verl/utils/reward_score/opd_benchmark_judge.py, with benchmark zwz_rl_vqa_bbox_teacher (the training
   data source). A row is correct when rule_judge_one returns judge "Yes". A row without a letter is a
   no-letter row and counts as incorrect.
2. all_same: every row of the prompt has a letter and all letters are equal. all_correct: every row is
   correct. all_wrong: no row is correct. mixed: at least one correct and one incorrect row.
   all_correct + all_wrong + mixed = 1.
3. answer_entropy_bits: Shannon entropy (base 2) of the letters of a prompt, no-letter as one category.
4. pass_at_n: fraction of prompts with at least one correct row. mean_at_n: mean of the per-prompt
   fraction of correct rows.
5. distinct_1, distinct_2: per prompt, unique n-grams / all n-grams over the outputs of the prompt
   (whitespace tokens), then the mean over prompts. self_bleu: per row, BLEU-4 against the other rows of
   the prompt (uniform weights, brevity penalty, a zero n-gram count replaced by 0.1), mean over rows.
6. response_words_mean and _max: whitespace tokens of the output. repetition_fraction: per row,
   1 - unique 4-grams / all 4-grams (0 below 4 tokens), mean over rows.
7. Token statistics: per row, the mean of each vector over the valid response tokens, and r_norm at the
   last valid token. For each, the mean over rows, the within-prompt variance (pooled, MSW), the
   between-prompt variance component ((MSB - MSW) / k0, floor 0), and for jsd the intra-class correlation
   ICC(1) = between / (between + within). Prompts with n < 2 give no within-prompt values.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "zwz_rl_vqa_bbox_teacher"
VECTORS = ("r_norm", "rho", "jsd", "anchor_jsd")


def load_judge():
    path = ROOT / "verl/utils/reward_score/opd_benchmark_judge.py"
    spec = importlib.util.spec_from_file_location("opd_benchmark_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct(outputs: list[list[str]], n: int):
    grams = [g for tokens in outputs for g in ngrams(tokens, n)]
    return len(set(grams)) / len(grams) if grams else None


def bleu4(hypothesis: list[str], references: list[list[str]]) -> float:
    if not hypothesis or not references:
        return 0.0
    log_precision = 0.0
    for n in range(1, 5):
        hyp = Counter(ngrams(hypothesis, n))
        total = sum(hyp.values())
        if total == 0:
            return 0.0
        best = Counter()
        for ref in references:
            for gram, count in Counter(ngrams(ref, n)).items():
                best[gram] = max(best[gram], count)
        clipped = sum(min(count, best[gram]) for gram, count in hyp.items())
        log_precision += math.log((clipped or 0.1) / total) / 4
    ref_len = min((len(r) for r in references), key=lambda length: (abs(length - len(hypothesis)), length))
    brevity = 1.0 if len(hypothesis) > ref_len else math.exp(1 - ref_len / len(hypothesis))
    return brevity * math.exp(log_precision)


def repetition(tokens: list[str]) -> float:
    grams = ngrams(tokens, 4)
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def entropy_bits(values: list) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def variance_components(groups: list[list[float]]) -> dict:
    """One-way ANOVA: pooled within-prompt variance, between-prompt component, ICC(1)."""
    groups = [g for g in groups if g]
    total_n = sum(len(g) for g in groups)
    result = {"prompts": len(groups), "rows": total_n, "mean": mean([v for g in groups for v in g]),
              "within_var": None, "between_var": None, "icc": None}
    if len(groups) < 2 or total_n <= len(groups):
        return result
    grand = result["mean"]
    ss_within = sum((v - sum(g) / len(g)) ** 2 for g in groups for v in g)
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ms_within = ss_within / (total_n - len(groups))
    ms_between = ss_between / (len(groups) - 1)
    k0 = (total_n - sum(len(g) ** 2 for g in groups) / total_n) / (len(groups) - 1)
    between = max((ms_between - ms_within) / k0, 0.0)
    result.update(within_var=ms_within, between_var=between,
                  icc=between / (between + ms_within) if between + ms_within > 0 else None)
    return result


def prompt_metrics(rows: list[dict]) -> dict:
    letters = [row["letter"] or None for row in rows]
    correct = [row["correct"] for row in rows]
    tokens = [row["output"].split() for row in rows]
    n = len(rows)
    return {
        "n": n,
        "all_same": all(letters) and len(set(letters)) == 1,
        "all_correct": all(correct),
        "all_wrong": not any(correct),
        "mixed": any(correct) and not all(correct),
        "entropy": entropy_bits(["NONE" if x is None else x for x in letters]),
        "pass": any(correct),
        "mean_correct": sum(correct) / n,
        "no_letter": sum(x is None for x in letters),
        "distinct_1": distinct(tokens, 1),
        "distinct_2": distinct(tokens, 2),
        "self_bleu": [bleu4(t, tokens[:i] + tokens[i + 1 :]) for i, t in enumerate(tokens)] if n > 1 else [],
        "words": [len(t) for t in tokens],
        "repetition": [repetition(t) for t in tokens],
    }


def summarize(prompts: list[dict], stats_groups: dict[str, list[list[float]]]) -> dict:
    count = len(prompts)
    words = [w for p in prompts for w in p["words"]]
    table = {
        "prompts": count,
        "rows": sum(p["n"] for p in prompts),
        "n_values": sorted({p["n"] for p in prompts}),
        "all_same": sum(p["all_same"] for p in prompts) / count if count else None,
        "all_correct": sum(p["all_correct"] for p in prompts) / count if count else None,
        "all_wrong": sum(p["all_wrong"] for p in prompts) / count if count else None,
        "mixed": sum(p["mixed"] for p in prompts) / count if count else None,
        "answer_entropy_bits": mean([p["entropy"] for p in prompts]),
        "pass_at_n": sum(p["pass"] for p in prompts) / count if count else None,
        "mean_at_n": mean([p["mean_correct"] for p in prompts]),
        "no_letter": sum(p["no_letter"] for p in prompts),
        "distinct_1": mean([p["distinct_1"] for p in prompts]),
        "distinct_2": mean([p["distinct_2"] for p in prompts]),
        "self_bleu": mean([b for p in prompts for b in p["self_bleu"]]),
        "response_words_mean": mean(words),
        "response_words_max": max(words) if words else None,
        "repetition_fraction": mean([r for p in prompts for r in p["repetition"]]),
    }
    table["tokstats"] = {name: variance_components(groups) for name, groups in stats_groups.items()}
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("tokstats_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    judge = load_judge()

    step_files = sorted((p for p in args.rollout_dir.glob("*.jsonl") if p.stem.isdigit()), key=lambda p: int(p.stem))
    stat_files = sorted(args.tokstats_dir.glob("step_*_rank_*.jsonl")) if args.tokstats_dir.is_dir() else []
    inputs = {str(p): sha256_file(p) for p in step_files + stat_files}

    stats = {}
    for path in stat_files:
        step = int(re.match(r"step_(\d+)_rank_\d+$", path.stem).group(1))
        for record in read_jsonl(path):
            stats[(step, int(record["row_index"]))] = record
    join = {"dump_rows": 0, "tokstats_rows": len(stats), "matched": 0, "uid_mismatch": 0, "missing_tokstats": 0}

    per_step, all_prompts = {}, []
    all_groups = {f"{name}_mean": [] for name in VECTORS} | {"r_norm_last": []}
    for path in step_files:
        step = int(path.stem)
        by_uid = defaultdict(list)
        step_groups = defaultdict(lambda: defaultdict(list))
        for row in read_jsonl(path):
            join["dump_rows"] += 1
            result = judge.rule_judge_one(BENCHMARK, row.get("input", ""), row.get("gts"), row.get("output", ""))
            letter = judge.extract_first_option(result["extracted_answer"])
            by_uid[row["uid"]].append({"letter": letter, "correct": result.get("judge") == "Yes",
                                       "output": row.get("output", "")})
            record = stats.get((step, int(row["row_index"])))
            if record is None:
                join["missing_tokstats"] += 1
                continue
            if record["uid"] != row["uid"]:
                join["uid_mismatch"] += 1
                continue
            join["matched"] += 1
            for name in VECTORS:
                if record[name]:
                    step_groups[f"{name}_mean"][row["uid"]].append(sum(record[name]) / len(record[name]))
            if record["r_norm"]:
                step_groups["r_norm_last"][row["uid"]].append(record["r_norm"][-1])
        prompts = [prompt_metrics(rows) for rows in by_uid.values()]
        groups = {name: list(step_groups[name].values()) for name in all_groups}
        per_step[step] = summarize(prompts, groups)
        all_prompts.extend(prompts)
        for name in all_groups:
            all_groups[name].extend(groups[name])

    tables = {"inputs": inputs, "join": join, "per_step": per_step, "all_steps": summarize(all_prompts, all_groups)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables.json").write_text(json.dumps(tables, indent=1, sort_keys=True) + "\n")

    columns = ["prompts", "all_same", "all_correct", "all_wrong", "mixed", "answer_entropy_bits", "pass_at_n",
               "mean_at_n", "no_letter", "distinct_1", "distinct_2", "self_bleu", "response_words_mean",
               "response_words_max", "repetition_fraction"]

    def cell(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["# Rollout tables", "", f"Join: {json.dumps(join)}", "",
             "| step | " + " | ".join(columns) + " | jsd ICC |", "|" + "---|" * (len(columns) + 2)]
    for label, table in [*per_step.items(), ("all", tables["all_steps"])]:
        icc = table["tokstats"]["jsd_mean"]["icc"]
        lines.append(f"| {label} | " + " | ".join(cell(table[c]) for c in columns) + f" | {cell(icc)} |")
    lines += ["", "## Token statistics over all steps", "", "| quantity | rows | mean | within_var | between_var | icc |",
              "|---|---|---|---|---|---|"]
    for name, values in tables["all_steps"]["tokstats"].items():
        lines.append(f"| {name} | {values['rows']} | {cell(values['mean'])} | {cell(values['within_var'])} | "
                     f"{cell(values['between_var'])} | {cell(values['icc'])} |")
    (args.out_dir / "tables.md").write_text("\n".join(lines) + "\n")
    print("steps", len(per_step), "join", json.dumps(join))
    return 0


if __name__ == "__main__":
    sys.exit(main())
