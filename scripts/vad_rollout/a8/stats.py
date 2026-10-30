"""A8 statistics (HANDOFF-VLM-VAD-ROLLOUT-001-R9, Part A items 5 to 9).

The driver streams per-trajectory gradients. It never stores them. This module gives:
- draw_trials, largest_n and trial_plan: the across-prompt trial design (two correct and two incorrect groups of
  size n, disjoint, at most one sibling per prompt in a group; 64 trials at n=4, 8 at n=16, 2 at the largest n).
- dot and trial_scalars: inner products of gradient vectors kept as lists of tensors (one tensor per parameter),
  accumulated in float64.
- summarize_trials: within-population cosines, cross cosine, noise-corrected cross cosine, the C-OPSD inversion at
  n=4 (primary), the predicted c_n at the other n, the least-squares fit (secondary), and the mixture angles.
- prompt_scalars and summarize_prompts: the within-prompt pairwise cosines from sums of unit vectors. For unit
  vectors u_i, ||sum_i u_i||^2 = m + 2 * sum_{i<j} cos(u_i, u_j), so a class needs one sum, not m stored vectors.
  Classes are fine classes such as C_f1 (correct, teacher flag 1); merged classes come from the Gram by linearity.
- summarize_cross, summarize_teacher and reading_rule_checks: Part A items 7 to 9.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import torch

GROUPS = ("C1", "C2", "I1", "I2")
CROSS_PAIRS = (("C1", "I1"), ("C1", "I2"), ("C2", "I1"), ("C2", "I2"))
TRIAL_SIZES = ((4, 64), (16, 8))  # plus 2 trials at the largest n
LARGEST_N_TRIALS = 2
FLAGS = ("f1", "f0")  # teacher flag 1 or 0; "fx" = no answer letter, kept in the pooled statistics only


def dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    """Inner product of two gradient vectors given as lists of tensors, accumulated in float64."""
    return float(sum(torch.sum(x.double() * y.double()) for x, y in zip(a, b)))


def _population_max_n(counts: Iterable[int]) -> int:
    # Each prompt gives at most one trajectory to each of the two groups, so at most min(k, 2) in total.
    return sum(min(k, 2) for k in counts) // 2


def largest_n(pool: Sequence[dict]) -> int:
    """Largest n with two disjoint correct groups and two disjoint incorrect groups of size n."""
    per_prompt = {True: defaultdict(int), False: defaultdict(int)}
    for row in pool:
        per_prompt[bool(row["correct"])][row["prompt"]] += 1
    return min(_population_max_n(per_prompt[True].values()), _population_max_n(per_prompt[False].values()))


def _two_groups(rows: list[dict], n: int, rng: random.Random) -> tuple[list, list]:
    """Two disjoint groups of size n from rows, at most one sibling per prompt in each group."""
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt"]].append(row["id"])
    prompts = list(by_prompt)
    rng.shuffle(prompts)
    for ids in by_prompt.values():
        rng.shuffle(ids)
    # Prompts with two or more members give one member to each group. Single-member prompts alternate.
    first, second, singles = [], [], []
    for prompt in prompts:
        ids = by_prompt[prompt]
        if len(ids) >= 2:
            first.append(ids[0])
            second.append(ids[1])
        else:
            singles.append(ids[0])
    for i, traj in enumerate(singles):
        (first if i % 2 == 0 else second).append(traj)
    if len(first) < n or len(second) < n:
        raise ValueError(f"pool too small for two disjoint groups of size {n}")
    # Keep a random subset of each group. A prompt stays in at most one position per group.
    return rng.sample(first, n), rng.sample(second, n)


def draw_trials(pool: Sequence[dict], n: int, count: int, seed: int) -> list[dict]:
    """Draw `count` trials of size n. pool rows have keys id, prompt, correct."""
    rng = random.Random(seed)
    correct = [row for row in pool if row["correct"]]
    incorrect = [row for row in pool if not row["correct"]]
    trials = []
    for _ in range(count):
        c1, c2 = _two_groups(correct, n, rng)
        i1, i2 = _two_groups(incorrect, n, rng)
        trials.append({"n": n, "C1": c1, "C2": c2, "I1": i1, "I2": i2})
    return trials


def trial_plan(pool: Sequence[dict], seed: int) -> list[dict]:
    """Trials of one pool: 64 at n=4, 8 at n=16, 2 at the largest n.

    A size above the largest n gets trials None ("not possible"). The largest-n entry is left out when the
    largest n is 4 or 16, because those sizes are already in the plan.
    """
    n_max = largest_n(pool)
    sizes = list(TRIAL_SIZES)
    if n_max >= 1 and n_max not in (n for n, _ in TRIAL_SIZES):
        sizes.append((n_max, LARGEST_N_TRIALS))
    plan = []
    for i, (n, count) in enumerate(sizes):
        trials = draw_trials(pool, n, count, seed + i) if n <= n_max else None
        plan.append({"n": n, "count": count, "largest_n": n_max, "trials": trials})
    return plan


def trial_scalars(sums: dict[str, Sequence[torch.Tensor]], n: int) -> dict:
    """Squared norms and inner products of the four group sums of one trial."""
    out = {"n": n}
    for g in GROUPS:
        out[f"sq_{g}"] = dot(sums[g], sums[g])
    out["dot_C1_C2"] = dot(sums["C1"], sums["C2"])
    out["dot_I1_I2"] = dot(sums["I1"], sums["I2"])
    for a, b in CROSS_PAIRS:
        out[f"dot_{a}_{b}"] = dot(sums[a], sums[b])
    return out


def _cos(dot_ab: float, sq_a: float, sq_b: float) -> float:
    return dot_ab / math.sqrt(sq_a * sq_b) if sq_a > 0 and sq_b > 0 else float("nan")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _sqrt(value: float) -> float:
    return math.sqrt(value) if value >= 0 else float("nan")


def _describe(values: list[float]) -> dict:
    values = sorted(v for v in values if not math.isnan(v))
    return {"count": len(values), "mean": _mean(values), "median": values[len(values) // 2] if values else float("nan")}


def _fit(points: dict[int, float]) -> tuple[float, float]:
    """Least squares of E||G_n||^2 / n = n ||mu||^2 + sigma^2 over the per-n means. Returns (||mu||^2, sigma^2)."""
    xs = sorted(points)
    ys = [points[x] / x for x in xs]
    if len(xs) < 2:
        return float("nan"), float("nan")
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def summarize_trials(trials: Sequence[dict], correct_fraction: float, norms: dict[str, list[float]]) -> dict:
    """Across-prompt statistics from per-trial scalars (output of trial_scalars).

    norms: the individual gradient norms of the pool by class, {"C": [...], "I": [...]}.
    Primary (C-OPSD section 4.3): per class, q = c_4 / (4 - 3 c_4) with c_4 the mean within-population cosine at
    n=4, ||mu||^2 = q E||g||^2, sigma^2 = (1 - q) E||g||^2, and the predicted c_n = n q / (1 + (n - 1) q) at the
    other n. Secondary: the least-squares fit of mean ||G_n||^2 / n on n.
    Mixture (m = p mu_C + (1 - p) mu_I): theta from the noise-corrected cross cosine at the largest n, norms from
    the primary inversion. phi = angle(m, mu_I) = atan2(p |mu_C| sin theta, (1 - p) |mu_I| + p |mu_C| cos theta).
    """
    by_n = defaultdict(list)
    for t in trials:
        by_n[t["n"]].append(t)
    result = {"per_n": {}, "correct_fraction": correct_fraction}
    sq_means = {"C": {}, "I": {}}
    for n, group in sorted(by_n.items()):
        c_c = [_cos(t["dot_C1_C2"], t["sq_C1"], t["sq_C2"]) for t in group]
        c_i = [_cos(t["dot_I1_I2"], t["sq_I1"], t["sq_I2"]) for t in group]
        cross = [_mean([_cos(t[f"dot_{a}_{b}"], t[f"sq_{a}"], t[f"sq_{b}"]) for a, b in CROSS_PAIRS]) for t in group]
        cc, ci, cx = _mean(c_c), _mean(c_i), _mean(cross)
        corrected = cx / math.sqrt(cc * ci) if cc > 0 and ci > 0 else float("nan")
        result["per_n"][n] = {"trials": len(group), "cos_correct": cc, "cos_incorrect": ci, "cos_cross": cx,
                              "cos_cross_corrected": corrected,
                              "q_from_cos_correct": cc / (n - (n - 1) * cc), "q_from_cos_incorrect": ci / (n - (n - 1) * ci)}
        sq_means["C"][n] = _mean([t["sq_C1"] for t in group] + [t["sq_C2"] for t in group])
        sq_means["I"][n] = _mean([t["sq_I1"] for t in group] + [t["sq_I2"] for t in group])

    result["norms"] = {pop: _describe(norms[pop]) for pop in ("C", "I")}
    primary, secondary = {}, {}
    for pop, key in (("C", "cos_correct"), ("I", "cos_incorrect")):
        e_sq = _mean([x * x for x in norms[pop]])
        c4 = result["per_n"][4][key] if 4 in result["per_n"] else float("nan")
        q = c4 / (4 - 3 * c4)
        mu_sq, sigma_sq = q * e_sq, (1 - q) * e_sq
        primary[pop] = {
            "c_4": c4, "q": q, "mean_sq_norm": e_sq, "mu_sq": mu_sq, "sigma_sq": sigma_sq,
            "snr_1": _sqrt(mu_sq) / _sqrt(sigma_sq) if sigma_sq > 0 else float("nan"),
            "c_n_check": {n: {"predicted": n * q / (1 + (n - 1) * q), "measured": v[key]}
                          for n, v in result["per_n"].items() if n != 4},
        }
        ols_mu_sq, ols_sigma_sq = _fit(sq_means[pop])
        ols_c4 = ols_mu_sq / (ols_mu_sq + ols_sigma_sq / 4)
        secondary[pop] = {"mu_sq": ols_mu_sq, "sigma_sq": ols_sigma_sq, "c_4": ols_c4, "q": ols_c4 / (4 - 3 * ols_c4),
                          "sq_mean_by_n": sq_means[pop]}
    result["primary_correct"], result["primary_incorrect"] = primary["C"], primary["I"]
    result["secondary_ols_correct"], result["secondary_ols_incorrect"] = secondary["C"], secondary["I"]
    result["q_ratio_incorrect_over_correct"] = primary["I"]["q"] / primary["C"]["q"]

    p = correct_fraction
    a_c, a_i = _sqrt(primary["C"]["mu_sq"]), _sqrt(primary["I"]["mu_sq"])
    n_top = max(result["per_n"]) if result["per_n"] else None
    cos_theta = result["per_n"][n_top]["cos_cross_corrected"] if n_top else float("nan")
    theta = math.acos(max(-1.0, min(1.0, cos_theta))) if not math.isnan(cos_theta) else float("nan")
    phi = math.atan2(p * a_c * math.sin(theta), (1 - p) * a_i + p * a_c * math.cos(theta))
    angle_c = math.atan2((1 - p) * a_i * math.sin(theta), p * a_c + (1 - p) * a_i * math.cos(theta))
    result["mixture"] = {
        "n": n_top, "cos_mu_correct_mu_incorrect": cos_theta, "theta_deg": math.degrees(theta),
        "phi_angle_mixture_incorrect_deg": math.degrees(phi), "angle_mixture_correct_deg": math.degrees(angle_c),
    }
    return result


def prompt_scalars(unit_sums: dict[str, Sequence[torch.Tensor]], counts: dict[str, int]) -> dict:
    """Gram matrix of the per-class sums of unit gradient vectors of one prompt (classes such as C_f1, I_f0)."""
    keys = sorted(k for k in unit_sums if counts.get(k, 0) > 0)
    gram = {f"{a}|{b}": dot(unit_sums[a], unit_sums[b]) for i, a in enumerate(keys) for b in keys[i:]}
    return {"counts": {k: counts[k] for k in keys}, "gram": gram}


def _gram(record: dict, a: str, b: str) -> float:
    return record["gram"].get(f"{a}|{b}", record["gram"].get(f"{b}|{a}", 0.0))


def merge_classes(record: dict, select: dict[str, Sequence[str]]) -> dict:
    """Record of merged classes, by linearity of the Gram: select maps a merged class to its fine classes."""
    keys = sorted(select)
    counts = {k: sum(record["counts"].get(f, 0) for f in select[k]) for k in keys}
    gram = {f"{a}|{b}": sum(_gram(record, x, y) for x in select[a] for y in select[b])
            for i, a in enumerate(keys) for b in keys[i:]}
    return {"counts": counts, "gram": gram}


def pair_mean(record: dict, a: str, b: str) -> float:
    """Mean pairwise cosine between classes a and b of one prompt (a == b gives the within-class pairs)."""
    ca, cb = record["counts"].get(a, 0), record["counts"].get(b, 0)
    if a == b:
        if ca < 2:
            return float("nan")
        return (_gram(record, a, a) - ca) / (ca * (ca - 1))
    if ca == 0 or cb == 0:
        return float("nan")
    return _gram(record, a, b) / (ca * cb)


def prompt_class(counts_c: int, counts_i: int) -> str:
    """Sibling class of a prompt from its correct and incorrect counts."""
    if counts_c >= 2 and counts_i >= 2:
        return "mixed"
    if counts_i == 0:
        return "all_correct"
    if counts_c == 0:
        return "all_wrong"
    return "other"


def _fine(record: dict, cls: str) -> list[str]:
    return [k for k in record["counts"] if k.startswith(cls + "_")]


def summarize_prompts(records: Sequence[dict]) -> dict:
    """Within-prompt means over prompts: mixed (>= 2 correct and >= 2 incorrect), all-correct, all-wrong.

    The sibling class of a prompt comes from all its siblings. The flag split keeps the class and uses only the
    pairs of trajectories with that teacher flag.
    """
    def avg(rows, a, b):
        values = [pair_mean(r, a, b) for r in rows]
        values = [v for v in values if not math.isnan(v)]
        return {"prompts": len(values), "mean": _mean(values)}

    def block(rows_by_class):
        mixed, all_correct, all_wrong = (rows_by_class[k] for k in ("mixed", "all_correct", "all_wrong"))
        out = {
            "mixed": {"cc": avg(mixed, "C", "C"), "ii": avg(mixed, "I", "I"), "ci": avg(mixed, "C", "I")},
            "all_correct": {"cc": avg(all_correct, "C", "C")},
            "all_wrong": {"ii": avg(all_wrong, "I", "I")},
        }
        qc, qi = out["mixed"]["cc"]["mean"], out["mixed"]["ii"]["mean"]
        out["mixed"]["q_ratio_incorrect_over_correct"] = qi / qc if qc and not math.isnan(qc) else float("nan")
        return out

    views = {"pooled": lambda r: {"C": _fine(r, "C"), "I": _fine(r, "I")}}
    for f in FLAGS:
        views[f"flag_{f}"] = lambda r, f=f: {"C": [f"C_{f}"], "I": [f"I_{f}"]}
    result = {}
    for name, select in views.items():
        rows_by_class = defaultdict(list)
        for r in records:
            pooled = merge_classes(r, views["pooled"](r))
            cls = prompt_class(pooled["counts"]["C"], pooled["counts"]["I"])
            rows_by_class[cls].append(merge_classes(r, select(r)))
        result[name] = block(rows_by_class)
        result[name]["prompt_counts"] = {k: len(v) for k, v in rows_by_class.items()}
    return result


def summarize_cross(rows: Sequence[dict]) -> dict:
    """Cross-objective cosines of each trajectory, split by correctness, sibling class, and teacher flag.

    rows: {"correct": bool, "prompt_class": str, "flag": str, "cos": {"O1|O2": value, ...}}.
    """
    pairs = sorted({p for r in rows for p in r["cos"]})
    splits: dict[str, Callable[[dict], str]] = {
        "all": lambda r: "all",
        "correctness": lambda r: "C" if r["correct"] else "I",
        "prompt_class": lambda r: r["prompt_class"],
        "correctness_prompt_class": lambda r: ("C" if r["correct"] else "I") + "|" + r["prompt_class"],
        "flag": lambda r: r["flag"],
        "correctness_flag": lambda r: ("C" if r["correct"] else "I") + "|" + r["flag"],
    }
    out = {}
    for name, key in splits.items():
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        out[name] = {k: {p: _describe([r["cos"][p] for r in g if p in r["cos"]]) for p in pairs}
                     for k, g in sorted(groups.items())}
    return out


def summarize_teacher(rows: Sequence[dict]) -> dict:
    """Teacher check counts and log-odds gaps by correctness and flag. rows: {"correct", "flag", "gap"}."""
    groups = defaultdict(list)
    for r in rows:
        groups[("C" if r["correct"] else "I") + "|" + r["flag"]].append(r["gap"])
    return {k: _describe([g for g in v if g is not None]) | {"trajectories": len(v)} for k, v in sorted(groups.items())}


def reading_rule_checks(trials_summary: dict, prompts_summary: dict) -> dict:
    """The comparisons of Part A item 9. The report states the outcome; this function only compares."""
    q_c, q_i = trials_summary["primary_correct"]["q"], trials_summary["primary_incorrect"]["q"]
    mixed = prompts_summary["pooled"]["mixed"]
    w_c, w_i = mixed["cc"]["mean"], mixed["ii"]["mean"]
    across, within = trials_summary["q_ratio_incorrect_over_correct"], mixed["q_ratio_incorrect_over_correct"]
    return {
        "across_ratio": across, "within_ratio": within,
        "across_ratio_ge_2.0": across >= 2.0, "within_ratio_ge_2.0": within >= 2.0, "within_ratio_lt_1.5": within < 1.5,
        "across_q_correct_gt_q_incorrect": q_c > q_i, "within_q_correct_gt_q_incorrect": w_c > w_i,
    }
