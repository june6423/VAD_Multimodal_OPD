"""G3 coverage study (DIRECTIVE-VLM-VAD-GRAD-001-002 D7.3): synthetic draws through accum and stats.

For each draw and each quantity: is the true value inside value +/- 2 SE (and 3 SE)? One-fold quantities are scored
with the sd/2 rule and with the jackknife; cross-fold quantities with the jackknife. Also the Monte Carlo mean of each
estimate against the truth (bias in units of the Monte Carlo standard error). m_all is scored against
m + same_pair_share x sigma_b^2 (the predicted bias).
Usage: python g3_coverage.py <out json> [draws] [dimension]
"""

from __future__ import annotations

import json
import math
import sys
import time

import stats
import synthetic

ONE_FOLD = ("m", "E2", "q", "sigma_b2", "sigma_w2", "ICC", "s_same", "m_all")


def main():
    out_path = sys.argv[1]
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    truth = synthetic.make_truth(d=dim)
    rows = {}
    start = time.time()
    for s in range(draws):
        out = stats.compute(synthetic.draw(truth, seed=s))
        for metric in ("raw", "D"):
            for cls in ("C", "I"):
                true = synthetic.true_values(truth, cls, metric)
                got = out["metrics"][metric]["subsets"][cls]
                for key in ONE_FOLD:
                    t = true["m"] + got["same_pair_share"]["value"] * true["sigma_b2"] if key == "m_all" else true[key]
                    rows.setdefault(f"{metric}|{cls}|{key}", []).append(
                        (got[key]["value"], t, got[key]["se"], got[key]["jackknife_se"]))
            cross = out["metrics"][metric]["cross_fold"]
            for key, t in synthetic.true_cross(truth, metric).items():
                rows.setdefault(f"{metric}|cross|{key}", []).append((cross[key]["value"], t, cross[key]["se"], None))
    result = {"draws": draws, "dimension": dim, "seconds": round(time.time() - start, 1), "quantities": {}}
    for key, vals in rows.items():
        n = len(vals)
        est = [v for v, _, _, _ in vals]
        mean = sum(est) / n
        mc_se = math.sqrt(sum((x - mean) ** 2 for x in est) / (n - 1) / n)
        true = sum(t for _, t, _, _ in vals) / n
        entry = {"true_mean": true, "estimate_mean": mean, "bias_in_mc_se": (mean - true) / mc_se if mc_se else None,
                 "sd_of_estimates": mc_se * math.sqrt(n)}
        rules = {"jackknife": 2} if "|cross|" in key else {"sd/2": 2, "jackknife": 3}  # column of the SE in a row
        for rule, col in rules.items():
            for k in (2, 3):
                hit = sum(1 for row in vals if abs(row[0] - row[1]) <= k * row[col])
                entry[f"coverage_{k}se_{rule}"] = hit / n
            entry[f"mean_se_{rule}"] = sum(row[col] for row in vals) / n
        result["quantities"][key] = entry
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    for key, e in result["quantities"].items():
        cov = " ".join(f"{k}={v:.3f}" for k, v in e.items() if k.startswith("coverage"))
        print(f"{key}: bias {e['bias_in_mc_se']:.2f} mc-se, {cov}")


if __name__ == "__main__":
    main()
