"""VLM-VAD-ROLLOUT-001 gate G6: scripts/vad_rollout/analyze_rollouts.py against hand-computed values."""

import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/vad_rollout/analyze_rollouts.py"

# 3 prompts x 8 siblings of step 5.
# P1 (gt A): 8 x "The red cup is left. Answer: A"         -> all same, all correct, 7 words each.
# P2 (gt B): 4 x "I think so. Answer: C", 4 x "... D"     -> all wrong, not all same, 5 words each.
# P3 (gt A): 3 x "Answer: A", 3 x "Answer: B", 2 x "I cannot tell" (no letter) -> mixed, 2/2/3 words.
PROMPTS = [
    ("uid-1", "A", ["The red cup is left. Answer: A"] * 8),
    ("uid-2", "B", ["I think so. Answer: C"] * 4 + ["I think so. Answer: D"] * 4),
    ("uid-3", "A", ["Answer: A"] * 3 + ["Answer: B"] * 3 + ["I cannot tell"] * 2),
]

# Hand-computed values.
P3_ENTROPY = -(2 * (3 / 8) * math.log2(3 / 8) + (2 / 8) * math.log2(2 / 8))  # A 3, B 3, no letter 2
EXPECTED = {
    "prompts": 3,
    "rows": 24,
    "all_same": 1 / 3,  # P1
    "all_correct": 1 / 3,  # P1
    "all_wrong": 1 / 3,  # P2
    "mixed": 1 / 3,  # P3
    "answer_entropy_bits": (0.0 + 1.0 + P3_ENTROPY) / 3,  # P2: C 4, D 4 -> 1 bit
    "pass_at_n": 2 / 3,  # P1, P3
    "mean_at_n": (1.0 + 0.0 + 3 / 8) / 3,
    "no_letter": 2,
    # distinct-2: P1 6 unique / 48; P2 5 unique / 32; P3 4 unique / 10.
    "distinct_2": (6 / 48 + 5 / 32 + 4 / 10) / 3,
    "response_words_mean": (8 * 7 + 8 * 5 + 3 * 2 + 3 * 2 + 2 * 3) / 24,
    "response_words_max": 7,
}


def test_g6_tables_match_hand_computed_values(tmp_path):
    rollouts, tokstats, out = tmp_path / "rollouts", tmp_path / "tokstats", tmp_path / "out"
    rollouts.mkdir()
    tokstats.mkdir()
    dump, stats = [], []
    for prompt_index, (uid, gt, outputs) in enumerate(PROMPTS):
        for output in outputs:
            row_index = len(dump)
            dump.append({"input": "Which cup is left? (A) red (B) blue (C) green (D) none", "output": output,
                         "gts": gt, "score": 0.0, "step": 5, "uid": uid, "row_index": row_index})
            # jsd constant per prompt (1, 2, 3): within-prompt variance 0, between-prompt component 1, ICC 1.
            stats.append({"uid": uid, "row_index": row_index, "response_length": 3, "r_norm": [0.5, 1.0, 2.0],
                          "rho": [0.1, 0.2, 0.3], "jsd": [float(prompt_index + 1)] * 3, "anchor_jsd": [0.0] * 3})
    (rollouts / "5.jsonl").write_text("\n".join(json.dumps(row) for row in dump) + "\n")
    (tokstats / "step_5_rank_0.jsonl").write_text("\n".join(json.dumps(row) for row in stats) + "\n")

    result = subprocess.run([sys.executable, str(SCRIPT), str(rollouts), str(tokstats), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    tables = json.loads((out / "tables.json").read_text())
    step = tables["per_step"]["5"]
    for key, value in EXPECTED.items():
        assert math.isclose(step[key], value, rel_tol=0, abs_tol=1e-12), (key, step[key], value)
        assert math.isclose(tables["all_steps"][key], value, rel_tol=0, abs_tol=1e-12), key
    assert step["n_values"] == [8]
    assert tables["join"] == {"dump_rows": 24, "tokstats_rows": 24, "matched": 24, "uid_mismatch": 0,
                              "missing_tokstats": 0}
    jsd = step["tokstats"]["jsd_mean"]
    assert (jsd["within_var"], jsd["between_var"], jsd["icc"]) == (0.0, 1.0, 1.0)
    assert step["tokstats"]["r_norm_last"]["mean"] == 2.0
    assert sorted(tables["inputs"]) == sorted([str(rollouts / "5.jsonl"), str(tokstats / "step_5_rank_0.jsonl")])
    assert (out / "tables.md").read_text().startswith("# Rollout tables")
    print("G6 " + " ".join(f"{key}={step[key]:.6g}" for key in EXPECTED) +
          f" jsd_icc={jsd['icc']} r_norm_last_mean={step['tokstats']['r_norm_last']['mean']}")
