"""VLM-VAD-ROLLOUT-001 R6 Part A item 2: analyze_rollouts.py on n=1 rollouts (arm A2).

At n=1 the per-prompt group quantities have no meaning. The script must run, and the tables must give the
correctness rate (mean_at_n), the response length, and the per-rollout means of the token statistics.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "analyze_rollouts.py"

# 4 prompts x 1 rollout of step 3: rows 0 and 2 correct, row 1 wrong letter, row 3 no letter.
ROWS = [("uid-1", "A", "Answer: A"), ("uid-2", "B", "I think so. Answer: C"), ("uid-3", "A", "Answer: A"),
        ("uid-4", "D", "I cannot tell from the image")]


def test_n1_tables(tmp_path):
    rollouts, tokstats, out = tmp_path / "rollouts", tmp_path / "tokstats", tmp_path / "out"
    rollouts.mkdir()
    tokstats.mkdir()
    dump, stats = [], []
    for i, (uid, gt, output) in enumerate(ROWS):
        dump.append({"input": "Which cup is left? (A) red (B) blue (C) green (D) none", "output": output,
                     "gts": gt, "score": 0.0, "step": 3, "uid": uid, "row_index": i})
        stats.append({"uid": uid, "row_index": i, "response_length": 2, "r_norm": [1.0, float(i)],
                      "rho": [0.5, 0.5], "jsd": [float(i), float(i) + 2], "anchor_jsd": [0.0, 0.0]})
    (rollouts / "3.jsonl").write_text("\n".join(json.dumps(row) for row in dump) + "\n")
    (tokstats / "step_3_rank_0.jsonl").write_text("\n".join(json.dumps(row) for row in stats) + "\n")

    result = subprocess.run([sys.executable, str(SCRIPT), str(rollouts), str(tokstats), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    step = json.loads((out / "tables.json").read_text())["per_step"]["3"]
    assert step["n_values"] == [1] and step["prompts"] == 4 and step["rows"] == 4
    assert math.isclose(step["mean_at_n"], 0.5)  # correctness rate: rows 0 and 2
    assert step["no_letter"] == 1
    assert math.isclose(step["response_words_mean"], (2 + 5 + 2 + 6) / 4)
    assert step["self_bleu"] is None  # no sibling rows
    jsd = step["tokstats"]["jsd_mean"]
    assert math.isclose(jsd["mean"], (1 + 2 + 3 + 4) / 4)  # per-rollout means 1, 2, 3, 4
    assert jsd["within_var"] is None and jsd["icc"] is None  # no within-prompt values at n=1
    assert math.isclose(step["tokstats"]["r_norm_last"]["mean"], (0 + 1 + 2 + 3) / 4)
