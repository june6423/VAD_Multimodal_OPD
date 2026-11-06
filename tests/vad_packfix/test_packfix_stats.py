"""Tests of the PACKFIX-001 analysis scripts (metrics.py, val_acc.py, stats.py, projection.py) on synthetic arm
directories with hand-computed values."""

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "vad_packfix"


def run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True)


def write_arm(d: Path, kl: float, accs: list[int], with_score: bool, steps: int = 2):
    d.mkdir(parents=True)
    # Two file-logger runs: run1 has step 1, run2 has steps 1 (repeated, later wins) and 2.
    base = {"actor/grad_norm": 1.0, "actor/entropy": 0.5, "actor/pg_loss": 0.01, "actor/vopd_loss": 0.01,
            "response_length/mean": 80.0, "timing_s/step": 100.0, "timing_s/save_checkpoint": 20.0}
    if with_score:
        base["critic/score/mean"] = 0.4
    (d / "metrics_run1_100.jsonl").write_text(
        json.dumps({"step": 0, "data": {"timing_s/testing": 30.0}}) + "\n"
        + json.dumps({"step": 1, "data": {**base, "rollout_corr/kl": 999.0, "timing_s/testing": 10.0}}) + "\n")
    lines = [json.dumps({"step": s, "data": {**base, "rollout_corr/kl": kl + s}}) for s in range(1, steps + 1)]
    (d / "metrics_run2_200.jsonl").write_text("\n".join(lines) + "\n")
    (d / "console.log").write_text("step:1 - actor/entropy:0.5\n")
    (d / "rollouts").mkdir()
    for s in range(1, steps + 1):
        (d / "rollouts" / f"{s}.jsonl").write_text("\n".join(json.dumps({"score": v}) for v in (1.0, 0.0, 0.0, 1.0)) + "\n")
    (d / "validation").mkdir()
    for s in (0, steps):
        rows = [{"input": f"q{i}", "gts": "A", "score": float(a), "acc": float(a), "judge_source": "rule",
                 "benchmark": "vstar"} for i, a in enumerate(accs)]
        (d / "validation" / f"{s}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    (d / "RUN_PACKFIX.txt").write_text("train run1 exit 0 wall_seconds 1000 latest 1\n")


def test_metrics_later_run_wins_and_reward_source(tmp_path):
    write_arm(tmp_path / "F", kl=0.1, accs=[1, 0], with_score=False)
    r = run("metrics.py", tmp_path / "F", 2)
    assert r.returncode == 0, r.stdout + r.stderr
    sm = json.loads((tmp_path / "F" / "STEP_METRICS.json").read_text())
    assert sm["steps"]["1"]["rollout_corr/kl"]["value"] == 1.1  # run2 replaces 999.0
    assert sm["steps"]["1"]["reward_mean"] == {"value": 0.5, "source": "rollouts/1.jsonl score", "rows": 4}
    assert sm["steps"]["0"]["timing_s/testing"]["value"] == 30.0


def test_metrics_reward_from_file_logger_and_missing_metric_fails(tmp_path):
    write_arm(tmp_path / "F", kl=0.1, accs=[1, 0], with_score=True)
    assert run("metrics.py", tmp_path / "F", 2).returncode == 0
    sm = json.loads((tmp_path / "F" / "STEP_METRICS.json").read_text())
    assert sm["steps"]["2"]["reward_mean"]["source"] == "file:critic/score/mean"
    assert run("metrics.py", tmp_path / "F", 3).returncode == 1  # step 3 has no metrics


def test_val_acc_and_stats(tmp_path):
    write_arm(tmp_path / "F", kl=0.1, accs=[1, 1, 1, 0, 1, 1], with_score=False)
    write_arm(tmp_path / "B", kl=0.2, accs=[0, 0, 1, 0, 1, 0], with_score=False)
    for arm in ("F", "B"):
        assert run("metrics.py", tmp_path / arm, 2).returncode == 0
        assert run("val_acc.py", tmp_path / arm).returncode == 0
    a1 = tmp_path / "a1.json"
    a1.write_text(json.dumps({"steps": {"1": {"rollout_corr/kl": 0.14}}}))
    r = run("stats.py", tmp_path / "F", tmp_path / "B", a1, tmp_path / "S.json")
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "S.json").read_text())
    assert math.isclose(s["per_step"]["1"]["rollout_corr/kl"]["F_minus_B"], -0.1)
    v = s["validation"]["2"]
    assert (v["n_F_better"], v["n_B_better"], v["paired_rows"]) == (3, 0, 6)
    assert math.isclose(v["sign_test_p"], 0.25)  # 2 x (1/2)^3
    assert math.isclose(v["F_minus_B_points"], 100 * (5 / 6 - 2 / 6))
    assert s["reward_means"]["F"]["steps_1_10"]["mean"] == 0.5
    assert s["reward_means"]["F"]["steps_11_20"]["mean"] is None
    assert s["arm_B_vs_A1"]["1"]["rollout_corr/kl"] == {"B": 1.2, "A1": 0.14}


def test_projection(tmp_path):
    for arm in ("F", "B"):
        write_arm(tmp_path / arm, kl=0.1, accs=[1, 0], with_score=False)
        assert run("metrics.py", tmp_path / arm, 2).returncode == 0
        assert run("val_acc.py", tmp_path / arm).returncode == 0
    r = run("projection.py", tmp_path, 20, tmp_path / "P.json")
    assert r.returncode == 0, r.stderr
    p = json.loads((tmp_path / "P.json").read_text())["arms"]["F"]
    # startup = 1000 - 100 - (30 + 10) - 20 = 840; val full = 30 x 20 / 2 = 300; total = 840 + 2000 + 900 + 40
    assert math.isclose(p["startup_s"], 840.0) and math.isclose(p["full_val_s"], 300.0)
    assert math.isclose(p["total_h"], 3780 / 3600)
