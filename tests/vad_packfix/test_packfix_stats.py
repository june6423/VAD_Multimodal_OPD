"""Tests of the PACKFIX-001 analysis scripts (metrics.py, val_acc.py, stats.py, projection.py, rollout_acc.py) on
synthetic arm directories with hand-computed values."""

import json
import math
import subprocess
import sys
from pathlib import Path

from scripts.vad_packfix import rollout_acc

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "vad_packfix"


def run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True)


def write_arm(d: Path, kl: float, accs: list[int], steps: int = 2, rollout_acc_steps=None):
    d.mkdir(parents=True)
    # Two file-logger runs: run1 has step 1, run2 has steps 1 (repeated, later wins) and 2.
    base = {"actor/grad_norm": 1.0, "actor/entropy": 0.5, "actor/pg_loss": 0.01, "actor/vopd_loss": 0.01,
            "response_length/mean": 80.0, "timing_s/step": 100.0, "timing_s/save_checkpoint": 20.0}
    (d / "metrics_run1_100.jsonl").write_text(
        json.dumps({"step": 1, "data": {**base, "rollout_corr/kl": 999.0}}) + "\n")
    lines = [json.dumps({"step": s, "data": {**base, "rollout_corr/kl": kl + s}}) for s in range(1, steps + 1)]
    (d / "metrics_run2_200.jsonl").write_text("\n".join(lines) + "\n")
    (d / "console.log").write_text("step:1 - actor/entropy:0.5\n")
    acc_steps = range(1, steps + 1) if rollout_acc_steps is None else rollout_acc_steps
    (d / "ROLLOUT_ACC.json").write_text(json.dumps({"steps": {
        str(s): {"rows": 4, "acc_mean": 0.5, "judge_calls": 3, "seconds": 40.0 + s} for s in acc_steps}}))
    (d / "validation").mkdir()
    for s in (0, steps):
        rows = [{"input": f"q{i}", "gts": "A", "score": float(a), "acc": float(a), "judge_source": "rule",
                 "benchmark": "vstar"} for i, a in enumerate(accs)]
        (d / "validation" / f"{s}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    (d / "RUN_PACKFIX.txt").write_text(
        "train run1 exit 0 wall_seconds 1000 latest 2\n"
        "judge ready 1 port 5 at 2026-11-07T01:00:00Z judge_start_seconds 80\n"
        "validate val0 exit 0 wall_seconds 400 dump x\n"
        f"validate val{steps} exit 0 wall_seconds 500 dump x\n")
    hm = "(WorkerDict pid=7) VAD_HOSTMEM pid=7 copy {} seconds 1.00 gib 16.91 utc {}\n"
    log = "(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step {} utc {}\n"
    (d / "console_val0.log").write_text(hm.format("offload_model", "10:00:00") + hm.format("load_model", "10:05:00")
                                        + hm.format("offload_model", "10:05:10") + log.format(0, "10:06:50"))
    (d / f"console_val{steps}.log").write_text(hm.format("offload_model", "11:00:00")
                                               + hm.format("offload_model", "11:06:00") + log.format(steps, "11:08:00"))


def test_metrics_later_run_wins_and_rollout_acc(tmp_path):
    write_arm(tmp_path / "F", kl=0.1, accs=[1, 0])
    r = run("metrics.py", tmp_path / "F", 2)
    assert r.returncode == 0, r.stdout + r.stderr
    sm = json.loads((tmp_path / "F" / "STEP_METRICS.json").read_text())
    assert sm["steps"]["1"]["rollout_corr/kl"]["value"] == 1.1  # run2 replaces 999.0
    assert sm["steps"]["1"]["rollout_acc"] == {"value": 0.5, "source": "ROLLOUT_ACC.json", "rows": 4}


def test_metrics_missing_rollout_acc_or_metric_fails(tmp_path):
    write_arm(tmp_path / "F", kl=0.1, accs=[1, 0], rollout_acc_steps=[1])
    r = run("metrics.py", tmp_path / "F", 2)
    assert r.returncode == 1
    assert [2, "rollout_acc"] in json.loads(r.stdout)["missing"]
    write_arm(tmp_path / "G", kl=0.1, accs=[1, 0])
    assert run("metrics.py", tmp_path / "G", 3).returncode == 1  # step 3 has no metrics


def test_val_acc_and_stats(tmp_path):
    write_arm(tmp_path / "F", kl=0.1, accs=[1, 1, 1, 0, 1, 1])
    write_arm(tmp_path / "B", kl=0.2, accs=[0, 0, 1, 0, 1, 0])
    for arm in ("F", "B"):
        assert run("metrics.py", tmp_path / arm, 2).returncode == 0
        assert run("val_acc.py", tmp_path / arm).returncode == 0
    a1 = tmp_path / "a1.json"
    a1.write_text(json.dumps({"steps": {"1": {"rollout_corr/kl": 0.14}}}))
    r = run("stats.py", tmp_path / "F", tmp_path / "B", a1, tmp_path / "S.json")
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "S.json").read_text())
    assert math.isclose(s["per_step"]["1"]["rollout_corr/kl"]["F_minus_B"], -0.1)
    assert s["per_step"]["1"]["rollout_acc"]["F_minus_B"] == 0.0
    v = s["validation"]["2"]
    assert (v["n_F_better"], v["n_B_better"], v["paired_rows"]) == (3, 0, 6)
    assert math.isclose(v["sign_test_p"], 0.25)  # 2 x (1/2)^3
    assert math.isclose(v["F_minus_B_points"], 100 * (5 / 6 - 2 / 6))
    assert s["rollout_acc_means"]["F"]["steps_1_10"] == {"mean": 0.5, "steps": [1, 2], "sources": ["ROLLOUT_ACC.json"]}
    assert s["rollout_acc_means"]["F"]["steps_11_20"]["mean"] is None
    assert s["arm_B_vs_A1"]["1"]["rollout_corr/kl"] == {"B": 1.2, "A1": 0.14}
    assert s["arm_B_vs_A1"]["1"]["rollout_acc"] == {"B": 0.5, "A1": None}


def test_projection(tmp_path):
    for arm in ("F", "B"):
        write_arm(tmp_path / arm, kl=0.1, accs=[1, 0])
        assert run("metrics.py", tmp_path / arm, 2).returncode == 0
        assert run("val_acc.py", tmp_path / arm).returncode == 0
    # Arm F validated step 0 on all rows in 4 processes. Arm B copied that dump.
    with (tmp_path / "F" / "RUN_PACKFIX.txt").open("a") as f:
        f.writelines(f"validate val0_p{k} exit 0 wall_seconds {300 + 10 * k} dump x\n" for k in range(4))
    r = run("projection.py", tmp_path, 2, tmp_path / "P.json")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "P.json").read_text())
    p, b = res["arms"]["F"], res["arms"]["B"]
    # startup = 1000 - 100 - 100 = 800. Full validation = 300 + 310 + 320 + 330 = 1260.
    # F: 800 + 20 x 100 + 80 + 3 x 1260 + 20 x 42 = 7500. B: also 3 validations (DIRECTIVE-009 action 10), 7500.
    assert math.isclose(p["startup_s"], 800.0) and math.isclose(p["judge_start_s"], 80.0)
    assert math.isclose(res["val_full"]["full_s"], 1260.0) and res["val_full"]["rows"] == 2
    assert p["ran_val0"] and not b["ran_val0"]
    assert math.isclose(p["rollout_score_step_s"], 42.0) and p["judge_calls_per_step"] == 3
    assert math.isclose(p["total_h"], 7500 / 3600) and math.isclose(b["total_h"], 7500 / 3600)
    # The step-0 dump must have the full validation rows.
    assert run("projection.py", tmp_path, 2636, tmp_path / "P2.json").returncode != 0


def test_rollout_acc_scores_each_step_once_with_the_reward_inputs(tmp_path):
    (tmp_path / "rollouts").mkdir()
    for s in (1, 2):
        rows = [{"input": "p", "output": f"ans{i}", "gts": "A", "score": 0.0, "data_source": "zwz",
                 "extra_info": {"question": f"q{i}"}} for i in range(4)]
        (tmp_path / "rollouts" / f"{s}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    calls = []

    def compute(data_sources, solution_strs, ground_truths, extra_infos):
        calls.append((list(data_sources), list(solution_strs), list(ground_truths), list(extra_infos)))
        return [{"score": float(i % 2), "judge_source": "llm" if i < 3 else "rule"} for i in range(len(solution_strs))]

    assert rollout_acc.main(tmp_path, 2, compute) == 0
    res = json.loads((tmp_path / "ROLLOUT_ACC.json").read_text())
    assert calls[0] == (["zwz"] * 4, ["ans0", "ans1", "ans2", "ans3"], ["A"] * 4, [{"question": f"q{i}"} for i in range(4)])
    st = res["steps"]["1"]
    assert (st["rows"], st["acc_mean"], st["judge_calls"], st["row_score"]) == (4, 0.5, 3, [0.0, 1.0, 0.0, 1.0])
    assert res["check"]["pass"] and res["reward_kwargs"]["judge_max_tokens"] == 2048
    assert rollout_acc.main(tmp_path, 2, compute) == 0 and len(calls) == 2  # a scored step is not scored again


def test_rollout_acc_fails_on_a_dump_without_the_reward_inputs(tmp_path):
    (tmp_path / "rollouts").mkdir()
    (tmp_path / "rollouts" / "1.jsonl").write_text(json.dumps({"input": "p", "output": "a", "gts": "A"}) + "\n")
    assert rollout_acc.main(tmp_path, 1, lambda *a: []) == 1
    assert "lacking ['data_source', 'extra_info']" in json.loads(
        (tmp_path / "ROLLOUT_ACC.json").read_text())["check"]["problems"][0][1]
