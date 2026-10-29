"""VLM-VAD-ROLLOUT-001 R6 gate G6: scripts/vad_rollout/compare_arms.py against hand-computed values."""

import json
import subprocess
import sys
from math import comb
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "compare_arms.py"
BENCHMARKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")

# vstar, step 80: 10 rows. A1 correct on rows 0-5, A2 correct on rows 2-8.
# both correct 2-5 (4), only A1 0-1 (2), only A2 6-8 (3), both wrong 9 (1). A1 60 %, A2 70 %.
# Every other benchmark and step 130: 4 rows, both arms correct on rows 0-1 (50 %, no discordant row).
VSTAR_A1 = [i < 6 for i in range(10)]
VSTAR_A2 = [2 <= i <= 8 for i in range(10)]


def write_judge(root: Path, arm: str, step: int, benchmark: str, correct: list) -> None:
    folder = root / f"step_{step}" / "run" / "judge" / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    # Rows in reverse order for A2, so the join must use sample_uid and not the position.
    rows = [{"sample_uid": f"{benchmark}:index:{i}", "judge": "Yes" if c else "No"} for i, c in enumerate(correct)]
    if arm == "A2":
        rows.reverse()
    (folder / f"{arm}_step{step}_seed42_answer.jsonl").write_text(json.dumps(rows))


def write_metrics(root: Path, offset: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    steps = {}
    for step in (1, 10, 20):
        steps[str(step)] = {"timing_s/step": 600.0}
        if step % 10 == 0:
            for j, b in enumerate(BENCHMARKS[:4]):
                steps[str(step)][f"val-core/{b}/acc/mean@1"] = 0.5 + 0.1 * j + offset * step / 10
    (root / "STEP_METRICS.json").write_text(json.dumps(steps))


def test_compare_arms(tmp_path):
    for arm in ("A1", "A2"):
        for step in (80, 130):
            for b in BENCHMARKS:
                if b == "vstar" and step == 80:
                    correct = VSTAR_A1 if arm == "A1" else VSTAR_A2
                else:
                    correct = [True, True, False, False]
                write_judge(tmp_path / f"eval_{arm}", arm, step, b, correct)
    write_metrics(tmp_path / "train_A1", 0.0)
    write_metrics(tmp_path / "train_A2", 0.01)
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "eval_A1"), str(tmp_path / "eval_A2"),
                    str(tmp_path / "train_A1"), str(tmp_path / "train_A2"), str(out)], check=True,
                   capture_output=True)
    result = json.loads((out / "compare.json").read_text())

    vstar = result["steps"]["80"]["benchmarks"]["vstar"]
    assert vstar["rows"] == 10
    assert (vstar["both_correct"], vstar["only_A1"], vstar["only_A2"], vstar["both_wrong"]) == (4, 2, 3, 1)
    assert abs(vstar["acc_A1"] - 60.0) < 1e-9 and abs(vstar["acc_A2"] - 70.0) < 1e-9
    assert abs(vstar["diff_A2_minus_A1"] - 10.0) < 1e-9
    assert abs(vstar["sign_test_p"] - min(1.0, 2 * (comb(5, 0) + comb(5, 1) + comb(5, 2)) / 2**5)) < 1e-12  # 1.0

    # Avg6 at step 80: vstar 60 / 70, five others 50 / 50.
    avg6 = result["steps"]["80"]["avg6"]
    assert abs(avg6["A1"] - (60 + 5 * 50) / 6) < 1e-9 and abs(avg6["A2"] - (70 + 5 * 50) / 6) < 1e-9
    union = result["steps"]["80"]["union"]
    assert union["rows"] == 10 + 5 * 4 and union["only_A1"] == 2 and union["only_A2"] == 3
    step130 = result["steps"]["130"]["union"]
    assert step130["only_A1"] == 0 and step130["only_A2"] == 0 and step130["sign_test_p"] == 1.0

    curves = result["curves"]
    assert curves["steps_both"] == [10, 20]
    # A2 adds 0.01 * step / 10 to each benchmark: 0.01 at step 10, 0.02 at step 20.
    assert abs(curves["max_abs_average_diff"] - 0.02) < 1e-12 and curves["max_abs_average_diff_step"] == 20
    assert len(result["inputs_sha256"]) == 2 * 2 * 6 + 2


def test_sign_test_values():
    sys.path.insert(0, str(SCRIPT.parent))
    from compare_arms import sign_test_p

    assert sign_test_p(0, 0) == 1.0
    assert abs(sign_test_p(0, 6) - 2 / 64) < 1e-12
    assert abs(sign_test_p(1, 9) - 2 * (1 + 10) / 1024) < 1e-12
