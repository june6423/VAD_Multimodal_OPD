"""crosscheck.py must use stats.compute of scripts/vad_grad/stats.py. moments puts scripts/vad_rollout/a8 first on
sys.path, and a8/grads.py imports the A8 stats.py under the same module name (TASK-VLM-VAD-GRAD-001-011 failure:
"module 'stats' has no attribute 'compute'"). A fresh process, so that no other test module loads stats first. The
probe imports moments, then crosscheck, in the order of the run (DIRECTIVE-VLM-VAD-GRAD-001-005 Decision 3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); import moments; import crosscheck; "
    "print(json.dumps({'grad': crosscheck.stats.__file__, 'a8': crosscheck.a8_stats.__file__, "
    "'compute': hasattr(crosscheck.stats, 'compute'), 'draw_trials': hasattr(crosscheck.a8_stats, 'draw_trials')}))"
)


def test_crosscheck_uses_the_grad_stats_module():
    pytest.importorskip("torch")
    res = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "scripts" / "vad_grad")], capture_output=True,
                         text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    rec = json.loads(res.stdout.strip().splitlines()[-1])
    assert Path(rec["grad"]) == ROOT / "scripts" / "vad_grad" / "stats.py"
    assert rec["compute"]
    assert Path(rec["a8"]) == ROOT / "scripts" / "vad_rollout" / "a8" / "stats.py"
    assert rec["draw_trials"]


def test_compare_on_a_synthetic_gram():
    """The compare part of crosscheck.py on stats.compute of a synthetic Gram, with two synthetic trial rows."""
    sys.path.insert(0, str(ROOT / "scripts" / "vad_grad"))
    crosscheck = pytest.importorskip("crosscheck")
    import synthetic

    pred = crosscheck.stats.compute(synthetic.draw(synthetic.make_truth(), seed=0))["metrics"]
    rows = [{f"c4_{c}_{m}": v for c in ("C", "I") for m in ("raw", "D")} for v in (0.2, 0.4)]
    out = crosscheck.compare(rows, pred)
    assert sorted(out) == ["D|C", "D|I", "raw|C", "raw|I"]
    for key, rec in out.items():
        name, cls = key.split("|")
        p = pred[name]["subsets"][cls]["c_4"]
        assert rec["predicted_c4"] == p["value"] and rec["predicted_se"] == p["se"]
        assert rec["trial_mean_c4"] == pytest.approx(0.3) and rec["trial_se"] == pytest.approx(0.1)
        assert rec["difference_in_trial_se"] == pytest.approx((0.3 - p["value"]) / 0.1)
