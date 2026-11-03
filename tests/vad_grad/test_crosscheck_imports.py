"""crosscheck.py must use stats.compute of scripts/vad_grad/stats.py. moments puts scripts/vad_rollout/a8 first on
sys.path, and a8/grads.py imports the A8 stats.py under the same module name (TASK-VLM-VAD-GRAD-001-011 failure:
"module 'stats' has no attribute 'compute'"). A fresh process, so that no other test module loads stats first."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); import crosscheck; "
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
