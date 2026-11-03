"""moments.GramMemory (DIRECTIVE-VLM-VAD-GRAD-001-005 Decision 7): samples of RssAnon, RssFile and the cgroup anon and
file values while a stage runs, and one sample at the stop."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vad_grad"))
moments = pytest.importorskip("moments")


def test_samples_during_the_stage_and_at_the_stop():
    mem = moments.GramMemory(every=0.05)
    mem.start()
    time.sleep(0.3)
    rec = mem.stop()
    assert len(rec["samples"]) >= 3 and rec["every_s"] == 0.05
    for k in ("rss_anon_gb", "rss_file_gb", "cgroup_anon_gb", "cgroup_file_gb"):
        assert rec[f"max_{k}"] == max(s[k] for s in rec["samples"])
        assert all(math.isfinite(s[k]) and s[k] >= 0 for s in rec["samples"])
    assert rec["max_rss_anon_gb"] > 0
