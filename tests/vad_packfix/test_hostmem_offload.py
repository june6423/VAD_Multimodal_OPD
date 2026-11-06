"""Tests of the DIRECTIVE-VLM-VAD-PACKFIX-001-005 parts of scripts/vad_packfix/hostmem.py (candidates 2, 4 and 5).

Each test runs one check of scripts/vad_packfix/offload_selftest.py. b200 runs the same checks with CUDA
(run_teacher_check.sh), because its env has no pytest.
"""

import json

import pytest

from scripts.vad_packfix import offload_selftest


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    offload_selftest.init_process_group(str(tmp_path_factory.mktemp("pg")))
    yield
    offload_selftest.dist.destroy_process_group()


@pytest.mark.parametrize("name,check", offload_selftest.CHECKS, ids=[n for n, _ in offload_selftest.CHECKS])
def test_offload_selftest_check(name, check):
    check()


def test_selftest_main_writes_a_pass_file(tmp_path, monkeypatch):
    monkeypatch.setattr(offload_selftest, "CHECKS", [("ok", lambda: {}), ("bad", lambda: 1 / 0)])
    offload_selftest.dist.destroy_process_group()  # main() makes its own process group
    try:
        assert offload_selftest.main(str(tmp_path / "s.json")) == 1
    finally:
        offload_selftest.init_process_group(str(tmp_path))
    result = json.loads((tmp_path / "s.json").read_text())
    assert result["ok"]["pass"] and not result["bad"]["pass"] and not result["pass"]
