"""Tests of pilot.py select and project (HANDOFF-VLM-VAD-GRAD-003-R2 Part A item 9) on synthetic inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

DIR = Path(__file__).resolve().parents[2] / "scripts" / "vad_grad3"
sys.path.insert(0, str(DIR))

import batches  # noqa: E402
import pilot  # noqa: E402


def synthetic(n_prompts: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    rollouts, labels = [], {}
    for p in range(n_prompts):
        for s in range(8):
            t = 8 * p + s
            rollouts.append({"traj": t, "prompt_pos": p, "sibling": s, "dataset_index": p,
                             "prompt_len": int(rng.integers(500, 3000)), "response_ids": [0] * int(rng.integers(2, 300))})
            labels[t] = int(rng.integers(0, 2))
    return rollouts, labels


def test_select_takes_two_folds_the_longest_and_a_mixed_prompt():
    rollouts, labels = synthetic()
    sel = pilot.select(rollouts, labels)
    folds = {p: f + 1 for p, f in batches.fold_of(list(range(40))).items()}
    assert len(sel["prompts"]) == 12 and sel["folds"][0] == 1
    assert sorted({folds[p] for p in sel["prompts"]}) == sorted(sel["folds"])
    assert all(sum(folds[p] == f for p in sel["prompts"]) == 6 for f in sel["folds"])
    longest = max(range(40), key=lambda p: max(len(r["response_ids"]) for r in rollouts if r["prompt_pos"] == p))
    assert sel["longest_response_prompt"] == longest and longest in sel["prompts"]
    assert sel["mixed_prompts"]
    batches_, summary = batches.build(rollouts, labels, sel["prompts"])
    assert {k for k in batches_ if k.startswith("B8")} == {f"B8-f{f}" for f in sel["folds"]}
    assert summary["batches"]["B8-f1"]["rows"] == 48 and "R8-f1-s1" in batches_


def test_project_scales_the_pilot_times(tmp_path):
    rollouts, labels = synthetic()
    d = tmp_path / "M0"
    (d / "rows_full").mkdir(parents=True)
    full, _ = batches.build(rollouts, labels)
    for name, recs in full.items():
        (d / "rows_full" / f"ROWS_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    micro = lambda tok, s: {"tokens": tok, "seconds_inputs": 0.0, "seconds_forward": s, "seconds_backward": s,  # noqa: E731
                            "seconds_rounding": s / 2}
    for tag, n in (("O__B8-f1", 6), ("O__B1-f1-j0", 3), ("O__R8-f1-s1", 1), ("Fx__B8-f1", 1)):
        (d / f"progress_{tag}.json").write_text(json.dumps({"micro": [micro(1000, 1.0)], "sq_norm": {}}))
    results = [{"key": t, "seconds_end": 10.0 * n, "sq_norm": {str(k): 1.0 for k in range(n)}}
               for t, n in (("O__B8-f1", 6), ("O__B1-f1-j0", 3), ("O__R8-f1-s1", 1), ("Fx__B8-f1", 1))]
    (d / "runs.jsonl").write_text(json.dumps({"load_seconds": 50.0, "results": results}) + "\n")
    (tmp_path / "TIMES.jsonl").write_text(json.dumps({"stage": "displacement", "seconds": 600}) + "\n" +
                                          json.dumps({"stage": "gram_M0", "seconds": 150}) + "\n")
    res = pilot.project(tmp_path, ["M0"])
    ck = res["checkpoints"]["M0"]
    tok = lambda name: sum(r["total_tokens"] for r in full[name])  # noqa: E731
    b8 = sum(tok(f"B8-f{f}") for f in range(1, 5))
    assert ck["compute_seconds"]["B8"] == pytest.approx(2.5 / 1000 * b8)
    assert ck["compute_seconds"]["Fx"] == pytest.approx(2.5 / 1000 * 4 * tok("B8-f1"))
    assert ck["rounding_seconds"]["B8"] == pytest.approx(0.5 / 1000 * b8)
    assert res["rounding_hours"] == pytest.approx(sum(ck["rounding_seconds"].values()) / 3600)
    assert ck["save_seconds"]["B1"] == pytest.approx(10.0 * 96)
    assert ck["gram_seconds"] == pytest.approx(150 * 132 / (11 + 4))
    assert ck["load_seconds"] == 100.0
    assert res["total_hours"] == pytest.approx((600 + ck["total_seconds"]) / 3600)
    assert res["peak_disk_tb"] == pytest.approx((128 + 7 + 4) * 4 * 3569090560 / 1e12)


def test_mixed_b1_finds_a_micro_batch_with_both_labels(tmp_path):
    """DIRECTIVE-VLM-VAD-GRAD-003-005: the pilot needs a B1 micro-batch with >= 2 rows of mixed labels."""
    row = lambda rank, micro, label: {"rank": rank, "mini": 0, "micro": micro, "label": label}  # noqa: E731
    path = tmp_path / "ROWS_B1-f1-j0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in
                            (row(0, 0, 1), row(1, 0, 0), row(1, 0, 0), row(2, 1, 0), row(2, 1, 1))))
    assert pilot.mixed_b1(path) == [[2, 0, 1]]
    path.write_text("".join(json.dumps(r) + "\n" for r in (row(0, 0, 1), row(1, 0, 0), row(1, 0, 0))))
    assert pilot.mixed_b1(path) == []
