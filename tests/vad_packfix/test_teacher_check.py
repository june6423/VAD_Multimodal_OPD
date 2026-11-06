"""Tests of the pure helpers of scripts/vad_packfix/teacher_check.py (CPU)."""

import json

import torch

from scripts.vad_packfix import teacher_check


def test_first_groups_keeps_whole_groups_in_batch_order():
    uids = ["b", "a", "b", "c", "a", "b", "c"]
    idx, chosen = teacher_check.first_groups(uids, 2)
    assert chosen == ["b", "a"]
    assert idx.tolist() == [0, 1, 2, 4, 5]


def test_compare_sees_a_one_ulp_difference():
    a = {"log_probs": torch.tensor([[-1.0, -2.0]]), "topk_indices": torch.tensor([[[3, 1]]])}
    b = {"log_probs": torch.tensor([[-1.0, torch.nextafter(torch.tensor(-2.0), torch.tensor(0.0)).item()]]),
         "topk_indices": torch.tensor([[[3, 1]]])}
    same = teacher_check.compare(a, a)
    diff = teacher_check.compare(a, b)
    assert same["log_probs"]["equal"] and same["log_probs"]["max_abs_diff"] == 0.0
    assert not diff["log_probs"]["equal"] and diff["log_probs"]["max_abs_diff"] > 0
    assert diff["topk_indices"]["equal"] and "topk_logps" not in diff


def test_main_exit_code_follows_pass(tmp_path):
    log = {"pass": False, "compared": [{"micro_batch": 0, "view": "teacher",
                                        "log_probs": {"equal": False, "max_abs_diff": 1e-3}}]}
    p = tmp_path / "TEACHER_CHECK.json"
    p.write_text(json.dumps(log))
    assert teacher_check.main(str(p)) == 1
    log["pass"] = True
    p.write_text(json.dumps(log))
    assert teacher_check.main(str(p)) == 0
    assert teacher_check.main(str(tmp_path / "missing.json")) == 1
