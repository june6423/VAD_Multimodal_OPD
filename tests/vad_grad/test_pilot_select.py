"""pilot.select (DIRECTIVE-VLM-VAD-GRAD-001-005 Decisions 5 and 6): the 4 roles in their order, each prompt from a fold
that no earlier chosen prompt uses, else the lowest prompt of the role. 16 prompts, accum.fold_of folds:
0 = {5, 6, 12, 13}, 1 = {1, 8, 10, 14}, 2 = {4, 9, 11, 15}, 3 = {0, 2, 3, 7}."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vad_grad"))
import accum  # noqa: E402
import pilot  # noqa: E402

CORRECT = {"all_correct": 8, "all_wrong": 0, "mixed": 4, "other": 1}


def write(tmp_path: Path, classes: dict[int, str], longest: int) -> tuple[str, str]:
    rows, labels = [], []
    for p in range(16):
        for s in range(8):
            t = p * 8 + s
            rows.append({"traj": t, "prompt_pos": p, "sibling": s, "response_ids": [0] * (900 if p == longest else 10)})
            labels.append({"traj": t, "label": int(s < CORRECT[classes.get(p, "other")])})
    (tmp_path / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    return str(tmp_path / "rollouts.jsonl"), str(tmp_path / "labels.jsonl")


def test_fold_map_of_the_design():
    folds = accum.fold_of(list(range(16)))
    assert {f: sorted(p for p in range(16) if folds[p] == f) for f in range(4)} == {
        0: [5, 6, 12, 13], 1: [1, 8, 10, 14], 2: [4, 9, 11, 15], 3: [0, 2, 3, 7]}


def test_each_prompt_from_a_free_fold(tmp_path):
    # The lowest all_correct prompt (5) shares fold 0 with the longest prompt (13), and the lowest all_wrong prompt (1)
    # shares fold 1 with the chosen all_correct prompt (8).
    classes = {5: "all_correct", 8: "all_correct", 1: "all_wrong", 4: "all_wrong", 0: "mixed", 2: "mixed"}
    chosen = pilot.select(*write(tmp_path, classes, longest=13))
    folds = accum.fold_of(list(range(16)))
    assert chosen == [13, 8, 4, 0]
    assert sorted(folds[p] for p in chosen) == [0, 1, 2, 3]


def test_lowest_prompt_of_the_role_when_no_fold_is_free(tmp_path):
    # Both mixed prompts are in folds that the longest (13, fold 0) and all_correct (8, fold 1) prompts use.
    classes = {8: "all_correct", 4: "all_wrong", 10: "mixed", 6: "mixed"}
    chosen = pilot.select(*write(tmp_path, classes, longest=13))
    assert chosen == [13, 8, 4, 6]
