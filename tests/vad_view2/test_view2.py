"""Tests of scripts/vad_view2 (HANDOFF-VLM-VAD-VIEW-002-R1 Part A item 8)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vad_view2"))
import g2check  # noqa: E402
import hook  # noqa: E402
import pilot_rows  # noqa: E402
import stats  # noqa: E402
import viewmap  # noqa: E402
import views  # noqa: E402

from verl.protocol import DataProto  # noqa: E402


def test_view_names_follow_definition_2():
    assert hook.VIEW_NAMES == ("p0.00-z1.5", "p0.00-z2.0", "p0.00-z3.0", "p1-z1.5", "p1-z2.0", "p1-z3.0",
                               "p0.30-z2.0", "p0.30-z3.0")
    assert hook.VIEW_NAMES[hook.V0_SLOT - 1] == views.V0
    assert set(hook.VIEW_NAMES) == {views.view_name(lab, z) for lab, z in views.VIEW_KEYS}


def test_generation_n():
    assert hook.generation_n("B", 8) == 8
    assert hook.generation_n("T8", 8) == 8
    assert hook.generation_n("N1V8", 8) == 1
    assert hook.generation_n("N1R", 1) == 1
    for a, n in (("B", 1), ("T8", 1), ("N1V8", 1), ("N1R", 8)):
        with pytest.raises(ValueError):
            hook.generation_n(a, n)


def test_view_slots_b_t8_n1v8():
    assert hook.view_slots("B", 768, 3) == [5] * 768
    for a in ("T8", "N1V8"):
        s = hook.view_slots(a, 768, 3)
        assert s[:16] == [1, 2, 3, 4, 5, 6, 7, 8] * 2
        assert Counter(s) == {k: 96 for k in range(1, 9)}
        with pytest.raises(ValueError):
            hook.view_slots(a, 12, 1)


def test_view_slots_n1r_draw():
    s1 = hook.view_slots("N1R", 96, 1)
    assert s1 == hook.view_slots("N1R", 96, 1)
    assert s1 != hook.view_slots("N1R", 96, 2)
    assert s1 == [int(x) + 1 for x in np.random.default_rng([20261113, 1]).integers(0, 8, 96)]
    assert s1 == g2check.rule("N1R", 96, 1)
    allc = Counter(x for step in range(1, 41) for x in hook.view_slots("N1R", 96, step))
    assert set(allc) == set(range(1, 9)) and sum(allc.values()) == 3840
    assert all(380 < c < 580 for c in allc.values())


def test_g2_rule_equals_hook_rule():
    for a, rows in (("B", 768), ("T8", 768), ("N1V8", 768), ("N1R", 96)):
        for step in (1, 7):
            assert g2check.rule(a, rows, step) == hook.view_slots(a, rows, step)


def _gen_output(prompts: int) -> DataProto:
    return DataProto.from_dict(
        tensors={"responses": torch.arange(prompts * 3).view(prompts, 3),
                 "rollout_log_probs": torch.arange(prompts * 3, dtype=torch.float32).view(prompts, 3) / 10},
        non_tensors={"request_id": np.array([f"r{i}" for i in range(prompts)], dtype=object)},
        meta_info={"x": 1})


def test_copy_responses_n1v8():
    g = _gen_output(4)
    c = hook.copy_responses(g, "N1V8")
    assert len(c) == 32
    for i in range(4):
        for k in range(8):
            assert torch.equal(c.batch["responses"][8 * i + k], g.batch["responses"][i])
            assert torch.equal(c.batch["rollout_log_probs"][8 * i + k], g.batch["rollout_log_probs"][i])
            assert c.non_tensor_batch["request_id"][8 * i + k] == f"r{i}"
    for a in ("B", "T8", "N1R"):
        assert hook.copy_responses(g, a) is g


def test_n1v8_copies_keep_prompt_uid():
    """The trainer path: batch.repeat(8, interleave) of the prompts, union with the copied responses."""
    prompts = DataProto.from_dict(tensors={"input_ids": torch.arange(4).view(4, 1)},
                                  non_tensors={"uid": np.array(["u0", "u1", "u2", "u3"], dtype=object)})
    batch = prompts.repeat(repeat_times=8, interleave=True).union(hook.copy_responses(_gen_output(4), "N1V8"))
    for j in range(32):
        assert batch.non_tensor_batch["uid"][j] == f"u{j // 8}"
        assert batch.non_tensor_batch["request_id"][j] == f"r{j // 8}"


def _batch(prompts: int, n: int, step_keys: list[str]) -> DataProto:
    uids = np.array([f"u{i}" for i in range(prompts) for _ in range(n)], dtype=object)
    items = np.empty(prompts * n, dtype=object)
    for j in range(prompts * n):
        items[j] = np.array([{"path": step_keys[j // n]}], dtype=object)
    return DataProto.from_dict(tensors={"x": torch.zeros(prompts * n, 1)},
                               non_tensors={"uid": uids, "bbox_images": items})


def _vmap(tmp_path: Path, keys: list[str]) -> Path:
    rows = {}
    for k in keys:
        paths = []
        for s in range(1, 9):
            if s == hook.V0_SLOT:
                p = k
            else:
                p = str(tmp_path / f"{Path(k).stem}_v{s}.png")
            Path(p).write_bytes(f"{p}".encode())
            paths.append(p)
        rows[k] = paths
    f = tmp_path / "VIEWMAP.json"
    f.write_text(json.dumps({"views": list(hook.VIEW_NAMES), "rows": rows}))
    return f


@pytest.mark.parametrize("arm,n", [("B", 8), ("T8", 8), ("N1V8", 8), ("N1R", 1)])
def test_assign_views_and_g2check(tmp_path, monkeypatch, arm, n):
    keys = [str(tmp_path / f"teacher{i}.png") for i in range(24)]
    vm = _vmap(tmp_path, keys)
    monkeypatch.setenv("VAD_VIEW_ARM", arm)
    monkeypatch.setenv("VAD_VIEW_MAP", str(vm))
    root = tmp_path / "root"
    monkeypatch.setenv("VAD_VIEW_LOG_DIR", str(root / arm / "views_log"))
    monkeypatch.setattr(hook, "_viewmap", None)
    batch = _batch(24, n, keys)
    original = batch.non_tensor_batch["bbox_images"]
    hook.assign_views(batch, 1)
    slots = hook.view_slots(arm, 24 * n, 1)
    if arm == "B":
        assert batch.non_tensor_batch["bbox_images"] is original
        assert "vad_view" not in batch.non_tensor_batch
    else:
        assert list(batch.non_tensor_batch["vad_view"]) == slots
        for j, s in enumerate(slots):
            got = hook.row_key(batch.non_tensor_batch["bbox_images"][j])
            key = keys[j // n]
            assert got == (key if s == hook.V0_SLOT else str(tmp_path / f"{Path(key).stem}_v{s}.png"))
            if s == hook.V0_SLOT:
                assert batch.non_tensor_batch["bbox_images"][j] is original[j]
    assert hook.dump_views(batch) == slots
    # the order changes before the teacher builder (_balance_batch); the record must follow the rows
    perm = list(reversed(range(len(batch))))
    batch.reorder(torch.tensor(perm))
    hook.log_teacher_images(batch, 1)
    teacher = [json.loads(x) for x in open(root / arm / "views_log" / "teacher_1.jsonl")]
    assert [t["slot"] for t in teacher] == [slots[p] for p in perm]
    res = g2check.check_arm(arm, root / arm, json.loads(vm.read_text()), 0)
    assert res["fail_count"] == 0 and res["checked_prompts"] == 20


def test_g2check_detects_wrong_image(tmp_path, monkeypatch):
    keys = [str(tmp_path / f"teacher{i}.png") for i in range(24)]
    vm = _vmap(tmp_path, keys)
    monkeypatch.setenv("VAD_VIEW_ARM", "T8")
    monkeypatch.setenv("VAD_VIEW_MAP", str(vm))
    root = tmp_path / "root"
    monkeypatch.setenv("VAD_VIEW_LOG_DIR", str(root / "T8" / "views_log"))
    monkeypatch.setattr(hook, "_viewmap", None)
    batch = _batch(24, 8, keys)
    hook.assign_views(batch, 1)
    for j in range(3, 24 * 8, 8):  # slot 4 of each prompt gets a wrong image
        batch.non_tensor_batch["bbox_images"][j] = np.array([{"path": keys[0]}], dtype=object)
    hook.log_teacher_images(batch, 1)
    res = g2check.check_arm("T8", root / "T8", json.loads(vm.read_text()), 0)
    assert res["checked_prompts"] == 20
    assert {f[1] for f in res["fails"]} == {"path", "sha256"}
    assert res["fail_count"] == 40


def test_pilot_epoch_has_two_steps():
    """TASK-VLM-VAD-VIEW-002-002: with 16 rows the resume after step 1 is at the epoch end and run2 has no step."""
    import pandas as pd

    man = [{"row": i, "max_teacher_tokens": 100 + (i == 7) * 900,
            "views": {"p0.30-z3.0": {"size": [10, 10 + (i == 30) * 90]}}} for i in range(60)]
    sel = pilot_rows.choose(man)
    assert len(sel["rows"]) == 16 and {7, 30} <= set(sel["rows"])
    df = pd.DataFrame({"row": range(60)})
    p = pilot_rows.pilot_frame(df, sel["rows"])
    assert len(p) // 16 == 2 and len(p) % 16 == 0
    assert list(p["row"]) == sel["rows"] * 2


def test_build_map():
    names = hook.VIEW_NAMES
    teacher = ["/t/a.png", "/t/b.png"]
    man = [{"row": i, "views": {nm: {"path": (teacher[i] if nm == views.V0 else f"/v/{i}/{nm}.png")} for nm in names}}
           for i in range(2)]
    rows, problems = viewmap.build_map(teacher, man)
    assert problems == []
    assert rows["/t/a.png"][hook.V0_SLOT - 1] == "/t/a.png"
    assert rows["/t/b.png"][0] == "/v/1/p0.00-z1.5.png"
    man[1]["views"][views.V0]["path"] = "/t/other.png"
    _, problems = viewmap.build_map(teacher, man)
    assert len(problems) == 1
    _, problems = viewmap.build_map(teacher + ["/t/c.png"], man)
    assert problems and "parquet rows 3" in problems[0]


def test_sign_test_and_letter_first():
    assert stats.sign_test(0, 0) == 1.0
    assert stats.sign_test(5, 5) == 1.0
    assert stats.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert stats.sign_test(0, 10) == pytest.approx(2 / 1024)
    assert stats.sign_test(7, 3) == pytest.approx(2 * (1 + 10 + 45 + 120) / 1024)
    for t in ("A", " B.", "\nC) red", "D", "F\n"):
        assert stats.letter_first(t)
    for t in ("Answer: A", "Apple", "G", "", None, "A1", "<think>A"):
        assert not stats.letter_first(t)


def _write_arm(root: Path, arm: str, acc: list[int], outputs: list[str], keys: list[str]):
    d = root / arm
    (d / "validation").mkdir(parents=True)
    steps = {}
    for s in ("20", "40"):
        steps[s] = {"acc_mean": sum(acc) / len(acc), "rows": len(acc), "row_acc": acc, "row_key": keys}
        (d / "validation" / f"{s}.jsonl").write_text("".join(json.dumps({"output": o}) + "\n" for o in outputs))
        (d / "heldout" / s).mkdir(parents=True)
        (d / "heldout" / s / "HELDOUT_B.json").write_text(json.dumps(
            {"n": 10, "mean_log_R": -1.0, "mean_m": 0.7, "accuracy": 0.6, "extra": 1}))
    (d / "VAL_ACC.json").write_text(json.dumps({"steps": steps}))
    sm = {str(k): {"actor/grad_norm": {"value": 0.1 * k, "source": "log"},
                   "rollout_acc": {"value": 0.5, "source": "dump"}} for k in range(0, 41)}
    (d / "STEP_METRICS.json").write_text(json.dumps({"steps": sm}))


def test_stats_synthetic(tmp_path):
    rows = 40
    keys = [f"k{i}" for i in range(rows)]
    base = [i % 2 for i in range(rows)]
    t8 = [1] * rows  # 20 rows better than B, 0 worse
    accs = {"B": base, "T8": t8, "N1V8": base, "N1R": base}
    outs = {a: ["A"] * (rows // 2) + ["The answer"] * (rows // 2) for a in stats.ARMS}
    outs["T8"] = ["B"] * rows
    for a in stats.ARMS:
        _write_arm(tmp_path, a, accs[a], outs[a], keys)
    (tmp_path / "N1R" / "views_log").mkdir()
    for step in (1, 2, 10):
        (tmp_path / "N1R" / "views_log" / f"assign_{step}.jsonl").write_text(
            "".join(json.dumps({"slot": s}) + "\n" for s in hook.view_slots("N1R", 96, step)))
    out = tmp_path / "STATS.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/vad_view2/stats.py"), str(tmp_path), str(out), "20,40",
                    str(rows)], check=True, capture_output=True)
    r = json.loads(out.read_text())
    c1 = r["comparisons"]["C1"]
    assert c1["40"]["difference_points"] == pytest.approx(50.0)
    assert (c1["40"]["n_x_better"], c1["40"]["n_y_better"]) == (20, 0)
    assert c1["40"]["sign_test_p"] == pytest.approx(2 / 2 ** 20)
    assert c1["40"]["detected_rule1"] is True and "detected_rule1" not in c1["20"]
    assert c1["40"]["letter_first_rate_x_minus_y"] == pytest.approx(0.5)
    c2 = r["comparisons"]["C2"]["40"]
    assert c2["difference_points"] == 0 and c2["sign_test_p"] == 1.0 and c2["detected_rule1"] is False
    assert r["validation"]["B"]["20"]["letter_first_rate"] == pytest.approx(0.5)
    assert r["validation"]["T8"]["40"]["acc"] == 1.0
    assert sorted(r["step_metrics"]["B"], key=int) == [str(k) for k in range(1, 41)]
    assert r["heldout"]["N1V8"]["40"] == {"n": 10, "mean_log_R": -1.0, "mean_m": 0.7, "accuracy": 0.6}
    expect = Counter(s for step in (1, 2, 10) for s in hook.view_slots("N1R", 96, step))
    assert r["n1r_view_counts"]["steps"] == [1, 2, 10]
    assert r["n1r_view_counts"]["counts"] == {str(k): expect.get(k, 0) for k in range(1, 9)}


def test_stats_refuses_unpaired_rows(tmp_path):
    keys = [f"k{i}" for i in range(4)]
    x = {"acc_mean": 0.5, "rows": 4, "row_acc": [1, 0, 1, 0], "row_key": keys}
    y = dict(x, row_key=keys[:3] + ["other"])
    with pytest.raises(SystemExit):
        stats.compare(x, y, 4)
    with pytest.raises(SystemExit):
        stats.compare(x, x, 5)
