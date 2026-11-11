import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import check_pv  # noqa: E402


def test_letter_values():
    lp = [math.log(0.5), math.log(0.2), math.log(0.1)]
    log_r, m, best = check_pv.letter_values(lp, 1)
    assert m == pytest.approx(0.8)
    assert log_r == pytest.approx(math.log(0.2 / 0.8))
    assert best == 0
    assert check_pv.letter_values([-1.0, -1.0], 0)[2] == 0  # first letter at a tie


def write_lines(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_check_p(tmp_path):
    for ck in check_pv.CHECKPOINTS:
        bf = [{"index": 1, "set": "A", "letter_logp": [-1.0, -2.0], "log_R": -0.5, "m": 0.6},
              {"index": 2, "set": "A", "letter_logp": [-2.0, -1.0], "log_R": -0.1, "m": 0.9},
              {"index": 3, "set": "B", "letter_logp": [-1.0, -3.0], "log_R": -0.2, "m": 0.5}]
        fp = [{"index": 1, "set": "A", "letter_logp": [-1.0, -2.0], "log_R": -0.4, "m": 0.6},
              {"index": 2, "set": "A", "letter_logp": [-0.5, -1.0], "log_R": -0.1, "m": 0.7},
              {"index": 3, "set": "B", "letter_logp": [-1.0, -3.0], "log_R": -0.2, "m": 0.5}]
        write_lines(tmp_path / ck / "HELDOUT_VALUES.jsonl", bf)
        write_lines(tmp_path / ck / "linesearch" / "points" / "a0-r1.jsonl", fp)
    check_pv.check_p([str(tmp_path), str(tmp_path / "CHECK_P.json")])
    res = json.loads((tmp_path / "CHECK_P.json").read_text())
    a = res["M0"]["A"]
    assert a["n"] == 2
    assert a["log_R"]["max_abs"] == pytest.approx(0.1)
    assert a["log_R"]["mean_abs"] == pytest.approx(0.05)
    assert a["m"]["max_abs"] == pytest.approx(0.2)
    assert a["argmax_changed"] == 1
    assert res["A1_80"]["B"] == {"n": 1, "log_R": {"mean_abs": 0.0, "median_abs": 0.0, "max_abs": 0.0},
                                 "m": {"mean_abs": 0.0, "median_abs": 0.0, "max_abs": 0.0}, "argmax_changed": 0}


def test_check_p_item_mismatch(tmp_path):
    for ck in check_pv.CHECKPOINTS:
        write_lines(tmp_path / ck / "HELDOUT_VALUES.jsonl",
                    [{"index": 1, "set": "A", "letter_logp": [0.0], "log_R": 0.0, "m": 1.0}])
        write_lines(tmp_path / ck / "linesearch" / "points" / "a0-r1.jsonl",
                    [{"index": 2, "set": "A", "letter_logp": [0.0], "log_R": 0.0, "m": 1.0}])
    with pytest.raises(SystemExit):
        check_pv.check_p([str(tmp_path), str(tmp_path / "CHECK_P.json")])


def test_gen_and_check_v(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("WORKERS", "1")
    src = tmp_path / "views"
    src.mkdir()
    student = tmp_path / "s.png"
    Image.new("RGB", (100, 80), (10, 20, 30)).save(student)
    released = tmp_path / "t.png"
    Image.new("RGB", (48, 40), (10, 20, 30)).save(released)
    v0 = check_pv.views.V0
    man = [{"row": r, "bbox": [30, 20, 50, 40], "student": str(student),
            "views": {v0: {"path": str(released), "size": [48, 40]}}} for r in (0, 1)]
    write_lines(src / "MANIFEST.jsonl", man)
    (src / "REMOVED.json").write_text(json.dumps({"removed": [{"row": 1}]}))
    p1 = tmp_path / "P1_FIT.json"
    p1.write_text(json.dumps({"p1": 0.1}))
    out = tmp_path / "gen"
    check_pv.gen([str(src), str(p1), str(out)])
    gman = [json.loads(x) for x in open(out / "MANIFEST.jsonl")]
    assert [r["row"] for r in gman] == [0]
    g = gman[0]["views"][check_pv.GEN]
    # crop: box 20 x 20, pad 0.1 -> 2 px each side -> 24 x 24, zoom 2 -> 48 x 48
    assert g["crop"] == [28, 18, 52, 42] and g["size"] == [48, 48]
    assert Path(g["path"]).exists()
    assert gman[0]["views"][v0]["path"] == str(released)
    assert json.loads((out / "REMOVED.json").read_text())["removed"] == [{"row": 1}]

    lp = {v0: [-0.1, -3.0], "p1-z1.5": [-2.0, -0.2], "p1-z3.0": [-0.1, -3.0]}
    full = [{"row": 0, "gt": "A", "letters": ["A", "B"], "logp": lp}]
    new = [{"row": 0, "gt": "A", "letters": ["A", "B"], "logp": {v0: [-0.1, -3.0], check_pv.GEN: [-0.3, -2.0]}}]
    write_lines(tmp_path / "FRESH.jsonl", full)
    write_lines(tmp_path / "GEN_FRESH.jsonl", new)
    check_pv.check_v([str(tmp_path / "FRESH.jsonl"), str(tmp_path / "GEN_FRESH.jsonl"), str(out),
                      str(tmp_path / "CHECK_V.json"), str(tmp_path / "SIZES.tsv")])
    res = json.loads((tmp_path / "CHECK_V.json").read_text())
    assert res["rows"] == 1
    assert res["released_v0_repeat"]["mean_abs_log_R"] == 0.0
    assert res["p1-z3.0"]["argmax_changed"] == 0
    assert res["p1-z1.5"]["argmax_changed"] == 1
    ref = check_pv.letter_values(lp[v0], 0)
    gen = check_pv.letter_values([-0.3, -2.0], 0)
    assert res[check_pv.GEN]["mean_abs_log_R"] == pytest.approx(abs(gen[0] - ref[0]))
    assert res[check_pv.GEN]["mean_abs_m"] == pytest.approx(abs(gen[1] - ref[1]))
    assert res["size_different_rows"] == 1
    assert (tmp_path / "SIZES.tsv").read_text().splitlines()[1] == "0\t48\t48\t48\t40"
