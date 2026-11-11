"""heldout.py select and subset (Definition 7) on synthetic MME-RealWorld items."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import heldout  # noqa: E402


def query(n: int, bad: bool = False) -> str:
    opts = "\n".join(f"({chr(65 + i)}) option {i}" for i in range(n))
    if bad:
        opts = opts.replace("(B)", "(Q)")
    return f"Question?\nThe choices are listed below:\n{opts}\nSelect the best answer.\nThe best answer is: "


def items(count: int = 1500) -> list[dict]:
    out = []
    for i in range(count):
        n = 5 if i % 2 else 6
        out.append({"index": 10000 + i, "images": [f"/data/img/{i:06d}.jpg"], "query": query(n, bad=i % 7 == 0),
                    "response": "C" if i % 11 else "Z"})
    return out


def test_option_letters():
    assert heldout.option_letters(query(5)) == ["A", "B", "C", "D", "E"]
    assert heldout.option_letters(query(5, bad=True)) is None
    assert heldout.option_letters("no options") is None


def test_select_order_replacement_and_sets():
    its = items()
    sel = heldout.select(its)
    kept = sel["items"]
    assert len(kept) == 1000
    order = list(range(len(its)))
    random.Random(heldout.SEED).shuffle(order)
    valid = [its[i] for i in order if heldout.option_letters(its[i]["query"]) and
             its[i]["response"] in heldout.option_letters(its[i]["query"])]
    assert [it["index"] for it in kept] == [it["index"] for it in valid[:1000]]
    last = order.index(kept[-1]["index"] - 10000)
    bad_before = [its[i] for i in order[:last] if its[i] not in valid]
    assert [r["index"] for r in sel["replaced"]] == [it["index"] for it in bad_before]
    a = [it for it in kept if it["set"] == "A"]
    assert len(a) == 500 and [sum(it["part"] == h for it in a) for h in range(1, 5)] == [125] * 4
    assert [it["part"] for it in a] == sorted(it["part"] for it in a)
    assert all(it["set"] == "B" and it["part"] == 0 for it in kept[500:])
    assert all(isinstance(it["image"], str) and it["image"].startswith("/data/img/") for it in kept)


def test_select_rejects_two_images():
    its = items()
    its = [{**it, "images": it["images"] * 2} for it in its]
    with pytest.raises(SystemExit):
        heldout.select(its)


def test_subset():
    sel = heldout.select(items())
    sub = heldout.subset(sel, 2, 10, 10)
    a = [it for it in sub["items"] if it["set"] == "A"]
    b = [it for it in sub["items"] if it["set"] == "B"]
    assert [it["part"] for it in a] == [1] * 10 + [2] * 10 and len(b) == 10
    full_a = [it for it in sel["items"] if it["set"] == "A"]
    assert [it["index"] for it in a[:10]] == [it["index"] for it in full_a if it["part"] == 1][:10]
    assert [it["index"] for it in b] == [it["index"] for it in sel["items"] if it["set"] == "B"][:10]
