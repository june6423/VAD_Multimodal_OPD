"""permissive.py: pattern order, last match of the first matching pattern, and the label rule."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_grad"))
from permissive import perm, permissive_label  # noqa: E402


def test_first_matching_pattern_wins():
    text = "B\nThe final answer is (C)."
    assert perm(text)[:2] == ("C", 0)


def test_last_match_of_the_pattern():
    assert perm("Option A looks wrong. Choice d is right.")[:2] == ("D", 1)


def test_line_patterns_and_bold():
    assert perm("Reasoning...\n**(b)**")[:2] == ("B", 2)
    assert perm("Reasoning...\nE. the red car")[:2] == ("E", 3)
    assert perm("It is **A.** because")[:2] == ("A", 4)


def test_no_match_is_incorrect():
    assert perm("I cannot tell.") == ("", None, None)
    assert permissive_label("I cannot tell.", "") == 0
    assert permissive_label("Answer: B", "B") == 1
    assert permissive_label("Answer: B", "C") == 0
