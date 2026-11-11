"""letters.py: option letters, letter token ids, and the answer-letter position rule of TASK-056."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vad_view"))
sys.path.insert(0, str(ROOT / "scripts" / "vad_grad3"))

import letters  # noqa: E402

PROMPT = ("<image>\nWhat color is the lighting?\n\nOnly focus on the red box.\n\nA. black\nB. white\nC. blue\n"
          "D. purple\n\nAnswer with the option's letter from the given choices.")


class Tok:
    """Character-level fake tokenizer: one token per piece of VOCAB."""

    VOCAB = ["A", "B", "C", "D", " A", " B", " C", " D", "Answer", ":", " is", "The", " answer", "<eos>", "x"]

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for j in sorted(range(len(self.VOCAB)), key=lambda k: -len(self.VOCAB[k])):
                if text.startswith(self.VOCAB[j], i):
                    out.append(j)
                    i += len(self.VOCAB[j])
                    break
            else:
                raise ValueError(text[i:])
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if (skip_special_tokens and self.VOCAB[i] == "<eos>") else self.VOCAB[i] for i in ids)


def test_option_letters():
    assert letters.option_letters(PROMPT) == ["A", "B", "C", "D"]
    assert letters.option_letters("Q?\nA. x\nB. y\nE. z\n") == ["A", "B", "E"]


def test_fresh_ids():
    assert letters.fresh_ids(Tok(), ["A", "B", "C", "D"]) == [0, 1, 2, 3]


def test_prefix_ids():
    pytest.importorskip("verl.utils.reward_score.opd_benchmark_judge")
    import objectives as obj

    tok = Tok()
    text = "The answer is C"
    ids = tok.encode(text)
    got = letters.prefix_ids(obj, tok, ids, text, ["A", "B", "C", "D"])
    assert got is not None
    pos, sampled, lid = got
    assert sampled == "C" and pos == 3 and lid == [4, 5, 6, 7]
    # a bare first-token letter
    got = letters.prefix_ids(obj, tok, [1, 13], "B", ["A", "B", "C", "D"])
    assert got == (0, "B", [0, 1, 2, 3])
    assert letters.prefix_ids(obj, tok, [14], "x", ["A", "B"]) is None


def test_letter_logp():
    assert letters.letter_logp([0.0, -1.0, -2.0, -3.0], [3, 1]) == [-3.0, -1.0]
