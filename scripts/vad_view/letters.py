"""Letter probabilities of HANDOFF-VLM-VAD-VIEW-001-R1 Definition 6.

option_letters: the option letters of a prompt text (lines "X. text", X in A to F), in letter order.
fresh_ids: the token id of each letter as one token (tokenizer.encode(letter)), the form of a first response token.
prefix_ids: at the answer-letter position of a student response (the teacher-check rule of TASK-VLM-VAD-ROLLOUT-001-056,
  scripts/vad_grad3/objectives.py answer_letter_char, letter_token, gt_token_like), the token of each option letter in
  the form of the sampled token (gt_token_like with that letter); None when the rule gives no position or a letter has
  no one-token form.
letter_logp: the log probabilities of the letter tokens from one row of full-vocabulary log probabilities.
"""

from __future__ import annotations

import re

OPTION = re.compile(r"(?m)^\s*([A-F])\.\s")


def option_letters(text: str) -> list[str]:
    return sorted(set(OPTION.findall(text)))


def fresh_ids(tokenizer, letters: list[str]) -> list[int]:
    ids = []
    for letter in letters:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f"STOP: letter {letter} is not one token: {enc}")
        ids.append(enc[0])
    return ids


def prefix_ids(obj, tokenizer, response_ids: list[int], response_text: str, letters: list[str]):
    """(position, sampled letter, token ids of the letters) or None. obj: the objectives module of vad_grad3."""
    found = obj.answer_letter_char(response_text)
    if not found:
        return None
    pos = obj.letter_token(tokenizer, response_ids, found[0])
    if pos is None:
        return None
    sampled = response_ids[pos]
    ids = [obj.gt_token_like(tokenizer, sampled, found[1], letter) for letter in letters]
    if any(i is None for i in ids):
        return None
    return pos, found[1], ids


def letter_logp(logp_row, ids: list[int]) -> list[float]:
    return [float(logp_row[i]) for i in ids]
