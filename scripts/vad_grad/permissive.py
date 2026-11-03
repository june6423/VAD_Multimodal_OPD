"""Permissive answer-letter extractor for the POST_RUN label-agreement table (HANDOFF-VLM-VAD-GRAD-001-R1).

Source: the DT extractor of analysis E, runtime/drafts/dt/e_recovery_check.py lines 13-28 (file sha256
45f333e9fd716de6034d2841a345d4852f137ce68943be3dd717969679134f37), copied without change below.
It is a DT reading tool, not a validated grader. The gradient labels stay the D2 labels of label.py.
A row is permissive-correct when the letter equals the ground-truth letter. A row without a match is incorrect.
"""

import re

PERMISSIVE = [
    r"(?:final\s+answer|correct\s+answer|answer|答案)\s*(?:is|:|：)\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
    r"(?:option|choice)\s*[*_`]*\s*[\(\[]?\s*([A-F])\b",
    r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\]]?\s*[*_`]*\s*$",
    r"^\s*[*_`]*\s*[\(\[]?\s*([A-F])\s*[\)\].:：]",
    r"\*\*\s*[\(\[]?\s*([A-F])\s*[\)\]]?\s*[\.:：]",
]
RX = [re.compile(p, flags=re.IGNORECASE | re.MULTILINE) for p in PERMISSIVE]


def perm(text):
    for i, rx in enumerate(RX):
        ms = list(rx.finditer(text))
        if ms:
            return ms[-1].group(1).upper(), i, ms[-1].start(1)
    return "", None, None


def permissive_label(response_text: str, gt: str) -> int:
    letter = perm(response_text)[0]
    return int(bool(letter) and letter == gt)
