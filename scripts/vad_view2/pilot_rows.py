"""The 16 pilot prompts of HANDOFF-VLM-VAD-VIEW-002-R1 Part A item 7.1.

1. The row with the longest teacher input: the largest max_teacher_tokens of the views MANIFEST.jsonl (views.py: text
   tokens of the teacher prompt - 1 + image tokens, the largest over the 8 views).
2. The row with the largest v8 image (p0.30-z3.0): the most pixels (width x height of the manifest).
3. Random rows (random.Random(20261113).sample of the other rows) until 16 distinct rows.
Writes the rows of train_vad.parquet (all columns, parquet order) two times (32 rows) to <out parquet>, and
PILOT_ROWS.json. With train batch 16 the epoch has 2 steps, so the resume after step 1 starts inside the epoch as in the
FULL run. (TASK-VLM-VAD-VIEW-002-002: with 16 rows the step 1 checkpoint is at the epoch end, the restored dataloader
gives no batch, and run2 ends without step 2.)
Usage: python pilot_rows.py <train_vad.parquet> <views dir> <out parquet> <PILOT_ROWS.json>
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

N, SEED = 16, 20261113


def choose(manifest: list[dict]) -> dict:
    longest = max(manifest, key=lambda m: (m["max_teacher_tokens"], -m["row"]))
    v8 = max(manifest, key=lambda m: (m["views"]["p0.30-z3.0"]["size"][0] * m["views"]["p0.30-z3.0"]["size"][1],
                                      -m["row"]))
    fixed = sorted({longest["row"], v8["row"]})
    rest = sorted(m["row"] for m in manifest if m["row"] not in fixed)
    rows = sorted(fixed + random.Random(SEED).sample(rest, N - len(fixed)))
    return {"rows": rows, "longest_teacher_row": longest["row"], "longest_teacher_tokens": longest["max_teacher_tokens"],
            "largest_v8_row": v8["row"], "largest_v8_size": v8["views"]["p0.30-z3.0"]["size"], "seed": SEED}


def pilot_frame(df, rows: list[int]):
    """The selected rows two times (2 steps of 16 prompts in one epoch)."""
    return df.iloc[rows + rows].reset_index(drop=True)


def main():
    import pandas as pd

    parquet, vdir, out_parquet, out_json = sys.argv[1:]
    manifest = [json.loads(x) for x in open(Path(vdir) / "MANIFEST.jsonl")]
    sel = choose(manifest)
    df = pd.read_parquet(parquet)
    pilot_frame(df, sel["rows"]).to_parquet(out_parquet)
    Path(out_json).write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel))


if __name__ == "__main__":
    main()
