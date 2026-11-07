"""Fixed 4-way row split of the fast-validation parquet (DIRECTIVE-VLM-VAD-PACKFIX-001-008, option B).

TASK-VLM-VAD-PACKFIX-001-026: one val_only process over the full set (2636 rows) reached the container memory limit,
because the agent worker RSS grows across the validation batches. Each full validation therefore runs in 4 val_only
processes, one for each part. The parts are contiguous row ranges in the original order, so the concatenation of the
4 part dumps (val_concat.py) keeps the original row order. The split depends only on the row count, so it is equal
for each validation of both arms and for step 0.
Writes <out dir>/part_<k>.parquet (k = 0 .. parts-1) and <out dir>/VAL_SPLIT.json (source, rows, row range of each part).
Usage: python val_split.py <fast-validation parquet> <out dir> [parts]
"""

import json
import sys
from pathlib import Path

import pandas as pd


def ranges(rows: int, parts: int):
    """[(start, end)] of `parts` contiguous ranges that cover range(rows), sizes differ by at most 1."""
    base, extra = divmod(rows, parts)
    out, start = [], 0
    for k in range(parts):
        end = start + base + (1 if k < extra else 0)
        out.append((start, end))
        start = end
    return out


def main():
    src, out_dir = sys.argv[1], Path(sys.argv[2])
    parts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    spans = ranges(len(df), parts)
    for k, (a, b) in enumerate(spans):
        df.iloc[a:b].reset_index(drop=True).to_parquet(out_dir / f"part_{k}.parquet")
    (out_dir / "VAL_SPLIT.json").write_text(json.dumps(
        {"source": src, "rows": len(df), "parts": [{"part": k, "start": a, "end": b} for k, (a, b) in enumerate(spans)]},
        indent=1))
    print("val split rows", len(df), "parts", [b - a for a, b in spans])


if __name__ == "__main__":
    main()
