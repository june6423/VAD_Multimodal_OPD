"""Concatenate the part dumps of one split validation (DIRECTIVE-VLM-VAD-PACKFIX-001-008, option B).

The parts of val_split.py are contiguous row ranges, so the dumps in part order give the original row order. Each
row keeps its fields (input, gts, score, ...), so the row key of val_acc.py (sha256 of input and gts) stays.
The check fails (exit 1, no output file) when a part dump has another row count than its range in VAL_SPLIT.json.
Usage: python val_concat.py <VAL_SPLIT.json> <out jsonl> <part 0 jsonl> ... <part n-1 jsonl>
"""

import json
import sys
from pathlib import Path


def main():
    split = json.loads(Path(sys.argv[1]).read_text())
    out, dumps = Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    if len(dumps) != len(split["parts"]):
        print(f"val concat: {len(dumps)} dumps for {len(split['parts'])} parts")
        return 1
    lines = []
    for part, path in zip(split["parts"], dumps):
        rows = [line for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != part["end"] - part["start"]:
            print(f"val concat: part {part['part']} has {len(rows)} rows, expected {part['end'] - part['start']}")
            return 1
        lines += rows
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(out)
    print("val concat rows", len(lines), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
