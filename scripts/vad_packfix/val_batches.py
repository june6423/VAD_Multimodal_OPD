"""Row range and image size of each validation batch (DIRECTIVE-VLM-VAD-PACKFIX-001-009 action 6).

The val_only processes read one part of val_split.py each, without shuffle (data.validation_shuffle False), in
batches of data.val_batch_size rows. Batch b of part k is therefore the part rows [b x size, (b + 1) x size), and the
global rows start at the part start of VAL_SPLIT.json. For each batch this script gives the global row range, the
image count, the total image pixels (width x height of each image file as stored, before the processor resize) and the
total image file bytes. The batch index is equal to the "batch=" index of the VAD_VALBATCH lines of valtrim.py, so
VT can relate the memory of each batch (memwatch.log) to its content.
Writes <out json> and prints one VAL_BATCH line for each batch.
Usage: python val_batches.py <split dir with VAL_SPLIT.json and part_<k>.parquet> <val batch size> <out json>
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


def images(cell):
    """(count, pixels, bytes) of the images of one row (items with a file path, as in val_subset.py)."""
    n = px = nb = 0
    for item in cell:
        with Image.open(item["path"]) as im:
            w, h = im.size
        n, px, nb = n + 1, px + w * h, nb + os.path.getsize(item["path"])
    return n, px, nb


def main():
    split_dir, size, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    split = json.loads((split_dir / "VAL_SPLIT.json").read_text())
    batches = []
    for part in split["parts"]:
        k, start = part["part"], part["start"]
        rows = [images(c) for c in pd.read_parquet(split_dir / f"part_{k}.parquet")["images"]]
        for b, a in enumerate(range(0, len(rows), size)):
            chunk = rows[a:a + size]
            x = {"part": k, "batch": b, "row_start": start + a, "row_end": start + a + len(chunk),
                 "images": sum(r[0] for r in chunk), "pixels": sum(r[1] for r in chunk),
                 "image_bytes": sum(r[2] for r in chunk)}
            batches.append(x)
            print(f"VAL_BATCH part {k} batch {b} rows {x['row_start']}-{x['row_end']} images {x['images']} "
                  f"pixels {x['pixels']} image_bytes {x['image_bytes']}")
    out.write_text(json.dumps({"val_batch_size": size, "split": split, "batches": batches}, indent=1))


if __name__ == "__main__":
    main()
