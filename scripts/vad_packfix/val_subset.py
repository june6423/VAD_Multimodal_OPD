"""20-row subset of the fast-validation parquet for the PACKFIX-001 pilot (HANDOFF-VLM-VAD-PACKFIX-001-R2 Part A item 5).

For each of the 4 benchmarks (data_source), 5 rows: the row with the largest image file (the longest prompt in
tokens, since the image tokens grow with the image) and the 4 rows with the smallest extra_info.index of the others.
Writes the subset parquet and a JSON list of the chosen rows.
Usage: python val_subset.py <fast-validation parquet> <out parquet> <out json>
"""

import json
import os
import sys

import pandas as pd

PER_SOURCE = 5


def image_bytes(cell) -> int:
    return sum(os.path.getsize(item["path"]) for item in cell if isinstance(item, dict) and item.get("path"))


def main():
    src, out_parquet, out_json = sys.argv[1:4]
    df = pd.read_parquet(src)
    df["_bytes"] = df["images"].map(image_bytes)
    df["_index"] = df["extra_info"].map(lambda e: int(e["index"]))
    keep = []
    for source, part in sorted(df.groupby("data_source"), key=lambda kv: kv[0]):
        largest = part["_bytes"].idxmax()
        others = part.drop(index=largest).sort_values("_index").index[: PER_SOURCE - 1]
        keep += [largest, *others]
    sub = df.loc[keep]
    rows = [{"row": int(i), "data_source": r["data_source"], "index": int(r["_index"]), "image_bytes": int(r["_bytes"])}
            for i, r in sub.iterrows()]
    sub.drop(columns=["_bytes", "_index"]).reset_index(drop=True).to_parquet(out_parquet)
    with open(out_json, "w") as f:
        json.dump({"source": src, "rows": rows, "count": len(rows)}, f, indent=1)
    print("val subset rows", len(rows), "largest image bytes", max(r["image_bytes"] for r in rows))


if __name__ == "__main__":
    main()
