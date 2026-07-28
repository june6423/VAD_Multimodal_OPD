# Generated data

This directory intentionally contains no released image payloads.

Run:

```bash
python scripts/prepare_data.py --data-dir data
bash scripts/prepare_benchmarks.sh
```

The commands create the training parquet, the three aligned visual views, the
benchmark JSON files, and the lightweight in-training validation parquet.
