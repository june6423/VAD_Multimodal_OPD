# Training data and evidence views

## Data source

The public preparation command downloads
`yuanqianhao/Vision-OPD-6K` from Hugging Face. The source dataset supplies:

- the full image seen by the student;
- an evidence-centered clear crop seen by the privileged teacher;
- the question and answer metadata.

The release does not commit the source images or generated parquet.

## Three aligned conditions

Every training row represents one question under three image conditions.

| Condition | Meaning | Runtime column |
| --- | --- | --- |
| Student input | Full image used during rollout and deployment | `images` |
| Evidence present | Clear evidence crop used by the fixed teacher | `bbox_images` |
| Evidence degraded | Matched counterfactual crop with fine detail removed | `degraded_bbox_images` |

The evidence-degraded image is produced from the evidence-present crop by:

1. bilinear downsampling to 10 percent of the original width and height;
2. nearest-neighbor upsampling to the original crop resolution.

Only visual pixels change between the evidence-present and evidence-degraded
conditions. The teacher parameters, question, and student-generated prefix stay
fixed.

## Build command

```bash
python scripts/prepare_data.py \
  --data-dir data \
  --scale 0.10 \
  --workers 16
```

Useful modes:

```bash
# Rebuild from already downloaded source files.
python scripts/prepare_data.py --data-dir data --skip-download --overwrite

# Validate every row and image without rewriting.
python scripts/prepare_data.py --data-dir data --verify-only

# Create a small explicit smoke-test output.
python scripts/prepare_data.py \
  --data-dir data \
  --skip-download \
  --limit 16 \
  --output data/train_vcd_smoke.parquet \
  --manifest data/vcd_smoke_manifest.json \
  --overwrite
```

## Generated artifacts

| Path | Purpose |
| --- | --- |
| `data/train_vcd.parquet` | Training records consumed by `train_vcd.sh` |
| `data/evidence_degraded/` | Generated evidence-degraded PNG files |
| `data/vcd_data_manifest.json` | Row count, column mapping, degradation settings, and verification result |

The parquet stores absolute image paths. If the data tree is moved to another
location, rerun the preparation command so the paths are regenerated.

## Validation guarantees

Preparation fails unless:

- all three required image columns exist;
- every image column contains exactly one path;
- every referenced file exists;
- the evidence-present and evidence-degraded images have identical dimensions;
- row-level metadata maps the semantic conditions to the expected columns; and
- the output row count matches the selected source records.

The training launcher also checks that `train_vcd.parquet` exists before
starting Ray or allocating model memory.
