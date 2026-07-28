# Official evaluation

## Two evaluation paths

The release separates training monitoring from final reporting.

| Path | Purpose | Main-table status |
| --- | --- | --- |
| Training-time four-benchmark validation | Curves, debugging, checkpoint selection | Not official |
| Vision-OPD official pipeline with GPT-OSS-120B | Final six-benchmark scoring | Official |

The official path uses:

1. `evaluation/vision_opd_official/infer.py`;
2. `evaluation/vision_opd_official/judge_qwenlm.py`;
3. `evaluation/vision_opd_official/cal_acc.py`.

The evaluator is derived from Vision-OPD commit `c2e345f`. Local repairs retain
the official scoring order and add deterministic seed forwarding, transport
error visibility, strict judge normalization, and completeness audits.

## Model inference settings

The formal wrapper enforces:

| Parameter | Value |
| --- | --- |
| Server `max_model_len` | 32768 |
| Client `MAX_TOKENS` | 8192 |
| `ENABLE_THINKING` | `False` |
| Seed | 42 |
| Temperature | 0 |

Start a merged model:

```bash
MAX_MODEL_LEN=32768 \
TENSOR_PARALLEL_SIZE=8 \
bash scripts/serve_checkpoint.sh /path/to/merged_checkpoint VCD
```

## Judge settings

The formal main-table judge is:

| Parameter | Value |
| --- | --- |
| Model | `openai/gpt-oss-120b` |
| Default endpoint | `http://127.0.0.1:8900/v1` |
| API key | `EMPTY` |
| Maximum tokens | 2048 |
| Temperature | 0 |

For access from another machine, start vLLM with `HOST=0.0.0.0` and use the
server's private-network IP in `JUDGE_API_BASE`.

## Six-benchmark command

```bash
MODEL_API_BASE="http://127.0.0.1:8000/v1" \
MODEL_ID="VCD" \
MODEL_TAG="vcd_selected_checkpoint" \
MODEL_PATH="/path/to/merged_checkpoint" \
JUDGE_API_BASE="http://JUDGE_HOST:8900/v1" \
bash scripts/evaluate_official.sh
```

The benchmark set is:

- VStar;
- ZoomBench;
- HRBench-4K;
- HRBench-8K;
- MME-RealWorld;
- MME-RealWorld-CN.

## Reusing existing model answers

If all model answers already exist, avoid re-running inference:

```bash
REUSE_ANSWER_ROOT="/path/to/model_answer" \
REUSE_MODEL_TAG="source_tag" \
MODEL_TAG="new_official_tag" \
MODEL_ID="VCD" \
MODEL_PATH="/path/to/checkpoint" \
JUDGE_API_BASE="http://JUDGE_HOST:8900/v1" \
bash scripts/evaluate_official.sh
```

The source layout must be:

```text
model_answer/
├── vstar/<source_tag>_seed42_answer.jsonl
├── zoombench/<source_tag>_seed42_answer.jsonl
├── hrbench-4k/<source_tag>_seed42_answer.jsonl
├── hrbench-8k/<source_tag>_seed42_answer.jsonl
├── mme-realworld/<source_tag>_seed42_answer.jsonl
└── mme-realworld-cn/<source_tag>_seed42_answer.jsonl
```

The wrapper creates links inside the new run and records the resolved source
path in each benchmark audit.

## Integrity gates

Before judging, the wrapper verifies:

- answer count equals the prepared benchmark count;
- every answer has a unique stable sample ID;
- no answer is empty;
- no answer begins with an API or future error marker.

After the official judge, it requires every `judge` field to be exactly `Yes`
or `No`. Explanatory outputs beginning with one of those labels are normalized.
Other invalid outputs are sent back to GPT-OSS one sample at a time, with up to
three retries. If any label is still invalid, scoring stops. An invalid or
missing judge response is never silently converted to `No`.

## Result directory

Each run under `outputs/official_eval/` contains:

```text
<run>/
├── model_answer/       # Generated or linked answers
├── judge/              # Official judge JSON lists
├── logs/               # Inference, judge, audit, and cal_acc output
├── audits/             # Per-benchmark machine-readable completeness checks
├── summary.md          # Six-benchmark table and counts
└── summary.json        # Full metadata and evaluator hashes
```

The formal main table should cite the protocol as:

```text
Vision-OPD official pipeline + GPT-OSS-120B judge
```

Any corrected local judge, first-letter-only proxy, or training-time score must
be labeled as a diagnostic result rather than an official final score.
