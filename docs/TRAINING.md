# Training

## Public profiles

The release exposes two controlled profiles:

| Setting | 4B | 9B |
| --- | ---: | ---: |
| Base model | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-9B` |
| Train batch size | 96 | 96 |
| PPO mini-batch size | 96 | 96 |
| Rollouts per sample | 8 | 8 |
| Learning rate | $2\times10^{-6}$ | $2\times10^{-6}$ |
| Positive support cap | 0.8 | 0.7 |
| Weak teacher anchor | 0.1 | 0.1 |
| Default training steps | 130 | 130 |

Both profiles use a fixed initial teacher, top-100 candidate support with a tail
bucket, the budgeted asymmetric target, and seed 42. The old internal names
`target_mode=vc` and `vc_target_variant=budgeted_asvc` remain in the runtime so
existing checkpoints and logs can be reproduced.

## W&B organization

Every public launcher uses:

```text
PROJECT_NAME=VAD
```

The 4B and 9B runs therefore appear in one W&B project. Change
`EXPERIMENT_NAME` to distinguish seeds or ablations, but keep `PROJECT_NAME`
constant when comparing algorithms.

The launcher never reads credentials from source files. Export them:

```bash
export WANDB_API_KEY="..."
```

## Mandatory training-time evaluation

The release launcher fixes:

```text
BENCHMARK_EVAL_ENABLE=True
BENCHMARK_EVAL_FREQ=10
BENCHMARK_EVAL_BENCHMARKS=vstar,zoombench,hrbench-4k,hrbench-8k
```

Profile files and inherited shell variables cannot turn these settings off.
The launcher checks the fast-evaluation parquet and judge endpoint before
starting training. The four score series are written into the same W&B run as
the loss and optimization metrics.

The training-time scorer is a fast monitoring path. It is useful for curve
inspection and checkpoint selection, but it is not the official GPT-OSS final
result. Final tables must be produced by `scripts/evaluate_official.sh`.

## Judge endpoint

The default endpoint is:

```text
http://127.0.0.1:8900/v1
```

When the judge is hosted on another machine:

```bash
export BENCHMARK_EVAL_JUDGE_API_BASE="http://10.x.x.x:8900/v1"
export BENCHMARK_EVAL_JUDGE_API_KEY="EMPTY"
export BENCHMARK_EVAL_JUDGE_MODEL="openai/gpt-oss-120b"
```

The judge service must advertise `openai/gpt-oss-120b` from its `/models`
endpoint. The launcher records the endpoint and model name, but never stores
the API key in its manifest.

## Launch and override

```bash
bash scripts/train_vad.sh --profile 4b
```

Environment variables override the profile defaults:

```bash
MODEL_PATH="/models/Qwen3.5-4B" \
TEACHER_MODEL_PATH="/models/Qwen3.5-4B" \
EXPERIMENT_NAME="vad-4b-seed42" \
PROJECT_NAME="VAD" \
bash scripts/train_vad.sh --profile 4b
```

Additional Hydra overrides can be passed after `--`:

```bash
bash scripts/train_vad.sh --profile 4b -- \
  trainer.max_actor_ckpt_to_keep=3
```

Do not use Hydra overrides to disable validation for a result intended for the
public training comparison.

## Outputs

| Path | Content |
| --- | --- |
| `checkpoints/<experiment>/` | FSDP actor checkpoints |
| `rollouts/<experiment>/` | Saved rollout records |
| `logs/<experiment>/launch.env` | Sanitized launch configuration |
| `wandb/` | Local W&B cache |

The launch manifest records the four-benchmark validation contract, judge
model, data path, fixed teacher, and seed. It intentionally omits credentials.
