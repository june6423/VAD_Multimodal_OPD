# Open-source release plan

## Phase 1: training and evaluation framework

The current directory implements the first public release boundary.

Included:

- the complete VCD target-construction and paired-view training path;
- fixed-initial-teacher 4B and 9B profiles;
- deterministic evidence-present and evidence-degraded data preparation;
- mandatory four-benchmark validation every 10 training steps;
- one shared W&B project contract;
- checkpoint merging and vLLM serving;
- the six-benchmark Vision-OPD official evaluation pipeline;
- GPT-OSS-120B serving defaults, judge repair, completeness audits, and
  reproducibility metadata;
- focused unit and smoke tests.

Excluded:

- historical analysis scripts and generated plots;
- ad hoc supervisor, watcher, and GPU-occupancy wrappers;
- corrected local Qwen judge pipelines;
- API-model comparison launchers;
- unrelated upstream recipes for VLA, fully-async policy training, one-step
  off-policy training, and transfer-queue experiments;
- logs, W&B caches, rollouts, benchmark outputs, and checkpoints;
- paper drafts, proposals, internal reports, and screenshot assets.

The vendored `verl/` directory is runtime infrastructure required for a
self-contained training release. It is not an archive of project analyses.

## Phase 2: public artifacts

Available checkpoints:

- [VAD for Qwen3.5-4B](https://huggingface.co/zhangkangning/VAD_for_Qwen3.5-4b);
- [VAD for Qwen3.5-9B](https://huggingface.co/zhangkangning/VAD_for_Qwen3.5-9b).

Add the remaining artifacts only after they are ready and licensed:

- processed training-data metadata or a reproducible dataset release;
- exact checkpoint-selection rules derived from training-time curves;
- public paper, project-page, and dataset links;
- a verified `CITATION.cff` and BibTeX entry;
- published official result metadata matching the downloadable checkpoints.

Large payloads should live on Hugging Face or another artifact host. GitHub
should contain checksums, cards, and download links rather than model weights
or image archives.

## Phase 3: curated analysis package

Analysis code can be released later as a separate, optional package after it is
reduced to the scripts needed to reproduce paper figures and tables. It should
not reintroduce the current collection of one-off probes into the main branch.

A curated analysis release should provide:

- one documented entrypoint per paper figure or table;
- small manifests instead of absolute internal paths;
- fixed input/output schemas;
- no private service endpoints or credentials;
- tests for the target-construction invariants used in the paper.

## Pre-publication checklist

- Run `pytest -q`.
- Run Python compilation and shell syntax checks.
- Run the 4B and 9B launcher dry runs and verify the four benchmark names and
  frequency 10 in their output.
- Build a small local data sample and run `--verify-only`.
- Run an answer/judge audit fixture with both valid and invalid labels.
- Scan tracked text for tokens, API keys, private hosts, user names, and
  absolute workspace paths.
- Confirm that generated data, outputs, W&B files, and weights remain ignored.
- Confirm Apache-2.0 attribution for Vision-OPD and verl.
- Confirm all six final benchmark rows were produced by the official GPT-OSS
  path and contain zero noncanonical judge labels.
- Add checkpoint, dataset, paper, and citation links only after verifying that
  each public target exists.

## GitHub main-branch policy

Keep the main branch centered on training, data preparation, evaluation,
documentation, and tests. New experiment-specific launchers should first be
expressed as configuration overrides. A new script belongs on main only when it
defines a reusable public workflow.
