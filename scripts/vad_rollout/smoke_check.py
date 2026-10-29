#!/usr/bin/env python3
"""A0 smoke check for VLM-VAD-ROLLOUT-001 (HANDOFF-VLM-VAD-ROLLOUT-001-R4, Part A item 4, gate G4).

Usage: python3 scripts/vad_rollout/smoke_check.py <A0 output root> <out dir>

Checks the outputs of train_arm.sh A0: rows of rollouts/1.jsonl and 2.jsonl and their uid and row_index
fields, the token-statistics records of steps 1 and 2 (all ranks), the one-to-one join by (uid, row_index),
merged/step_2/config.json, and the step metrics of STEP_METRICS.json that G4 names (timing_s/step, the
validation metrics of step 2, the memory metrics). Writes <out dir>/SMOKE_CHECK.json and exits 1 when a
check fails. Uses read_jsonl and sha256_file of analyze_rollouts.py.
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analyze_rollouts", HERE / "analyze_rollouts.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

STEPS = (1, 2)


def check_step(root: Path, step: int, rows_expected: int) -> dict:
    dump_path = root / "rollouts" / f"{step}.jsonl"
    stat_paths = sorted((root / "tokstats").glob(f"step_{step}_rank_*.jsonl"))
    dump = analyze.read_jsonl(dump_path) if dump_path.exists() else []
    stats = [record for path in stat_paths for record in analyze.read_jsonl(path)]
    dump_keys = Counter((row.get("uid"), row.get("row_index")) for row in dump)
    stat_keys = Counter((record.get("uid"), record.get("row_index")) for record in stats)
    lengths_ok = all(
        len(record[name]) == record["response_length"] for record in stats for name in analyze.VECTORS
    )
    result = {
        "dump_file": str(dump_path),
        "dump_sha256": analyze.sha256_file(dump_path) if dump_path.exists() else None,
        "dump_rows": len(dump),
        "dump_rows_with_uid_and_row_index": sum("uid" in row and "row_index" in row for row in dump),
        "tokstats_files": [p.name for p in stat_paths],
        "tokstats_records": len(stats),
        "join_one_to_one": dump_keys == stat_keys and max(dump_keys.values(), default=0) == 1,
        "vector_lengths_equal_response_length": lengths_ok,
    }
    result["ok"] = (
        result["dump_rows"] == rows_expected
        and result["dump_rows_with_uid_and_row_index"] == rows_expected
        and result["tokstats_records"] == rows_expected
        and result["join_one_to_one"]
        and lengths_ok
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--rows", type=int, default=768, help="rows per step (96 prompts x 8 rollouts)")
    args = parser.parse_args()

    report = {"steps": {step: check_step(args.root, step, args.rows) for step in STEPS}}
    metrics_path = args.root / "STEP_METRICS.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    report["step_metrics"] = {}
    for step in STEPS:
        values = metrics.get(str(step), {})
        report["step_metrics"][step] = {
            key: value for key, value in sorted(values.items())
            if key.startswith(("timing_s/", "perf/", "val-", "val/", "response_length/"))
        }
    step_two = report["step_metrics"][2]
    report["timing_step_present"] = all("timing_s/step" in report["step_metrics"][s] for s in STEPS)
    report["step2_validation_keys"] = sorted(k for k in step_two if k.startswith(("val-", "val/")))
    report["merged_step_2_config"] = (args.root / "merged" / "step_2" / "config.json").exists()
    report["ok"] = (
        all(report["steps"][s]["ok"] for s in STEPS)
        and report["timing_step_present"]
        and bool(report["step2_validation_keys"])
        and report["merged_step_2_config"]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "SMOKE_CHECK.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
