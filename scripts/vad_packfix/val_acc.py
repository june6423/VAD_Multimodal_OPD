"""Fast-validation accuracy of one arm from the per-row validation dumps (<out>/validation/<step>.jsonl).

Each dump row holds the input, the output, the ground truth, the score and the reward extra info of the A1 reward
function (acc, judge, judge_source, ...). The accuracy of a row is "acc" when present, else "score".
VAL_ACC.json gives, for each step: the row count, the mean accuracy, the accuracy by benchmark when the rows name
it, the judge_source counts, and the per-row accuracy with a sha256 of input + ground truth (the pairing key of
stats.py). The check fails (exit 1) when step 0 has no dump or a row has no accuracy.
Usage: python val_acc.py <arm output root>
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def row_acc(r):
    v = r.get("acc", r.get("score"))
    return None if v is None else float(v)


def main():
    out = Path(sys.argv[1])
    steps, bad = {}, []
    for path in sorted((out / "validation").glob("*.jsonl"), key=lambda p: int(p.stem)):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        accs, by_bench, keys = [], defaultdict(list), []
        for i, r in enumerate(rows):
            a = row_acc(r)
            if a is None:
                bad.append([path.stem, i])
                continue
            accs.append(a)
            bench = r.get("benchmark", r.get("data_source"))
            if bench is not None:
                by_bench[str(bench)].append(a)
            keys.append(hashlib.sha256((str(r.get("input")) + "\x00" + str(r.get("gts"))).encode()).hexdigest()[:16])
        steps[path.stem] = {
            "rows": len(rows), "acc_mean": sum(accs) / len(accs) if accs else None,
            "by_benchmark": {b: {"rows": len(v), "acc_mean": sum(v) / len(v)} for b, v in sorted(by_bench.items())},
            "judge_source": dict(Counter(str(r.get("judge_source")) for r in rows)),
            "row_acc": accs, "row_key": keys, "fields": sorted(rows[0]) if rows else [],
        }
    check = {"steps": sorted(steps, key=int), "rows_without_acc": bad, "pass": "0" in steps and not bad}
    (out / "VAL_ACC.json").write_text(json.dumps({"check": check, "steps": steps}, indent=1))
    print(json.dumps({"check": check, "acc": {s: (v["rows"], v["acc_mean"]) for s, v in steps.items()}}))
    sys.exit(0 if check["pass"] else 1)


if __name__ == "__main__":
    main()
