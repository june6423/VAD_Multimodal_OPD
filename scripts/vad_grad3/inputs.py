"""Input hashes of HANDOFF-VLM-VAD-GRAD-003-R2 Part B item 1.

1. The files of rollout001/ITEM1_MANIFEST.sha256 that this experiment reads (A1 and A2 fp32 model shards at steps 80
   and 130, the merged A1 models, G3_PARAMS.txt, the TASK-053 rollouts, the A1 COMMAND.txt), checked against the
   manifest (path prefix harness/outputs/VLM-VAD-ROLLOUT-001/ -> <B200_ROOT>/harness/data/rollout001/).
2. The other inputs, hashed and recorded: every file of the Base model directory (M0), the TASK-GRAD-001-005 labels,
   the TASK-GRAD-001-009 Adam states V_A1_80 and V_A1_130, and the training parquet.
Writes INPUTS.sha256 (all files) and INPUTS_CHECK.json. Exit 1 when a file is missing or a manifest hash differs.

Usage: python inputs.py <B200_ROOT> <LABELS_ROOT> <ADAM_ROOT> <out dir>
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PREFIX = "harness/outputs/VLM-VAD-ROLLOUT-001/"
USED = re.compile(r"TASK-VLM-VAD-ROLLOUT-001-0(22|36)/(keep/global_step_80|checkpoints/global_step_130)/actor/model_world_size_6_rank_\d\.pt$"
                  r"|TASK-VLM-VAD-ROLLOUT-001-022/merged/step_(80|130)/[^/]+$"
                  r"|TASK-VLM-VAD-ROLLOUT-001-047/G3_PARAMS\.txt$"
                  r"|TASK-VLM-VAD-ROLLOUT-001-053/step(0|80|130)/rollouts\.jsonl$"
                  r"|TASK-VLM-VAD-ROLLOUT-001-022/COMMAND\.txt$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    b200, labels_root, adam_root, out = (Path(a) for a in sys.argv[1:5])
    data = b200 / "harness" / "data"
    r1 = data / "rollout001"
    t0 = time.time()
    expected = {}
    for line in (r1 / "ITEM1_MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if USED.search(name):
            expected[r1 / name[len(PREFIX):]] = digest
    others = sorted(p for p in (data / "models" / "Qwen3.5-4B").rglob("*") if p.is_file())
    others += [labels_root / s / "LABELS.jsonl" for s in ("step0", "step80", "step130")]
    others += [adam_root / f"V_A1_{s}.safetensors" for s in (80, 130)] + [data / "vad" / "train_vad.parquet"]
    files = sorted(expected) + others
    missing = [str(p) for p in files if not p.is_file()]
    present = [p for p in files if p.is_file()]
    with ThreadPoolExecutor(8) as pool:
        digests = dict(zip(present, pool.map(sha256, present)))
    bad = [str(p) for p, d in expected.items() if p in digests and digests[p] != d]
    with open(out / "INPUTS.sha256", "w") as f:
        for p in present:
            f.write(f"{digests[p]}  {p}\n")
    res = {"manifest_lines_used": len(expected), "manifest_ok": len(expected) - len(bad) - sum(
        str(p) in missing for p in expected), "manifest_bad": bad, "missing": missing, "other_files": len(others),
           "bytes": sum(p.stat().st_size for p in present), "seconds": round(time.time() - t0, 1)}
    (out / "INPUTS_CHECK.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("manifest_lines_used", "manifest_ok", "other_files", "bytes", "seconds")}))
    if bad or missing:
        raise SystemExit(f"STOP: {len(bad)} manifest hashes differ, {len(missing)} files missing")


if __name__ == "__main__":
    main()
