"""Checkpoint retention for VLM-VAD-ROLLOUT-001 (HANDOFF-VLM-VAD-ROLLOUT-001-R1, Part A item 4, gate G5).

watch:   poll <ckpt_dir>/latest_checkpointed_iteration.txt every --poll-seconds (default 60). When it reads
         --step (default 80) or more, copy <ckpt_dir>/global_step_<step> to <keep_dir>/global_step_<step>,
         verify the SHA-256 of every file against the source, write <keep_dir>/KEEP.json, and exit.
cleanup: remove every <ckpt_dir>/global_step_* directory except global_step_<final step>, and write the
         removed directories and the freed bytes to --record (CLEANUP.json).
The training keeps at most 3 checkpoints (trainer.max_actor_ckpt_to_keep=3), so the watch copies step 80
before the save of step 110 removes it.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digests(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def latest_iteration(ckpt_dir: Path) -> int:
    try:
        return int((ckpt_dir / "latest_checkpointed_iteration.txt").read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def watch(ckpt_dir: Path, keep_dir: Path, step: int, poll_seconds: float) -> int:
    name = f"global_step_{step}"
    source, target, record = ckpt_dir / name, keep_dir / name, keep_dir / "KEEP.json"
    if record.exists() and json.loads(record.read_text()).get("verified") and target.is_dir():
        print(utc(), "KEEP.json exists and is verified; nothing to do", flush=True)
        return 0
    while latest_iteration(ckpt_dir) < step:
        time.sleep(poll_seconds)
    keep_dir.mkdir(parents=True, exist_ok=True)
    result = {"source": str(source), "target": str(target), "latest_at_copy": latest_iteration(ckpt_dir),
              "started_at": utc()}
    if not (source / "actor").is_dir():
        result.update(status="MISSING", verified=False, finished_at=utc())
        record.write_text(json.dumps(result, indent=1) + "\n")
        print(utc(), "source checkpoint missing:", source, flush=True)
        return 1
    staging = keep_dir / f"{name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)
    staging.rename(target)
    source_digests, target_digests = tree_digests(source), tree_digests(target)
    verified = source_digests == target_digests
    result.update(status="COPIED" if verified else "DIFFERENT", verified=verified, files=len(target_digests),
                  bytes=tree_bytes(target), sha256=target_digests, finished_at=utc())
    if not verified:
        result["differing_files"] = sorted(
            k for k in set(source_digests) | set(target_digests) if source_digests.get(k) != target_digests.get(k)
        )
    record.write_text(json.dumps(result, indent=1) + "\n")
    print(utc(), result["status"], target, "files", result["files"], "bytes", result["bytes"], flush=True)
    return 0 if verified else 1


def cleanup(ckpt_dir: Path, final_step: int, record: Path) -> int:
    keep = f"global_step_{final_step}"
    if not (ckpt_dir / keep / "actor").is_dir():
        print(utc(), "final checkpoint missing; nothing removed:", ckpt_dir / keep, flush=True)
        return 1
    removed = []
    for path in sorted(ckpt_dir.glob("global_step_*")):
        if path.is_dir() and path.name != keep:
            size = tree_bytes(path)
            shutil.rmtree(path)
            removed.append({"path": str(path), "bytes": size})
    result = {"ckpt_dir": str(ckpt_dir), "kept": keep, "removed": removed,
              "bytes_freed": sum(item["bytes"] for item in removed), "finished_at": utc()}
    record.write_text(json.dumps(result, indent=1) + "\n")
    print(utc(), "removed", len(removed), "directories, bytes freed", result["bytes_freed"], flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("watch")
    p.add_argument("ckpt_dir", type=Path)
    p.add_argument("keep_dir", type=Path)
    p.add_argument("--step", type=int, default=80)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p = sub.add_parser("cleanup")
    p.add_argument("ckpt_dir", type=Path)
    p.add_argument("--final-step", type=int, required=True)
    p.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "watch":
        return watch(args.ckpt_dir, args.keep_dir, args.step, args.poll_seconds)
    return cleanup(args.ckpt_dir, args.final_step, args.record)


if __name__ == "__main__":
    sys.exit(main())
