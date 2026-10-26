"""VLM-VAD-ROLLOUT-001 gate G5: scripts/vad_rollout/ckpt_keep.py keeps step 80 and the final step."""

import filecmp
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/vad_rollout/ckpt_keep.py"


def save_step(ckpt_dir: Path, step: int) -> None:
    """A synthetic checkpoint: actor shards, a Hugging Face config, the dataloader state, then the tracker."""
    actor = ckpt_dir / f"global_step_{step}" / "actor"
    (actor / "huggingface").mkdir(parents=True)
    for rank in range(2):
        (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(bytes([step % 256, rank]) * 4096)
    (actor / "huggingface" / "config.json").write_text(json.dumps({"step": step}))
    (ckpt_dir / f"global_step_{step}" / "data.pt").write_bytes(b"dataloader" * step)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text(str(step))


def rotate(ckpt_dir: Path, step: int) -> None:
    """max_actor_ckpt_to_keep removes the actor directory of an old step."""
    for path in (ckpt_dir / f"global_step_{step}" / "actor").rglob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted((ckpt_dir / f"global_step_{step}" / "actor").rglob("*"), reverse=True):
        path.rmdir()
    (ckpt_dir / f"global_step_{step}" / "actor").rmdir()


def test_g5_keep_step_80_and_cleanup(tmp_path):
    ckpt_dir, keep_dir = tmp_path / "checkpoints", tmp_path / "keep"
    for step in (60, 70):
        save_step(ckpt_dir, step)
    watch = subprocess.Popen(
        [sys.executable, str(SCRIPT), "watch", str(ckpt_dir), str(keep_dir), "--step", "80", "--poll-seconds", "0.2"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(1.0)
    assert watch.poll() is None and not (keep_dir / "global_step_80").exists()

    save_step(ckpt_dir, 80)
    output, _ = watch.communicate(timeout=30)
    assert watch.returncode == 0, output
    keep = json.loads((keep_dir / "KEEP.json").read_text())
    assert keep["verified"] is True and keep["status"] == "COPIED" and keep["latest_at_copy"] == 80

    source, copy = ckpt_dir / "global_step_80", keep_dir / "global_step_80"
    source_files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    copy_files = sorted(p.relative_to(copy) for p in copy.rglob("*") if p.is_file())
    assert source_files == copy_files and len(copy_files) == 4
    assert all(filecmp.cmp(source / f, copy / f, shallow=False) for f in source_files)

    for step in range(90, 131, 10):
        save_step(ckpt_dir, step)
        rotate(ckpt_dir, step - 30)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "cleanup", str(ckpt_dir), "--final-step", "130",
         "--record", str(tmp_path / "CLEANUP.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    remaining = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("global_step_*") if p.is_dir())
    assert remaining == ["checkpoints/global_step_130", "keep/global_step_80"]
    cleanup = json.loads((tmp_path / "CLEANUP.json").read_text())
    assert cleanup["kept"] == "global_step_130" and len(cleanup["removed"]) == 7
    print(f"G5 copy after latest 80: files {len(copy_files)} byte-equal True; KEEP.json {keep['status']}; "
          f"after cleanup: {remaining}; removed {len(cleanup['removed'])} bytes freed {cleanup['bytes_freed']}")
