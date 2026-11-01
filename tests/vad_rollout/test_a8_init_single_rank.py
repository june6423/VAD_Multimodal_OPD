"""TASK-VLM-VAD-ROLLOUT-001-054 defect: concurrent A8 jobs on one host must each get a one-rank process group.

run_grads.sh starts 6 grads.py processes at the same time. Each calls trainer_io.init_single_rank. The test starts
two such processes at the same time on CPU (gloo in place of nccl) and requires that both succeed.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

A8 = Path(__file__).resolve().parents[2] / "scripts" / "vad_rollout" / "a8"


def _child():
    import torch

    sys.path.insert(0, str(A8))
    import trainer_io

    real = torch.distributed.init_process_group

    def cpu_init(backend, *args, **kwargs):
        return real("gloo", *args, **kwargs)

    torch.distributed.init_process_group = cpu_init
    trainer_io.init_single_rank()
    value = torch.ones(1)
    torch.distributed.all_reduce(value)
    time.sleep(3)
    torch.distributed.destroy_process_group()
    print("ok", float(value))


def test_two_concurrent_jobs_each_get_a_process_group():
    env = {k: v for k, v in os.environ.items() if k not in ("MASTER_ADDR", "MASTER_PORT")}
    procs = []
    for _ in range(2):
        procs.append(subprocess.Popen([sys.executable, __file__, "child"], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True))
        time.sleep(1.0)
    results = [p.communicate(timeout=300) for p in procs]
    for p, (out, err) in zip(procs, results):
        assert p.returncode == 0, err[-2000:]
        assert "ok 1.0" in out


if __name__ == "__main__" and sys.argv[1:] == ["child"]:
    _child()
