"""Origin of the imported verl package (DIRECTIVE-VLM-VAD-GRAD-001-003 Decisions 5 and 6).

verl_origin(repo) imports verl and returns its resolved __file__ and the git HEAD of its directory. It stops (SystemExit)
when verl is not under the repo root, for example when the editable install of the vad environment points to the other
repo slot. The run scripts export PYTHONPATH=<repo root> so that verl imports from the task checkout.
Usage: python provenance.py <repo root>   (prints the record as one JSON line)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def verl_origin(repo: str) -> dict:
    import verl

    path = Path(verl.__file__).resolve()
    root = Path(repo).resolve()
    if not path.is_relative_to(root):
        raise SystemExit(f"STOP: verl imports from {path}, not from the repo root {root}")
    head = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    return {"verl_file": str(path), "verl_git_head": head}


if __name__ == "__main__":
    print(json.dumps(verl_origin(sys.argv[1])))
