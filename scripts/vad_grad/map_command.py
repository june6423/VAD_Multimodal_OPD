"""A copy of the A1 COMMAND.txt (TASK-VLM-VAD-ROLLOUT-001-022) with the Vast paths mapped to the b200 paths.

The A8 code (trainer_io.compose_a1_config) composes the A1 config from the main_ppo arguments of COMMAND.txt. These
hold absolute paths of the Vast instance. On b200 the data is under <harness root>/data and the code is the running
repo slot. Mapped prefixes: /workspace/harness/data/ -> <harness root>/data/, /workspace/harness/vad_train/ -> <repo
root>/. Output paths (/workspace/harness/outputs/...) stay; the A8 code does not read them. No other text changes.
Usage: python map_command.py <COMMAND.txt> <out file> <repo root> <harness root>
Prints the count of each mapping and every mapped line of the main_ppo arguments.
"""

import sys
from pathlib import Path

src, dst, repo, harness = sys.argv[1:5]
text = Path(src).read_text()
maps = [("/workspace/harness/data/", harness.rstrip("/") + "/data/"), ("/workspace/harness/vad_train/", repo.rstrip("/") + "/")]
counts = {old: text.count(old) for old, _ in maps}
for old, new in maps:
    text = text.replace(old, new)
Path(dst).write_text(text)
print("map_command", {old: f"{counts[old]} -> {new}" for old, new in maps})
block = text.split("== main_ppo arguments", 1)[1].split("\n==", 1)[0]
for line in block.splitlines():
    if any(new in line for _, new in maps):
        print("  ", line)
