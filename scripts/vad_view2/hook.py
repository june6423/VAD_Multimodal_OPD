"""Teacher-image hook of VLM-VAD-VIEW-002 (HANDOFF-VLM-VAD-VIEW-002-R1 Part A items 2 and 3, Definitions 2 and 3).

RayPPOTrainer.fit calls this module only when the environment flag VAD_VIEW_ARM is set (ray_trainer._vad_view2_hook).
run_view2.sh gives the flag to the Ray workers through ray_kwargs.ray_init.runtime_env.env_vars.

Views (Definition 2): slots 1 to 8 = v1 (0, 1.5), v2 (0, 2), v3 (0, 3), v4 (p1, 1.5), v5 = v0 (p1, 2), v6 (p1, 3),
v7 (0.30, 2), v8 (0.30, 3). Slot 5 is the released teacher image (the bbox_images item of the row, not changed). The
other slots are the VIEW-001 views.py images. VAD_VIEW_MAP is the VIEWMAP.json of viewmap.py: for each released teacher
image path (the row key), the 8 image paths. x-minus (degraded_bbox_images) does not change.

Arms (Definition 3), for the rows of one step after the generation (row j of the batch, before _balance_batch):
  B: slot 5 for each row. The hook does not change the batch.
  T8: rollout.n 8, row j is sibling k = j mod 8 + 1 of its prompt and gets slot k.
  N1V8: rollout.n 8 in the config, but the generation makes 1 response for each prompt (generation_n). The hook then
    copies each response with its rollout log probabilities 8 times (copy_responses, interleaved, so each copy keeps
    the uid of its prompt). Copy k = j mod 8 + 1 gets slot k. The update uses the 768 rows as in B.
  N1R: rollout.n 1. Row j gets slot 1 + the j-th value of numpy default_rng([20261113, step]).integers(0, 8, rows):
    one uniform draw for each prompt, independent for each prompt and step.
Non-B arms replace the bbox_images item of each row with [{"path": <view image>}] (slot 5 keeps the released item)
and add the column vad_view (the slot of each row). The rollout dump gets vad_view for each arm (dump_views).
VAD_VIEW_LOG_DIR (optional): assign_<step>.jsonl (each row at assignment: position, uid, slot, row key, image path) and
teacher_<step>.jsonl (each row just before the teacher inputs are built: position, uid, slot, image path, sha256 of the
image file). check_view2.sh reads these files for gate G2.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ARMS = ("B", "T8", "N1V8", "N1R")
N_VIEWS = 8
V0_SLOT = 5
N1R_SEED = 20261113
TEACHER_KEY = "bbox_images"
VIEW_COLUMN = "vad_view"
VIEW_NAMES = ("p0.00-z1.5", "p0.00-z2.0", "p0.00-z3.0", "p1-z1.5", "p1-z2.0", "p1-z3.0", "p0.30-z2.0", "p0.30-z3.0")

_viewmap: dict | None = None
_sha: dict[str, str] = {}


def arm() -> str:
    a = os.environ.get("VAD_VIEW_ARM", "")
    if a not in ARMS:
        raise ValueError(f"VAD_VIEW_ARM must be one of {ARMS}, not {a!r}")
    return a


def generation_n(a: str, n: int) -> int:
    """Responses per prompt in the generation. N1V8 generates 1 and copies it 8 times (copy_responses)."""
    if a == "N1V8":
        if n != N_VIEWS:
            raise ValueError(f"N1V8 needs rollout.n {N_VIEWS} in the config, not {n}")
        return 1
    if a == "N1R" and n != 1:
        raise ValueError(f"N1R needs rollout.n 1, not {n}")
    if a in ("B", "T8") and n != N_VIEWS:
        raise ValueError(f"{a} needs rollout.n {N_VIEWS}, not {n}")
    return n


def view_slots(a: str, rows: int, step: int) -> list[int]:
    """The slot (1 to 8) of each row of the step batch in the order after the generation (and the N1V8 copy)."""
    if a == "B":
        return [V0_SLOT] * rows
    if a in ("T8", "N1V8"):
        if rows % N_VIEWS:
            raise ValueError(f"{a}: {rows} rows is not a multiple of {N_VIEWS}")
        return [j % N_VIEWS + 1 for j in range(rows)]
    if a == "N1R":
        return [int(x) + 1 for x in np.random.default_rng([N1R_SEED, step]).integers(0, N_VIEWS, rows)]
    raise ValueError(a)


def row_key(item) -> str:
    """The released teacher image path of a bbox_images item (a list or array with one {"path": ...} dict)."""
    items = list(item)
    if len(items) != 1 or not isinstance(items[0], dict) or "path" not in items[0]:
        raise ValueError(f"unexpected bbox_images item {item!r}")
    return str(items[0]["path"])


def viewmap() -> dict:
    global _viewmap
    if _viewmap is None:
        _viewmap = json.loads(Path(os.environ["VAD_VIEW_MAP"]).read_text())
        if tuple(_viewmap["views"]) != VIEW_NAMES:
            raise ValueError(f"VIEWMAP views {_viewmap['views']} differ from {VIEW_NAMES}")
    return _viewmap


def copy_responses(gen_output, a: str):
    """N1V8: each response with its log probabilities 8 times, interleaved (copy k of prompt i is row 8 i + k - 1)."""
    if a != "N1V8":
        return gen_output
    return gen_output.repeat(repeat_times=N_VIEWS, interleave=True)


def _log(name: str, step: int, lines: list[dict]):
    d = os.environ.get("VAD_VIEW_LOG_DIR")
    if not d:
        return
    Path(d).mkdir(parents=True, exist_ok=True)
    tmp = Path(d) / f".{name}_{step}.jsonl.tmp"
    tmp.write_text("".join(json.dumps(x) + "\n" for x in lines))
    os.replace(tmp, Path(d) / f"{name}_{step}.jsonl")


def assign_views(batch, step: int) -> None:
    """Set the x-plus image of each row (Definition 3). Called after the generation output joins the batch."""
    a = arm()
    items = batch.non_tensor_batch[TEACHER_KEY]
    keys = [row_key(x) for x in items]
    slots = view_slots(a, len(keys), step)
    paths = keys if a == "B" else [k if s == V0_SLOT else viewmap()["rows"][k][s - 1] for k, s in zip(keys, slots)]
    if a != "B":
        col = np.empty(len(keys), dtype=object)
        for j, (s, p) in enumerate(zip(slots, paths)):
            col[j] = items[j] if s == V0_SLOT else np.array([{"path": p}], dtype=object)
        batch.non_tensor_batch[TEACHER_KEY] = col
        batch.non_tensor_batch[VIEW_COLUMN] = np.array(slots, dtype=object)
    uids = batch.non_tensor_batch.get("uid", [None] * len(keys))
    _log("assign", step, [{"position": j, "uid": str(u), "slot": s, "key": k, "path": p}
                          for j, (u, s, k, p) in enumerate(zip(uids, slots, keys, paths))])


def dump_views(batch) -> list[int]:
    """The slot of each row for the rollout dump."""
    if VIEW_COLUMN in batch.non_tensor_batch:
        return [int(x) for x in batch.non_tensor_batch[VIEW_COLUMN]]
    if arm() != "B":
        raise ValueError(f"arm {arm()}: the batch has no {VIEW_COLUMN} column")
    return [V0_SLOT] * len(batch)


def file_sha256(path: str) -> str:
    if path not in _sha:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        _sha[path] = h.hexdigest()
    return _sha[path]


def log_teacher_images(batch, step: int) -> None:
    """Gate G2 record: the x-plus image path of each row as the teacher input builder reads it, with its sha256."""
    if not os.environ.get("VAD_VIEW_LOG_DIR"):
        return
    slots = dump_views(batch)
    uids = batch.non_tensor_batch.get("uid", [None] * len(slots))
    lines = []
    for j, (item, u, s) in enumerate(zip(batch.non_tensor_batch[TEACHER_KEY], uids, slots)):
        p = row_key(item)
        lines.append({"position": j, "uid": str(u), "slot": s, "path": p, "sha256": file_sha256(p)})
    _log("teacher", step, lines)
