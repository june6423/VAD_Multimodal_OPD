"""Gate G2 of HANDOFF-VLM-VAD-VIEW-002-R1: the x-plus image of each row in step 1 of each arm.

Reads <root>/<arm>/views_log/assign_1.jsonl (each row when hook.py assigned its view: position, uid, slot, row key,
path) and teacher_1.jsonl (each row just before the teacher inputs are built: uid, slot, path, sha256 of the file),
and VIEWMAP.json. The rule of Definition 3 is coded here again (not imported from hook.py):
  B: slot 5. T8 and N1V8: position p gets slot p mod 8 + 1. N1R: slot 1 + numpy default_rng([20261113, step])
  .integers(0, 8, rows)[p].
For 20 random prompts (uids, random.Random(20261113 + arm number).sample) of each arm, each row of the prompt must have:
  1. in the assignment file, the slot of the rule at its position;
  2. in the teacher file, the multiset of slots of the prompt equal to the multiset of the rule;
  3. path = VIEWMAP row of the prompt key, at the slot; sha256 of the logged file = sha256 that this script computes
     of that path;
  4. slot 5 (v0): path = the prompt key (the released teacher image of bbox_images).
Also: the N1R slot counts of steps 1 to 40 with 96 prompts (the rule), and for each arm the count of rows and of each slot
in the teacher file. Usage: python g2check.py <root> <VIEWMAP.json> <out json>. Exit 0 = pass, 1 = fail.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARMS = ("B", "T8", "N1V8", "N1R")
PROMPTS, SEED = 20, 20261113


def rule(arm: str, rows: int, step: int) -> list[int]:
    if arm == "B":
        return [5] * rows
    if arm in ("T8", "N1V8"):
        return [p % 8 + 1 for p in range(rows)]
    return [int(x) + 1 for x in np.random.default_rng([SEED, step]).integers(0, 8, rows)]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_arm(arm: str, d: Path, vmap: dict, k: int) -> dict:
    assign = [json.loads(x) for x in open(d / "views_log" / "assign_1.jsonl") if x.strip()]
    teacher = [json.loads(x) for x in open(d / "views_log" / "teacher_1.jsonl") if x.strip()]
    expect = rule(arm, len(assign), 1)
    by_uid_a, by_uid_t, key = defaultdict(list), defaultdict(list), {}
    for x in assign:
        by_uid_a[x["uid"]].append(x)
        key[x["uid"]] = x["key"]
    for x in teacher:
        by_uid_t[x["uid"]].append(x)
    uids = sorted(by_uid_a)
    pick = random.Random(SEED + k).sample(uids, min(PROMPTS, len(uids)))
    fails, rows = [], 0
    for u in pick:
        for x in by_uid_a[u]:
            if x["slot"] != expect[x["position"]]:
                fails.append([u, "assign slot", x["position"], x["slot"], expect[x["position"]]])
        want = sorted(expect[x["position"]] for x in by_uid_a[u])
        got = sorted(x["slot"] for x in by_uid_t[u])
        if want != got:
            fails.append([u, "teacher slots", got, want])
        for x in by_uid_t[u]:
            rows += 1
            path = vmap["rows"][key[u]][x["slot"] - 1]
            if x["path"] != path:
                fails.append([u, "path", x["path"], path])
            if x["slot"] == 5 and x["path"] != key[u]:
                fails.append([u, "v0 path", x["path"], key[u]])
            if x["sha256"] != sha256(path):
                fails.append([u, "sha256", x["path"]])
    return {"rows_assign": len(assign), "rows_teacher": len(teacher), "prompts": len(uids), "checked_prompts": len(pick),
            "checked_rows": rows, "teacher_slot_counts": {str(s): c for s, c in sorted(Counter(
                x["slot"] for x in teacher).items())}, "fails": fails[:50], "fail_count": len(fails)}


def main():
    root, vmap, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    res = {"arms": {a: check_arm(a, root / a, vmap, k) for k, a in enumerate(ARMS)}}
    counts = Counter(s for step in range(1, 41) for s in rule("N1R", 96, step))
    res["n1r_counts_steps_1_40"] = {str(s): counts.get(s, 0) for s in range(1, 9)}
    res["pass"] = all(v["fail_count"] == 0 and v["checked_prompts"] == PROMPTS for v in res["arms"].values())
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({a: (v["checked_rows"], v["fail_count"]) for a, v in res["arms"].items()}),
          json.dumps(res["n1r_counts_steps_1_40"]), "pass", res["pass"])
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
