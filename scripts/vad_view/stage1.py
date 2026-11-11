"""Stage 1 of HANDOFF-VLM-VAD-VIEW-001-R2 (Part A items 2 and 3): teacher letter probabilities for each view.

Teacher: the fixed M0 model with the A1 config (COMMAND_B200.txt), DataParallelPPOActor._forward_micro_batch at
temperature 1, one row per forward (no packing). Teacher input of a view: the A1 teacher input
(RayPPOTrainer._maybe_build_self_distillation_batch through scripts/vad_grad3/trainer_io.teacher_views) with the
bbox_images entry of the row set to the view image. The input builder runs with target_mode full_teacher, so that it
builds only the x-plus input (the x-plus branch does not depend on target_mode).
fresh: for each kept row (views MANIFEST.jsonl minus REMOVED.json) and each view, the response is [eos] and the record
  is the log probability of each option letter (letters.fresh_ids) at response position 0, the first response token.
prefix: for each trajectory of a rollouts.jsonl, the answer-letter position of the TASK-VLM-VAD-ROLLOUT-001-056 teacher
  check (letters.prefix_ids). The response is cut after that position (the forward is causal, so the log probabilities
  at the position do not change), and the record is the log probability of each option letter token in the form of
  the sampled token. flag: ok, fx (no position or no token form), removed (row in REMOVED.json).
The A1 teacher input truncates at max_reprompt_len; a teacher prompt of that length stops the run (views.py removes
those rows before).
Resume: the output file is appended one line per row or trajectory; done rows are skipped. STOP_AFTER (0) stops once
after that many new lines with exit 75 (pilot).
Prefetch (DIRECTIVE-VLM-VAD-VIEW-001-002 item 2): worker processes (prefetch.py, Builder) build the CPU inputs of the
next rows or trajectories in file order: the dataset item, the base trajectory and the x-plus teacher input of each
view (one base for all views of the row). The parent runs the teacher forwards. Each line also has wait_seconds (the
parent waited for the worker) and build_seconds (worker).

Usage: python stage1.py fresh <repo> <COMMAND_B200.txt> <views dir> <FRESH.jsonl> [<rows file>]
       python stage1.py prefix <repo> <COMMAND_B200.txt> <views dir> <rollouts.jsonl> <PREFIX_step<s>.jsonl> [<traj file>]
Env: STOP_AFTER, PREFETCH_WORKERS (16; 0 = build in the parent as before), PREFETCH_BUDGET (160 view inputs submitted
ahead).
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP_EXIT = 75


def setup(repo: str):
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(HERE.parent / "vad_grad3"))
    sys.path.insert(0, str(HERE))


def prompt_text(raw_prompt) -> str:
    parts = []
    for m in raw_prompt:
        c = m["content"]
        if isinstance(c, str):
            parts.append(c)
        else:
            parts += [x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text"]
    return "\n".join(parts)


def image_paths(images) -> list[str]:
    items = images.tolist() if hasattr(images, "tolist") else list(images)
    return [x["path"] if isinstance(x, dict) else str(x) for x in items]


def load_views(views_dir: str) -> tuple[dict[int, dict], set[int]]:
    d = Path(views_dir)
    man = {r["row"]: r for r in map(json.loads, open(d / "MANIFEST.jsonl"))}
    removed = {r["row"] for r in json.loads((d / "REMOVED.json").read_text())["removed"]}
    return man, removed


class Teacher:
    def __init__(self, repo: str, command_txt: str, model: bool = True):
        import objectives
        import trainer_io
        from omegaconf import open_dict

        self.io, self.obj = trainer_io, objectives
        self.cfg = trainer_io.compose_a1_config(repo, command_txt)
        self.tokenizer, self.processor = trainer_io.tokenizer_processor(self.cfg)
        self.dataset = trainer_io.load_dataset(self.cfg, self.tokenizer, self.processor)
        self.short = copy.deepcopy(self.cfg)
        self.plus = copy.deepcopy(self.cfg)
        with open_dict(self.plus):
            self.plus.actor_rollout_ref.actor.self_distillation.target_mode = "full_teacher"
        self.max_len = int(self.cfg.actor_rollout_ref.actor.self_distillation.max_reprompt_len)
        if model:
            trainer_io.init_single_rank()
            teacher = trainer_io.load_model(self.cfg.actor_rollout_ref.model.path, self.cfg, train=False)
            self.objectives = objectives.Objectives(trainer_io.make_actor(self.cfg, teacher, teacher), [])
        self.eos = self.tokenizer.eos_token_id
        self._items = {}

    def item(self, index: int) -> dict:
        if index not in self._items:
            self._items = {index: self.io.prompt_inputs(self.dataset, index, self.processor, self.cfg)}
        return self._items[index]

    def base(self, index: int, response_ids: list[int]):
        """DataProto of one trajectory with the response padded to its own length."""
        self.short.actor_rollout_ref.rollout.response_length = len(response_ids)
        it = self.item(index)
        return self.io.assemble_batch([{**it, "response_ids": response_ids, "uid": "0"}], self.tokenizer,
                                      self.processor, self.short)

    def view_input(self, b, view_path: str):
        """The x-plus teacher input (CPU DataProto) of trajectory b with the view image, and its prompt tokens."""
        b.non_tensor_batch["bbox_images"][0] = [{"path": view_path}]
        v = self.io.teacher_views(b, self.plus, self.tokenizer, self.processor)
        n_prompt = int(v.batch["teacher_response_start_idx"][0])
        if n_prompt >= self.max_len:
            raise SystemExit(f"STOP: teacher prompt of {view_path} has {n_prompt} tokens (max_reprompt_len {self.max_len})")
        return v, n_prompt

    def logp(self, v, responses, view_path: str, position: int, ids: list[int]) -> list[float]:
        """Log probabilities of the tokens ids at response position `position` of the x-plus teacher input v."""
        import torch

        v = self.io.to_device(v, "cuda")
        mi = {"responses": responses.cuda(), **v.batch, **v.non_tensor_batch}
        with torch.no_grad():
            lp = self.objectives._forward(self.obj.view_inputs(mi, "plus"), grad=False,
                                          return_all_logps=True)["all_logps"][0, position]
        out = [float(lp[i]) for i in ids]
        if not all(math.isfinite(x) for x in out):
            raise SystemExit(f"STOP: non-finite letter log probability {out} for {view_path} (stop condition 3)")
        return out


class Builder:
    """Worker side of the prefetch: the CPU inputs of one fresh row or one prefix trajectory, all views."""

    def __init__(self, repo: str, command_txt: str):
        setup(repo)
        self.t = Teacher(repo, command_txt, model=False)

    def views(self, b, view_paths: list[list[str]]) -> list:
        import prefetch

        out = []
        for name, path in view_paths:
            v, n = self.t.view_input(b, path)
            out.append([name, path, prefetch.parts(v), n])
        return out

    def fresh(self, r: int, view_paths: list[list[str]]) -> dict:
        import letters

        t = self.t
        row = t.item(r)["row"]
        lets = letters.option_letters(prompt_text(row["raw_prompt"]))
        b = t.base(r, [t.eos])
        return {"row": r, "bbox_paths": image_paths(row["bbox_images"]), "gt": row["reward_model"]["ground_truth"],
                "letters": lets, "letter_ids": letters.fresh_ids(t.tokenizer, lets),
                "responses": b.batch["responses"], "views": self.views(b, view_paths)}

    def prefix(self, ro: dict, removed: bool, view_paths: list[list[str]]) -> dict:
        import letters

        t = self.t
        idx = ro["dataset_index"]
        row = t.item(idx)["row"]
        lets = letters.option_letters(prompt_text(row["raw_prompt"]))
        found = letters.prefix_ids(t.obj, t.tokenizer, ro["response_ids"], ro["response_text"], lets)
        res = {"traj": ro["traj"], "letters": lets, "found": found, "views": None}
        if not removed and found is not None:
            b = t.base(idx, ro["response_ids"][:found[0] + 1])
            res.update(responses=b.batch["responses"], views=self.views(b, view_paths))
        return res


def pool_or_none(repo: str, command_txt: str):
    import prefetch

    n = prefetch.workers()
    if n <= 0:
        return None, 0.0
    pool = prefetch.Pool([str(Path(repo).resolve()), str(HERE.parent / "vad_grad3"), str(HERE)], "stage1", "Builder",
                         (repo, command_txt), n)
    return pool, float(os.environ.get("PREFETCH_BUDGET", "160"))


def done_keys(path: Path, key: str) -> set:
    return {json.loads(x)[key] for x in open(path)} if path.exists() else set()


def stopper():
    stop_after = int(os.environ.get("STOP_AFTER", "0"))
    count = [0]

    def tick():
        count[0] += 1
        if stop_after and count[0] == stop_after:
            print(f"forced stop after {stop_after} lines", flush=True)
            sys.exit(STOP_EXIT)
    return tick


def built(t: Teacher, pool, budget: float, jobs: list):
    """(result, build seconds, wait seconds) of each job (cost, method, args) in order: from the worker pool, or built
    in this process (pool None; the wait is the build time)."""
    if pool is not None:
        yield from pool.ordered(jobs, budget)
        pool.close()
        return
    local = Builder.__new__(Builder)
    local.t = t
    for _, method, args in jobs:
        t0 = time.time()
        res = getattr(local, method)(*args)
        yield res, time.time() - t0, time.time() - t0


def view_paths(man: dict, r: int) -> list[list[str]]:
    return [[name, v["path"]] for name, v in man[r]["views"].items()]


def view_logps(t: Teacher, res: dict, position: int, ids: list[int]) -> tuple[dict, dict]:
    import prefetch

    logp, ntok = {}, {}
    for name, path, p, n in res["views"]:
        logp[name], ntok[name] = t.logp(prefetch.proto(p), res["responses"], path, position, ids), n
    return logp, ntok


def fresh(argv: list[str]):
    repo, command_txt, views_dir, out = argv[:4]
    setup(repo)
    import views

    man, removed = load_views(views_dir)
    rows = sorted(r for r in man if r not in removed)
    if len(argv) > 4:
        keep = set(json.loads(Path(argv[4]).read_text())["rows"])
        rows = [r for r in rows if r in keep]
    out = Path(out)
    done = done_keys(out, "row")
    pool, budget = pool_or_none(repo, command_txt)
    t = Teacher(repo, command_txt)
    tick = stopper()
    jobs = [(len(man[r]["views"]), "fresh", (r, view_paths(man, r))) for r in rows if r not in done]
    with open(out, "a") as f:
        for res, build, wait in built(t, pool, budget, jobs):
            t0 = time.time()
            r = res["row"]
            # join check: dataset index r holds the released teacher image of manifest row r
            if res["bbox_paths"] != [man[r]["views"][views.V0]["path"]]:
                raise SystemExit(f"STOP: row {r}: the dataset teacher image differs from the manifest v0")
            logp, ntok = view_logps(t, res, 0, res["letter_ids"])
            f.write(json.dumps({"row": r, "gt": res["gt"], "letters": res["letters"],
                                "letter_ids": res["letter_ids"], "logp": logp, "teacher_prompt_tokens": ntok,
                                "seconds": round(time.time() - t0 + wait, 3), "wait_seconds": round(wait, 3),
                                "build_seconds": round(build, 3)}) + "\n")
            f.flush()
            tick()


def prefix(argv: list[str]):
    repo, command_txt, views_dir, rollouts_path, out = argv[:5]
    setup(repo)

    man, removed = load_views(views_dir)
    ros = [json.loads(x) for x in open(rollouts_path)]
    if len(argv) > 5:
        keep = set(json.loads(Path(argv[5]).read_text())["trajs"])
        ros = [r for r in ros if r["traj"] in keep]
    ros.sort(key=lambda r: (r["dataset_index"], r["traj"]))
    out = Path(out)
    done = done_keys(out, "traj")
    pool, budget = pool_or_none(repo, command_txt)
    t = Teacher(repo, command_txt)
    tick = stopper()
    todo = [ro for ro in ros if ro["traj"] not in done]
    jobs = [(len(man[ro["dataset_index"]]["views"]), "prefix",
             ({k: ro[k] for k in ("traj", "dataset_index", "response_ids", "response_text")},
              ro["dataset_index"] in removed, view_paths(man, ro["dataset_index"]))) for ro in todo]
    with open(out, "a") as f:
        for ro, (res, build, wait) in zip(todo, built(t, pool, budget, jobs)):
            t0 = time.time()
            if res["traj"] != ro["traj"]:
                raise RuntimeError(f"prefetch order: trajectory {res['traj']}, expected {ro['traj']}")
            idx = ro["dataset_index"]
            rec = {"traj": ro["traj"], "prompt_pos": ro["prompt_pos"], "dataset_index": idx, "sibling": ro["sibling"],
                   "correct": ro["correct"], "letter": ro["letter"], "gt": ro["gt"], "flag": "fx", "logp": None,
                   "letters": res["letters"]}
            found = res["found"]
            if idx in removed:
                rec["flag"] = "removed"
            elif found is not None:
                pos, sampled, ids = found
                logp, _ = view_logps(t, res, pos, ids)
                rec.update(flag="ok", position=pos, rule_letter=sampled, letter_ids=ids, logp=logp)
            rec.update(seconds=round(time.time() - t0 + wait, 3), wait_seconds=round(wait, 3),
                       build_seconds=round(build, 3))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            tick()


def main():
    {"fresh": fresh, "prefix": prefix}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
