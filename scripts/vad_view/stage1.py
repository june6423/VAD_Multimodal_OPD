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

Usage: python stage1.py fresh <repo> <COMMAND_B200.txt> <views dir> <FRESH.jsonl> [<rows file>]
       python stage1.py prefix <repo> <COMMAND_B200.txt> <views dir> <rollouts.jsonl> <PREFIX_step<s>.jsonl> [<traj file>]
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
    def __init__(self, repo: str, command_txt: str):
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

    def letter_logp(self, b, view_path: str, position: int, ids: list[int]) -> tuple[list[float], int]:
        """Log probabilities of the tokens ids at response position `position` of the x-plus teacher with the view."""
        import torch

        b.non_tensor_batch["bbox_images"][0] = [{"path": view_path}]
        v = self.io.teacher_views(b, self.plus, self.tokenizer, self.processor)
        n_prompt = int(v.batch["teacher_response_start_idx"][0])
        if n_prompt >= self.max_len:
            raise SystemExit(f"STOP: teacher prompt of {view_path} has {n_prompt} tokens (max_reprompt_len {self.max_len})")
        v = self.io.to_device(v, "cuda")
        mi = {"responses": b.batch["responses"].cuda(), **v.batch, **v.non_tensor_batch}
        with torch.no_grad():
            lp = self.objectives._forward(self.obj.view_inputs(mi, "plus"), grad=False,
                                          return_all_logps=True)["all_logps"][0, position]
        out = [float(lp[i]) for i in ids]
        if not all(math.isfinite(x) for x in out):
            raise SystemExit(f"STOP: non-finite letter log probability {out} for {view_path} (stop condition 3)")
        return out, n_prompt


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


def fresh(argv: list[str]):
    repo, command_txt, views_dir, out = argv[:4]
    setup(repo)
    import letters
    import views

    man, removed = load_views(views_dir)
    rows = sorted(r for r in man if r not in removed)
    if len(argv) > 4:
        keep = set(json.loads(Path(argv[4]).read_text())["rows"])
        rows = [r for r in rows if r in keep]
    out = Path(out)
    done = done_keys(out, "row")
    t = Teacher(repo, command_txt)
    tick = stopper()
    with open(out, "a") as f:
        for r in rows:
            if r in done:
                continue
            t0 = time.time()
            it = t.item(r)
            row = it["row"]
            # join check: dataset index r holds the released teacher image of manifest row r
            if image_paths(row["bbox_images"]) != [man[r]["views"][views.V0]["path"]]:
                raise SystemExit(f"STOP: row {r}: the dataset teacher image differs from the manifest v0")
            lets = letters.option_letters(prompt_text(row["raw_prompt"]))
            ids = letters.fresh_ids(t.tokenizer, lets)
            b = t.base(r, [t.eos])
            logp, ntok = {}, {}
            for name, v in man[r]["views"].items():
                logp[name], ntok[name] = t.letter_logp(b, v["path"], 0, ids)
            f.write(json.dumps({"row": r, "gt": row["reward_model"]["ground_truth"], "letters": lets,
                                "letter_ids": ids, "logp": logp, "teacher_prompt_tokens": ntok,
                                "seconds": round(time.time() - t0, 3)}) + "\n")
            f.flush()
            tick()


def prefix(argv: list[str]):
    repo, command_txt, views_dir, rollouts_path, out = argv[:5]
    setup(repo)
    import letters

    man, removed = load_views(views_dir)
    ros = [json.loads(x) for x in open(rollouts_path)]
    if len(argv) > 5:
        keep = set(json.loads(Path(argv[5]).read_text())["trajs"])
        ros = [r for r in ros if r["traj"] in keep]
    ros.sort(key=lambda r: (r["dataset_index"], r["traj"]))
    out = Path(out)
    done = done_keys(out, "traj")
    t = Teacher(repo, command_txt)
    tick = stopper()
    with open(out, "a") as f:
        for ro in ros:
            if ro["traj"] in done:
                continue
            t0 = time.time()
            idx = ro["dataset_index"]
            rec = {"traj": ro["traj"], "prompt_pos": ro["prompt_pos"], "dataset_index": idx, "sibling": ro["sibling"],
                   "correct": ro["correct"], "letter": ro["letter"], "gt": ro["gt"], "flag": "fx", "logp": None}
            row = t.item(idx)["row"]
            lets = letters.option_letters(prompt_text(row["raw_prompt"]))
            rec["letters"] = lets
            found = letters.prefix_ids(t.obj, t.tokenizer, ro["response_ids"], ro["response_text"], lets)
            if idx in removed:
                rec["flag"] = "removed"
            elif found is not None:
                pos, sampled, ids = found
                b = t.base(idx, ro["response_ids"][:pos + 1])
                logp = {name: t.letter_logp(b, v["path"], pos, ids)[0] for name, v in man[idx]["views"].items()}
                rec.update(flag="ok", position=pos, rule_letter=sampled, letter_ids=ids, logp=logp)
            rec["seconds"] = round(time.time() - t0, 3)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            tick()


def main():
    {"fresh": fresh, "prefix": prefix}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
