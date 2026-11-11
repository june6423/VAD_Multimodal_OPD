"""PILOT=1 helpers of run_view.sh (HANDOFF-VLM-VAD-VIEW-001-R2 Part A item 10).

select: the pilot inputs.
  PILOT_FRESH.json rows: the row with the largest box (bbox area), the row with the smallest box, the row with the
    longest prompt (tokens of the prompt text), then 37 random rows (random.Random(20261111), other rows).
  PILOT_PREFIX_step<s>.json trajs: 20 random trajectories of each step (random.Random(20261111 + s)).
  PILOT_BUILD.json rows: the rows of the views build: the fresh rows, the rows of the prefix trajectories and the
    dataset rows of the stage 2 pilot prompts (PILOT_PROMPTS.json of vad_grad3 pilot.py select, each checkpoint).
project: the FULL wall time and the peak disk from the pilot outputs (PROJECTION.json). Each stage of TIMES.jsonl:
  FULL seconds = fixed part (pilot stage seconds - pilot work seconds: model loads, setup) + FULL work seconds.
  Work: views build: all seconds x 6241 / build rows. fresh: mean seconds per FRESH line x 6241. prefix_step<s>: mean
    seconds per line x the trajectories of the step's rollouts.jsonl. stats1: seconds x 6241 / fresh rows.
    vgrads_<ck>: the largest seconds per token of the pilot views (inputs, forward, backward, scale-3 backward of the
    progress files) x the FULL B8 tokens (rows_full/) x 8 views + the mean batch-end seconds x 32.
    heldout_<ck>: mean seconds per set-A item x 500 + per set-B item x 500 (HELDOUT_VALUES.jsonl).
    gram_<ck>: seconds x (32 + 8 + 2) / pilot vectors. linesearch_<ck>: mean seconds per item line x (2 + 9 x 3) points
    x 1000 items + the pilot direction setup seconds (stage - lines - model load) / pilot directions x 9.
    Other stages (views fit, held-out select, displacement, rows, remove, stats2): the pilot seconds.
  Peak disk (one checkpoint at a time; vgram.py remove runs before the next checkpoint): (32 + 8 vectors + 9 fold means
    + 2 partial accumulators + 4 displacements) x 4 x elements bytes, plus the view images of the pilot x 6241 / rows.

Usage: python vpilot.py select <train_vad.parquet> <train.jsonl> <model dir> <rollouts dir> <M0 PILOT_PROMPTS.json>
         <A1_80 PILOT_PROMPTS.json> <out dir>
       python vpilot.py project <out dir> <rollouts dir> <PROJECTION.json>
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261111
RANDOM_ROWS, PREFIX_TRAJS = 37, 20
STEPS = (0, 80, 130)
FULL_ROWS = 6241
ELEMENTS = 3569090560
FULL_VIEWS, FOLDS, PARTS = 8, 4, 4
FULL_HELDOUT = {"A": 500, "B": 500}
FULL_POINTS = 2 + 9 * 3
FULL_DIRECTIONS = 9
CHECKPOINTS = ("M0", "A1_80")


def box_area(bbox) -> int:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def pick_fresh(rows: list[dict], prompt_tokens: dict[int, int]) -> list[int]:
    """rows: row, bbox. The largest box, the smallest box, the longest prompt, then RANDOM_ROWS random other rows."""
    ids = [r["row"] for r in rows]
    area = {r["row"]: box_area(r["bbox"]) for r in rows}
    first = []
    for r in (max(ids, key=lambda i: (area[i], -i)), min(ids, key=lambda i: (area[i], i)),
              max(ids, key=lambda i: (prompt_tokens[i], -i))):
        if r not in first:
            first.append(r)
    rest = [i for i in ids if i not in first]
    return first + sorted(random.Random(SEED).sample(rest, RANDOM_ROWS))


def pick_prefix(trajs: list[int], step: int) -> list[int]:
    return sorted(random.Random(SEED + step).sample(sorted(trajs), PREFIX_TRAJS))


def select(argv: list[str]):
    parquet, train_jsonl, model_dir, rollouts_dir, pp_m0, pp_a1, out = argv
    sys.path.insert(0, str(HERE))
    import views
    from transformers import AutoTokenizer

    out = Path(out)
    rows = views.load_rows(parquet, train_jsonl)
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompt_tokens = {r["row"]: len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in rows}
    fresh = pick_fresh(rows, prompt_tokens)
    build = set(fresh)
    prefix = {}
    pos_to_row = {}
    for s in STEPS:
        ros = [json.loads(x) for x in open(Path(rollouts_dir) / f"step{s}" / "rollouts.jsonl")]
        prefix[s] = pick_prefix([r["traj"] for r in ros], s)
        by = {r["traj"]: r for r in ros}
        build |= {by[t]["dataset_index"] for t in prefix[s]}
        if s in (0, 80):
            pos_to_row[s] = {r["prompt_pos"]: r["dataset_index"] for r in ros}
    stage2 = {}
    for ck, s, pp in (("M0", 0, pp_m0), ("A1_80", 80, pp_a1)):
        stage2[ck] = sorted(pos_to_row[s][p] for p in json.loads(Path(pp).read_text())["prompts"])
        build |= set(stage2[ck])
    (out / "PILOT_FRESH.json").write_text(json.dumps({"rows": fresh, "seed": SEED, "first": fresh[:3],
                                                      "prompt_tokens_first": [prompt_tokens[i] for i in fresh[:3]],
                                                      "box_area_first": [box_area(rows[i]["bbox"]) for i in fresh[:3]]},
                                                     indent=1))
    for s in STEPS:
        (out / f"PILOT_PREFIX_step{s}.json").write_text(json.dumps({"trajs": prefix[s], "seed": SEED + s}))
    (out / "PILOT_BUILD.json").write_text(json.dumps({"rows": sorted(build), "stage2_rows": stage2}, indent=1))
    print(json.dumps({"fresh": len(fresh), "build": len(build)}))


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in open(path)] if path.exists() else []


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def view_rates(ck_dir: Path) -> dict[str, float]:
    """Seconds per token of each pilot view key from the progress files."""
    tok, sec = {}, {}
    for p in sorted(ck_dir.glob("progress_O__B8-f*@*.json")):
        view = p.stem.split("@", 1)[1]
        for m in json.loads(p.read_text())["micro"]:
            tok[view] = tok.get(view, 0) + m["tokens"]
            sec[view] = sec.get(view, 0.0) + (m["seconds_inputs"] + m["seconds_forward"] + m["seconds_backward"]
                                              + m["seconds_rounding"])
    return {v: sec[v] / tok[v] for v in tok}


def full_b8_tokens(rows_full: Path) -> int:
    return sum(json.loads(x)["total_tokens"] for f in range(1, FOLDS + 1) for x in open(rows_full / f"ROWS_B8-f{f}.jsonl"))


def stage_seconds(out: Path) -> dict[str, float]:
    """Seconds of each stage name of TIMES.jsonl; a forced-stop stage stop_<name> counts for <name>."""
    seconds: dict[str, float] = {}
    for t in lines(out / "TIMES.jsonl"):
        name = t["stage"][len("stop_"):] if t["stage"].startswith("stop_") else t["stage"]
        seconds[name] = seconds.get(name, 0.0) + t["seconds"]
    return seconds


def scaled(stage: float, pilot_work: float, full_work: float) -> dict:
    """FULL seconds of a stage = its fixed part (pilot stage seconds - pilot work seconds) + the FULL work seconds."""
    fixed = max(stage - pilot_work, 0.0)
    return {"pilot_stage": stage, "pilot_work": pilot_work, "fixed": fixed, "full_work": full_work,
            "full": fixed + full_work}


def project(argv: list[str]):
    out, rollouts_dir, dest = Path(argv[0]), Path(argv[1]), Path(argv[2])
    sec = stage_seconds(out)
    build = json.loads((out / "PILOT_BUILD.json").read_text())["rows"]
    fresh_rows = len(json.loads((out / "PILOT_FRESH.json").read_text())["rows"])
    st: dict[str, dict] = {}
    st["views_fit"] = scaled(sec["views_fit"], 0.0, 0.0)
    st["views_build"] = scaled(sec["views_build"], sec["views_build"], sec["views_build"] * FULL_ROWS / len(build))
    fresh = lines(out / "stage1" / "FRESH.jsonl")
    work = sum(x["seconds"] for x in fresh)
    st["fresh"] = scaled(sec["fresh"], work, work / len(fresh) * FULL_ROWS)
    for s in STEPS:
        pre = lines(out / "stage1" / f"PREFIX_step{s}.jsonl")
        n_full = sum(1 for _ in open(rollouts_dir / f"step{s}" / "rollouts.jsonl"))
        work = sum(x["seconds"] for x in pre)
        st[f"prefix_step{s}"] = scaled(sec[f"prefix_step{s}"], work, work / len(pre) * n_full)
    st["stats1"] = scaled(sec["stats1"], sec["stats1"], sec["stats1"] * FULL_ROWS / fresh_rows)
    for name in ("heldout_select", "displacement", "stats2"):
        st[name] = scaled(sec[name], 0.0, 0.0)
    for ck in CHECKPOINTS:
        d = out / ck
        st[f"rows_{ck}"] = scaled(sec[f"rows_{ck}"], 0.0, 0.0)
        rates = view_rates(d)
        b8 = full_b8_tokens(d / "rows_full")
        prog = [json.loads(p.read_text()) for p in sorted(d.glob("progress_O__B8-f*@*.json"))]
        micro = sum(m["seconds_inputs"] + m["seconds_forward"] + m["seconds_backward"] + m["seconds_rounding"]
                    for pr in prog for m in pr["micro"])
        ends = [r["seconds_end"] for run in lines(d / "runs.jsonl") for r in run.get("results", []) if "seconds_end" in r]
        st[f"vgrads_{ck}"] = {**scaled(sec[f"vgrads_{ck}"], micro + sum(ends),
                                       max(rates.values()) * b8 * FULL_VIEWS + mean(ends) * FULL_VIEWS * FOLDS),
                              "rates_seconds_per_token": rates, "full_b8_tokens": b8}
        hv = lines(d / "HELDOUT_VALUES.jsonl")
        st[f"heldout_{ck}"] = scaled(sec[f"heldout_{ck}"], sum(x["seconds"] for x in hv),
                                     sum(mean([x["seconds"] for x in hv if x["set"] == s]) * n
                                         for s, n in FULL_HELDOUT.items()))
        n_pilot = len(json.loads((d / f"GRAM_{ck}.json").read_text())["names"])
        st[f"gram_{ck}"] = scaled(sec[f"gram_{ck}"], sec[f"gram_{ck}"],
                                  sec[f"gram_{ck}"] * (FULL_VIEWS * FOLDS + 2 * PARTS + 2) / n_pilot)
        ls_lines = [x for p in sorted((d / "linesearch" / "points").glob("*.jsonl")) for x in lines(p)]
        ls_load = max(r["load_seconds"] for r in lines(d / "linesearch" / "runs.jsonl") if "load_seconds" in r)
        work = sum(x["seconds"] for x in ls_lines)
        n_dir = len(list((d / "linesearch" / "points").glob("*.direction.json")))
        setup = max(sec[f"linesearch_{ck}"] - work - ls_load, 0.0)
        st[f"linesearch_{ck}"] = {**scaled(sec[f"linesearch_{ck}"], work + setup,
                                           work / len(ls_lines) * FULL_POINTS * sum(FULL_HELDOUT.values())
                                           + setup / n_dir * FULL_DIRECTIONS),
                                  "load_seconds": ls_load, "setup_seconds_pilot": setup, "pilot_directions": n_dir}
        st[f"remove_{ck}"] = scaled(sec[f"remove_{ck}"], 0.0, 0.0)
    missing = sorted(set(sec) - set(st))
    res = {"elements": ELEMENTS, "stages": st, "stages_not_projected": {k: sec[k] for k in missing}}
    res["total_hours"] = (sum(x["full"] for x in st.values()) + sum(sec[k] for k in missing)) / 3600
    res["stage1_hours"] = sum(st[k]["full"] for k in st if k.startswith(("views", "fresh", "prefix", "stats1"))) / 3600
    images = sum(p.stat().st_size for p in (out / "views" / "images").rglob("*.png"))
    res["views_images_bytes_full"] = images * FULL_ROWS / len(build)
    res["peak_disk_tb"] = ((FULL_VIEWS * FOLDS + 2 * PARTS + 9 + 2 + 4) * 4 * ELEMENTS
                           + res["views_images_bytes_full"]) / 1e12
    res["gate_96h"] = res["total_hours"] <= 96
    res["gate_4tb"] = res["peak_disk_tb"] < 4
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("total_hours", "stage1_hours", "peak_disk_tb", "gate_96h", "gate_4tb")}))


def main():
    {"select": select, "project": project}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
