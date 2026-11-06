"""Data-equality check of hostmem.py on b200 (DIRECTIVE-VLM-VAD-PACKFIX-001-003 action 3). Check only, no training.

Ray loads setup() in each worker (HOSTMEM_CHECK_OUT set). setup() installs hostmem.py and then wraps
AgentLoopWorker.generate_sequences. Validation batches run unchanged. For the first training batch, each agent
worker takes its first HOSTMEM_CHECK_GROUPS prompt groups and runs the real agent loops and _postprocess twice, with a
fixed response of response_length tokens in place of the vLLM generation:
  (a) with the original verl methods (hostmem off),
  (b) with hostmem on.
It writes the sha256 of each field of each row for (a), for (b), and for (b) after a pickle round trip of the output
DataProto (as Ray sends it), then stops the run by design.

python hostmem_check.py <output_root> <HOSTMEM_CHECK.json> combines the worker files. Exit 1 at any hash
difference or a missing file.
"""

import asyncio
import functools
import glob
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

OUT = os.environ.get("HOSTMEM_CHECK_OUT")
GROUPS = int(os.environ.get("HOSTMEM_CHECK_GROUPS", "4"))


def _sha_tensor(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() + str(tuple(t.shape))


def _sha_obj(v):
    if isinstance(v, torch.Tensor):
        return _sha_tensor(v)
    if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
        return {k: _sha_tensor(x) for k, x in sorted(v.items())}
    return hashlib.sha256(pickle.dumps(v, protocol=5)).hexdigest()


def digest(dp):
    """{field: [hash of row 0, row 1, ...]} for the tensor batch and the non-tensor batch of a DataProto."""
    out = {}
    for k, t in dp.batch.items():
        out["batch." + k] = [_sha_tensor(t[i]) for i in range(t.shape[0])]
    for k, arr in dp.non_tensor_batch.items():
        out["non_tensor." + k] = [_sha_obj(arr[i]) for i in range(len(arr))]
    return out


class _FakeServerManager:
    def __init__(self, token_id, length):
        self.token_id, self.length = token_id, length

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        from verl.workers.rollout.replica import TokenOutput

        return TokenOutput(token_ids=[self.token_id] * self.length, log_probs=[0.0] * self.length)


def wait_for_workers(out, n_workers, timeout=900, poll=5):
    """Wait until n_workers worker files exist in out, or until the timeout. Return the count of worker files."""
    deadline = time.monotonic() + timeout
    while True:
        count = len(glob.glob(os.path.join(out, "hostmem_check_*.json")))
        if count >= n_workers or time.monotonic() >= deadline:
            return count
        time.sleep(poll)


async def _run_rows(self, sub, use_hostmem):
    from scripts.vad_packfix import hostmem
    from verl.experimental.agent_loop.agent_loop import get_trajectory_info

    config = self.config.actor_rollout_ref.rollout
    sampling_params = dict(temperature=config.temperature, top_p=config.top_p, repetition_penalty=1.0,
                           logprobs=config.calculate_log_probs)
    ntb = sub.non_tensor_batch
    if "agent_name" not in ntb:
        ntb["agent_name"] = np.array([config.agent.default_agent_loop] * len(sub), dtype=object)
    index = ntb["index"] if "index" in ntb else np.arange(len(sub))
    traj = await get_trajectory_info(-1, index.tolist(), False)
    cache = {"pixel_values": {}, "vision": {}, "shared": 0, "unique": 0, "vision_hits": 0} if use_hostmem else None
    token = hostmem._CACHE.set(cache)
    try:
        tasks = []
        for i in range(len(sub)):
            kwargs = {k: v[i] for k, v in ntb.items()}
            kwargs["__validate__"] = False
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, traj[i], trace=False, **kwargs)))
        outs = await asyncio.gather(*tasks)
    finally:
        hostmem._CACHE.reset(token)
    return self._postprocess(outs), cache


async def _check(self, batch, originals, patched):
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopWorker

    pid = os.getpid()
    n = int(self.config.actor_rollout_ref.rollout.n)
    sub = batch[0: GROUPS * n]
    self.server_manager = _FakeServerManager(self.tokenizer.encode(" the", add_special_tokens=False)[0],
                                             self._get_response_length(validate=False))

    async def _no_score(*args, **kwargs):
        return None

    self._compute_score = _no_score
    results = {}
    for mode, use, methods in [("original", False, originals), ("hostmem", True, patched)]:
        AgentLoopWorker._compute_multi_modal_inputs = methods["mmi"]
        AgentLoopBase.process_vision_info = methods["vision"]
        dp, cache = await _run_rows(self, sub, use)
        results[mode] = (dp, cache)
    dp_fixed, cache = results["hostmem"]
    d_orig = digest(results["original"][0])
    d_fixed = digest(dp_fixed)
    d_trip = digest(pickle.loads(pickle.dumps(dp_fixed, protocol=5)))
    mmi = dp_fixed.non_tensor_batch["multi_modal_inputs"]
    shared_rows = sum(1 for i in range(len(mmi)) if i % n and mmi[i]["pixel_values"] is mmi[i - i % n]["pixel_values"])
    diffs = []
    for name, d in [("hostmem", d_fixed), ("hostmem_pickle_round_trip", d_trip)]:
        if sorted(d) != sorted(d_orig):
            diffs.append({"run": name, "fields": sorted(set(d) ^ set(d_orig))})
        for k in d_orig:
            for i, (a, b) in enumerate(zip(d_orig[k], d.get(k, []))):
                if a != b:
                    diffs.append({"run": name, "field": k, "row": i})
    summary = {
        "pid": pid,
        "rows": len(sub),
        "groups": GROUPS,
        "rollout_n": n,
        "fields": sorted(d_orig),
        "row_hashes_compared": 2 * sum(len(v) for v in d_orig.values()),
        "differences": len(diffs),
        "first_differences": diffs[:20],
        "rows_with_shared_pixel_values": shared_rows,
        "hostmem_cache": {k: cache[k] for k in ("shared", "unique", "vision_hits")},
        "original_row_hashes": d_orig,
    }
    with open(os.path.join(OUT, f"hostmem_check_{pid}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"HOSTMEM_CHECK pid={pid} rows={len(sub)} differences={len(diffs)} shared_rows={shared_rows}", flush=True)
    # TASK-VLM-VAD-PACKFIX-001-012: the first worker that raised stopped the run before the other workers wrote.
    n_workers = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
    await asyncio.to_thread(wait_for_workers, OUT, n_workers)
    raise RuntimeError("HOSTMEM_CHECK: check done; the check run stops before the step-1 generation by design")


def setup():
    if not OUT:
        return
    from scripts.vad_packfix import hostmem
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopWorker

    originals = {"mmi": AgentLoopWorker._compute_multi_modal_inputs, "vision": AgentLoopBase.process_vision_info}
    hostmem.setup()
    patched = {"mmi": AgentLoopWorker._compute_multi_modal_inputs, "vision": AgentLoopBase.process_vision_info}
    hostmem_generate = AgentLoopWorker.generate_sequences

    @functools.wraps(hostmem_generate)
    async def generate_sequences(self, batch):
        if batch.meta_info.get("validate", False):
            return await hostmem_generate(self, batch)
        await _check(self, batch, originals, patched)

    AgentLoopWorker.generate_sequences = generate_sequences
    print(f"HOSTMEM_CHECK hook installed pid={os.getpid()}", flush=True)


def main(out, dest):
    files = sorted(glob.glob(os.path.join(out, "hostmem_check_*.json")))
    if not files:
        print("no hostmem_check_<pid>.json file")
        return 1
    workers = [json.load(open(p)) for p in files]
    for w in workers:
        w.pop("original_row_hashes")
    summary = {
        "workers": len(workers),
        "rows": sum(w["rows"] for w in workers),
        "row_hashes_compared": sum(w["row_hashes_compared"] for w in workers),
        "differences": sum(w["differences"] for w in workers),
        "rows_with_shared_pixel_values": sum(w["rows_with_shared_pixel_values"] for w in workers),
        "per_worker": workers,
    }
    with open(dest, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_worker"}))
    return 0 if summary["differences"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
