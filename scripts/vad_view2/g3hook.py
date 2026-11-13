"""Ray worker setup hook of the check runs of check_view2.sh (gates G2 and G3, HANDOFF-VLM-VAD-VIEW-002-R1).

check_view2.sh copies this file to <out>/g3py/vad_view2_g3.py and puts that folder on PYTHONPATH, so that the same hook
also loads in the run of the exp/VLM-VAD-PACKFIX-001 code (a git archive without scripts/vad_view2). setup():
1. scripts.vad_packfix.hostmem.setup() (the hook of the training runs, unchanged).
2. Replay (env VAD_G3_REPLAY=<file>): AgentLoopManager.generate_sequences. When the file does not exist, the call
   generates as before and saves the output (pickle) with the row key of each request row (the released teacher
   image path, unique for each data row). When the file exists, the call returns the saved output: for a request
   with fewer rows (rollout.n 1, or the N1V8 generation) every k-th saved row, k = saved rows / request rows. The row
   keys of the request must equal the keys of the rows used. The non-tensor keys that the request also has (uid and
   the other request columns) come from the request. So each check run trains on the same responses.
3. Gradient dump (env VAD_G3_GRAD=<file>): before the first actor optimizer step (before the clip), the gradient of
   each actor parameter in fp32, in named_parameters order, in one raw file, and <file>.json with the names, sizes,
   offsets and the squared norm.
"""

from __future__ import annotations

import functools
import json
import os
import pickle


def _row_keys(batch) -> list[str]:
    out = []
    for item in batch.non_tensor_batch["bbox_images"]:
        items = list(item)
        out.append(str(items[0]["path"]))
    return out


def _patch_replay(path: str):
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    orig = AgentLoopManager.generate_sequences

    @functools.wraps(orig)
    def generate_sequences(self, prompts):
        keys = _row_keys(prompts)
        if not os.path.exists(path):
            output = orig(self, prompts)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"keys": keys, "output": output}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            print(f"VAD_G3 replay saved {len(keys)} rows to {path}", flush=True)
            return output
        with open(path, "rb") as f:
            saved = pickle.load(f)
        n_saved, n_req = len(saved["keys"]), len(keys)
        if n_saved % n_req:
            raise RuntimeError(f"VAD_G3 replay: {n_saved} saved rows, {n_req} requested")
        k = n_saved // n_req
        idx = [i * k for i in range(n_req)]
        if [saved["keys"][i] for i in idx] != keys:
            raise RuntimeError("VAD_G3 replay: the requested rows differ from the saved rows")
        output = saved["output"] if k == 1 else saved["output"].select_idxs(idx)
        for key, val in prompts.non_tensor_batch.items():
            if key in output.non_tensor_batch:
                output.non_tensor_batch[key] = val
        print(f"VAD_G3 replay: {n_req} rows (every {k}th of {n_saved}) from {path}", flush=True)
        return output

    AgentLoopManager.generate_sequences = generate_sequences


def _patch_grad_dump(path: str):
    import torch
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    orig = DataParallelPPOActor._optimizer_step

    @functools.wraps(orig)
    def _optimizer_step(self):
        if not os.path.exists(path + ".json"):
            index, off, sq = [], 0, 0.0
            with open(path + ".tmp", "wb") as f:
                for name, p in self.actor_module.named_parameters():
                    if p.grad is None:
                        index.append({"name": name, "numel": 0, "offset": off, "grad": None})
                        continue
                    g = p.grad.detach().float().cpu()
                    sq += float(g.double().pow(2).sum())
                    f.write(g.numpy().tobytes())
                    index.append({"name": name, "numel": g.numel(), "offset": off, "shape": list(g.shape)})
                    off += g.numel()
            os.replace(path + ".tmp", path)
            with open(path + ".json", "w") as f:
                json.dump({"params": index, "elements": off, "sq_norm": sq}, f)
            print(f"VAD_G3 gradient dump: {off} elements, norm {sq ** 0.5:.6e}, {path}", flush=True)
            torch.cuda.empty_cache()
        return orig(self)

    DataParallelPPOActor._optimizer_step = _optimizer_step


def setup():
    from scripts.vad_packfix import hostmem

    hostmem.setup()
    if os.environ.get("VAD_G3_REPLAY"):
        _patch_replay(os.environ["VAD_G3_REPLAY"])
    if os.environ.get("VAD_G3_GRAD"):
        _patch_grad_dump(os.environ["VAD_G3_GRAD"])
    print(f"VAD_G3 hook installed pid={os.getpid()} replay={os.environ.get('VAD_G3_REPLAY')} "
          f"grad={os.environ.get('VAD_G3_GRAD')}", flush=True)
