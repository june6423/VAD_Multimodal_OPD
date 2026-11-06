"""Diagnosis of the reference release of hostmem.py (DT STATUS on TASK-VLM-VAD-PACKFIX-001-012, item 2.2). Check only.

TASK-VLM-VAD-PACKFIX-001-014: the release freed the module object, but the RSS fell only 60.7 -> 57.6 GiB and the
pinned host bytes in use stayed 21.5 GiB. This Ray worker hook (REFDIAG_OUT set) installs hostmem.py and wraps
hostmem.release_reference:
  1. Before the release: for each parameter of the reference, actor and teacher modules, the storage pointer, the
     storage bytes, the dtype and is_pinned, and the FSDP2 mixed-precision policy (DT ANSWER on TASK-014 item 2.1).
     The storage pointers that the reference shares with the actor or the teacher (item 1).
     A weakref to each reference parameter, local tensor and FSDP sharded data tensor.
  2. The release as in hostmem.py (its log lines too).
  3. After the release: the alive weakrefs and their storage bytes, and two levels of referrer types of one alive
     object.
It writes <REFDIAG_OUT>/refdiag_<pid>.json and then stops the run by design (before any training step).
"""

import gc
import json
import os
import weakref

import torch

OUT = os.environ.get("REFDIAG_OUT")


def _local(p):
    return getattr(p, "_local_tensor", p)


def _storages(module):
    """{storage pointer: (bytes, is_pinned, device)} of the local parameter and buffer tensors of a module."""
    out = {}
    if module is None:
        return out
    for t in list(module.parameters()) + list(module.buffers()):
        loc = _local(t)
        s = loc.untyped_storage()
        out[s.data_ptr()] = (s.nbytes(), bool(loc.is_pinned()) if loc.device.type == "cpu" else False,
                             f"{loc.device}_{str(loc.dtype).replace('torch.', '')}")
    return out


def _mp_policy(module):
    """The FSDP2 mixed-precision policy of a module (the dtype of the forward parameters), as strings."""
    try:
        mp = module._get_fsdp_state()._mp_policy
        return {"param_dtype": str(mp.param_dtype), "reduce_dtype": str(mp.reduce_dtype),
                "cast_forward_inputs": bool(mp.cast_forward_inputs)}
    except Exception as exc:  # the diagnosis records the failure and continues
        return f"no FSDP2 state: {type(exc).__name__}: {exc}"


def _summary(storages):
    by_kind = {}
    for nbytes, pinned, device in storages.values():
        key = f"{device}{'_pinned' if pinned else ''}"
        by_kind[key] = by_kind.get(key, 0) + nbytes
    return {"storages": len(storages), "gib_by_kind": {k: round(v / 2**30, 3) for k, v in sorted(by_kind.items())}}


def _fsdp_data(module):
    """The _sharded_param_data tensors of the FSDP2 parameters of a module (empty when not FSDP2)."""
    try:
        group = module._get_fsdp_state()._fsdp_param_group
        return [fp._sharded_param_data for fp in group.fsdp_params]
    except Exception as exc:  # the diagnosis records the failure and continues
        return [f"no FSDP2 state: {type(exc).__name__}: {exc}"]


def _refs(ref):
    """Weakrefs (name, ref, storage pointer, storage bytes) to the reference objects that hold parameter memory."""
    out = []
    for name, p in ref.named_parameters():
        out.append(("param:" + name, weakref.ref(p), _local(p).untyped_storage().data_ptr(),
                    _local(p).untyped_storage().nbytes()))
        if _local(p) is not p:
            out.append(("local:" + name, weakref.ref(_local(p)), _local(p).untyped_storage().data_ptr(),
                        _local(p).untyped_storage().nbytes()))
    for i, t in enumerate(_fsdp_data(ref)):
        if isinstance(t, torch.Tensor):
            out.append((f"fsdp_sharded_data:{i}", weakref.ref(t), t.untyped_storage().data_ptr(),
                        t.untyped_storage().nbytes()))
    return out


def _referrer_types(obj, skip):
    """Two levels of referrer types of obj, as strings."""
    lines = []
    for r in gc.get_referrers(obj):
        if any(r is s for s in skip):
            continue
        desc = type(r).__name__
        if isinstance(r, dict):
            owners = sorted({type(o).__name__ for o in gc.get_referrers(r) if not any(o is s for s in skip)})
            keys = [str(k) for k in list(r)[:6]]
            desc += f" keys={keys} owners={owners}"
        lines.append(desc)
    return lines[:20]


def _diagnose(worker, orig_release):
    ref = getattr(worker, "ref_module_fsdp", None)
    if ref is None:
        return orig_release(worker)
    diag = {"pid": os.getpid()}
    s_ref = _storages(ref)
    s_actor = _storages(getattr(worker, "actor_module_fsdp", None))
    s_teacher = _storages(getattr(worker, "teacher_module_fsdp", None))
    diag["reference"] = _summary(s_ref)
    diag["actor"] = _summary(s_actor)
    diag["teacher"] = _summary(s_teacher)
    diag["mp_policy"] = {name: _mp_policy(getattr(worker, attr, None)) for name, attr in
                         [("reference", "ref_module_fsdp"), ("actor", "actor_module_fsdp"),
                          ("teacher", "teacher_module_fsdp")]}
    for name, other in [("actor", s_actor), ("teacher", s_teacher)]:
        shared = [p for p in s_ref if p in other]
        diag[f"reference_shared_with_{name}"] = {"storages": len(shared),
                                                  "gib": round(sum(s_ref[p][0] for p in shared) / 2**30, 3)}
    fsdp = _fsdp_data(ref)
    diag["reference_fsdp_sharded_data"] = (fsdp[:1] if fsdp and isinstance(fsdp[0], str) else
                                           {"tensors": len(fsdp), "gib": round(sum(t.untyped_storage().nbytes()
                                                                                   for t in fsdp) / 2**30, 3)})
    del fsdp
    refs = _refs(ref)
    del ref
    diag["release_log"] = orig_release(worker)
    gc.collect()
    alive = [(n, w, ptr, nb) for n, w, ptr, nb in refs if w() is not None]
    unique = {ptr: nb for _, _, ptr, nb in alive}
    diag["weakrefs"] = len(refs)
    diag["alive"] = len(alive)
    diag["alive_kinds"] = sorted({n.split(":")[0] for n, _, _, _ in alive})
    diag["alive_storage_gib"] = round(sum(unique.values()) / 2**30, 3)
    if alive:
        name, w = alive[0][0], alive[0][1]
        obj = w()
        diag["first_alive"] = name
        diag["first_alive_referrers"] = _referrer_types(obj, skip=[alive, refs, diag])
        del obj
    with open(os.path.join(OUT, f"refdiag_{os.getpid()}.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print("REFDIAG " + json.dumps({k: diag[k] for k in ("pid", "alive", "alive_storage_gib", "alive_kinds")}),
          flush=True)
    raise RuntimeError("REFDIAG: diagnosis done; the run stops after init_model by design")


def setup():
    if not OUT:
        return
    from scripts.vad_packfix import hostmem

    hostmem.setup()
    orig_release = hostmem.release_reference
    hostmem.release_reference = lambda worker: _diagnose(worker, orig_release)
    print(f"REFDIAG hook installed pid={os.getpid()}", flush=True)
