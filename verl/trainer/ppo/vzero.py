"""Stub of the V-Zero helpers (VLM-VAD-ROLLOUT-001, HANDOFF-VLM-VAD-ROLLOUT-001-R2, Part A item 11).

verl/workers/actor/dp_actor.py imports compute_vzero_evidence_weights and compute_vzero_trajectory_scores from
this module at module level. The released code imports it, but no public commit of the repository holds it
(DeepExperience/VAD_Multimodal_OPD up to 4f2dd54 and the fork). Without this file, any import of
verl.workers.actor fails. The functions run only under target_mode vzero, which this experiment does not use.
"""


def compute_vzero_trajectory_scores(*args, **kwargs):
    raise NotImplementedError


def compute_vzero_evidence_weights(*args, **kwargs):
    raise NotImplementedError
