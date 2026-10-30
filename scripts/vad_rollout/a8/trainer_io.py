"""A8 access to the A1 trainer code path (HANDOFF-VLM-VAD-ROLLOUT-001-R9, Part A).

Each function calls the verl code of the A1 run (2627743a). It adds no numerics of its own:
- compose_a1_config: the hydra config of A1 (vopd + the main_ppo overrides in the A1 COMMAND.txt).
- tokenizer_processor: hf_tokenizer and hf_processor with the custom chat template (main_ppo.py 325-335).
- prompt_inputs: the agent-loop prompt of one dataset row (rl_dataset.py 350-405, agent_loop.py 263-331).
- assemble_batch: the per-trajectory postprocess of the agent loop (agent_loop.py 592-785, 818-885).
- teacher_views: RayPPOTrainer._maybe_build_self_distillation_batch (ray_trainer.py 1273-1853).
- load_model and make_actor: the worker model (fsdp_workers.py 394-437) without FSDP. Parameters are cast to
  bf16 and buffers stay fp32, as FSDP MixedPrecision(param_dtype=bf16, buffer_dtype=fp32) does in training.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def read_overrides(command_txt: str) -> list[str]:
    """The main_ppo overrides recorded in a train_arm COMMAND.txt."""
    lines = Path(command_txt).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("== main_ppo arguments"))
    args = []
    for line in lines[start + 1:]:
        if line.startswith("=="):
            break
        args.append(line)
    return args[args.index("vopd") + 1:]


def compose_a1_config(repo_root: str, command_txt: str):
    with initialize_config_dir(config_dir=str(Path(repo_root).resolve() / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="vopd", overrides=read_overrides(command_txt))
    OmegaConf.resolve(cfg)
    return cfg


def answer_hint_config(cfg):
    """The A1 config with the OPSD answer-hint teacher (opsd.yaml 16-17). The template is the actor.yaml value."""
    hint_cfg = copy.deepcopy(cfg)
    with open_dict(hint_cfg):
        hint_cfg.actor_rollout_ref.actor.self_distillation.teacher_prompt_mode = "answer_hint"
        hint_cfg.actor_rollout_ref.actor.self_distillation.teacher_always_on = True
    return hint_cfg


def tokenizer_processor(cfg):
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.chat_template import resolve_custom_chat_template

    path = cfg.actor_rollout_ref.model.path
    trust_remote_code = cfg.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(path, trust_remote_code=trust_remote_code)
    processor = hf_processor(path, trust_remote_code=trust_remote_code, use_fast=True)
    template = resolve_custom_chat_template(cfg.actor_rollout_ref.model)
    if template is not None:
        processor.chat_template = template
        tokenizer.chat_template = template
    return tokenizer, processor


def load_dataset(cfg, tokenizer, processor):
    from verl.utils.dataset.rl_dataset import RLHFDataset

    return RLHFDataset(cfg.data.train_files, tokenizer, cfg.data, processor)


def prompt_inputs(dataset, index: int, processor, cfg) -> dict:
    """Row, messages, images and prompt token ids of one dataset row, as the single-turn agent loop makes them."""
    from verl.utils.dataset.rl_dataset import RLHFDataset

    row = dataset[index]
    messages = row["raw_prompt"]
    images, _ = asyncio.run(RLHFDataset.process_vision_info(
        copy.deepcopy(messages), image_patch_size=processor.image_processor.patch_size, config=cfg.data))
    raw_prompt = processor.apply_chat_template(
        messages, tools=[], add_generation_prompt=True, tokenize=False, **(cfg.data.get("apply_chat_template_kwargs") or {}))
    model_inputs = processor(text=[raw_prompt], images=images, videos=None, video_metadatas=None,
                             return_tensors="pt", do_sample_frames=False)
    prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
    if len(prompt_ids) > cfg.actor_rollout_ref.rollout.prompt_length:
        raise RuntimeError(f"Prompt length {len(prompt_ids)} is longer than {cfg.actor_rollout_ref.rollout.prompt_length}.")
    return {"index": index, "row": row, "images": images, "prompt_ids": prompt_ids, "processor_outputs": model_inputs}


def _one_trajectory(stub, tokenizer, prompt_ids, response_ids, images, prompt_length, response_length):
    """agent_loop.py 622-663 and 689-690 for one trajectory."""
    tokenizer.padding_side = "left"
    prompt = tokenizer.pad({"input_ids": prompt_ids}, padding="max_length", max_length=prompt_length,
                           return_tensors="pt", return_attention_mask=True)
    tokenizer.padding_side = "right"
    response = tokenizer.pad({"input_ids": response_ids}, padding="max_length", max_length=response_length,
                             return_tensors="pt", return_attention_mask=True)
    mask = tokenizer.pad({"input_ids": [1] * len(response_ids)}, padding="max_length", max_length=response_length,
                         return_tensors="pt", return_attention_mask=False)
    for out in (prompt, response, mask):
        if out["input_ids"].dim() == 1:
            for key in list(out.keys()):
                out[key] = out[key].unsqueeze(0)
    response_mask = mask["input_ids"] * response["attention_mask"]
    attention_mask = torch.cat([prompt["attention_mask"], response["attention_mask"]], dim=1)
    input_ids = torch.cat([prompt["input_ids"], response["input_ids"]], dim=1)
    mm_inputs = stub._compute_multi_modal_inputs(SimpleNamespace(multi_modal_data={"images": images}), input_ids)
    position_ids = stub._compute_position_ids(input_ids, attention_mask, mm_inputs)
    return {"prompts": prompt["input_ids"], "responses": response["input_ids"], "response_mask": response_mask,
            "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}, mm_inputs


def assemble_batch(items: list[dict], tokenizer, processor, cfg):
    """DataProto of trajectories. items: prompt_inputs() dicts with response_ids added (and uid)."""
    from verl import DataProto
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    stub = AgentLoopWorker.__new__(AgentLoopWorker)
    stub.tokenizer, stub.processor = tokenizer, processor
    rollout = cfg.actor_rollout_ref.rollout
    tensors, mm_list = [], []
    for item in items:
        t, mm = _one_trajectory(stub, tokenizer, item["prompt_ids"], item["response_ids"], item["images"],
                                rollout.prompt_length, rollout.response_length)
        tensors.append(t)
        mm_list.append(mm)
    batch = {k: torch.cat([t[k] for t in tensors], dim=0) for k in tensors[0]}
    # update_policy reads old_log_probs; on_policy uses log_prob.detach() instead (dp_actor 1485-1486).
    batch["old_log_probs"] = torch.zeros_like(batch["response_mask"], dtype=torch.float32)

    def obj(values):
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        return arr

    rows = [item["row"] for item in items]
    non_tensors = {
        "multi_modal_inputs": obj(mm_list),
        "raw_prompt": obj([r["raw_prompt"] for r in rows]),
        "bbox_images": obj([r["bbox_images"] for r in rows]),
        "degraded_bbox_images": obj([r["degraded_bbox_images"] for r in rows]),
        "reward_model": obj([r["reward_model"] for r in rows]),
        "extra_info": obj([r["extra_info"] for r in rows]),
        "data_source": obj([r["data_source"] for r in rows]),
        "index": obj([r["index"] for r in rows]),
        "uid": obj([item.get("uid", str(uuid.uuid4())) for item in items]),
    }
    return DataProto.from_dict(tensors=batch, non_tensors=non_tensors)


def teacher_views(batch, cfg, tokenizer, processor):
    """teacher_* (and teacher_minus_* in the image mode) tensors for the batch, from the trainer function."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config, trainer.tokenizer, trainer.processor = cfg, tokenizer, processor
    reward = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    views, _ = trainer._maybe_build_self_distillation_batch(batch, reward, {})
    return views


def hint_views(batch, cfg, tokenizer, processor):
    """The answer-hint teacher inputs under the prefix hint_ (the trainer writes them as teacher_*)."""
    from verl import DataProto

    views = teacher_views(batch, answer_hint_config(cfg), tokenizer, processor)
    tensors = {"hint_" + k[len("teacher_"):]: v for k, v in views.batch.items() if k.startswith("teacher_")}
    non_tensors = {"hint_" + k[len("teacher_"):]: v for k, v in views.non_tensor_batch.items() if k.startswith("teacher_")}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def to_device(batch, device):
    """Move the batch tensors and every tensor inside the *multi_modal_inputs dicts to the device.

    In training the FSDP root forward moved the dict tensors (torch fsdp/_runtime_utils.py 596-606).
    """
    batch = batch.to(device)
    for key, values in batch.non_tensor_batch.items():
        if key.endswith("multi_modal_inputs"):
            for i, mm in enumerate(values):
                values[i] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in mm.items()}
    return batch


def load_model(path: str, cfg, train: bool):
    """The worker model of fsdp_workers.py 394-437, then bf16 parameters and fp32 buffers."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.transformers_compat import get_auto_model_for_vision2seq

    vision2seq = get_auto_model_for_vision2seq()
    model_cfg = cfg.actor_rollout_ref.model
    config = AutoConfig.from_pretrained(path, trust_remote_code=model_cfg.get("trust_remote_code", False),
                                        attn_implementation="flash_attention_2")
    for cls in (vision2seq, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModel):
        if cls is AutoModel or type(config) in cls._model_mapping.keys():
            break
    model = cls.from_pretrained(pretrained_model_name_or_path=path, torch_dtype=torch.float32, config=config,
                                trust_remote_code=model_cfg.get("trust_remote_code", False),
                                attn_implementation="flash_attention_2")
    apply_monkey_patch(model=model, use_remove_padding=model_cfg.get("use_remove_padding", True), ulysses_sp_size=1,
                       use_fused_kernels=model_cfg.get("use_fused_kernels", False), fused_kernels_backend=None)
    model.to(torch.float32)
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    return model.cuda()


def init_single_rank(port: int = 29531) -> None:
    """One-rank NCCL group: DataParallelPPOActor calls get_rank() and rearrange_micro_batches calls all_reduce."""
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)


def make_actor(cfg, student, teacher):
    """DataParallelPPOActor on plain modules (fsdp_workers.py 244-245, 845-850, 923)."""
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    actor_cfg_dict = copy.deepcopy(cfg.actor_rollout_ref.actor)
    with open_dict(actor_cfg_dict):
        actor_cfg_dict.ppo_mini_batch_size = (cfg.actor_rollout_ref.actor.ppo_mini_batch_size
                                              * cfg.actor_rollout_ref.rollout.n // cfg.trainer.n_gpus_per_node)
    actor = DataParallelPPOActor(config=omega_conf_to_dataclass(actor_cfg_dict), actor_module=student)
    actor.teacher_module = teacher
    return actor
