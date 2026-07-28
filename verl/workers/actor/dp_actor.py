# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_self_distillation_loss,
    compute_va_high_group_mask,
    compute_va_rollout_weights,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.trainer.ppo.vzero import compute_vzero_evidence_weights, compute_vzero_trajectory_scores
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "vopd":
            return
        teacher_model_source = getattr(self_distillation_cfg, "teacher_model_source", "legacy")
        if teacher_model_source != "legacy":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("Teacher updates require a separate teacher_module in the actor worker.")
        with torch.no_grad():
            if teacher_regularization == "ema":
                update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
                if update_rate == 0.0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    student_data = student_param.data.to(device=teacher_param.device)
                    teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
                return

            if teacher_regularization == "progressive":
                teacher_update_interval = getattr(self_distillation_cfg, "teacher_update_interval", None)
                if teacher_update_interval is None:
                    raise ValueError("Progressive teacher requires self_distillation.teacher_update_interval.")
                global_steps = getattr(self, "_current_global_steps", None)
                if global_steps is None or global_steps % teacher_update_interval != 0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
                for teacher_buffer, student_buffer in zip(
                    self.teacher_module.buffers(),
                    self.actor_module.buffers(),
                ):
                    teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
                return

            return

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    @staticmethod
    def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _build_response_positions(
        response_start_idx: torch.Tensor,
        response_length: int,
        seqlen: int,
    ) -> torch.Tensor:
        if response_start_idx.dim() != 1:
            raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
        if (response_start_idx < 1).any():
            raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

        offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
        response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        if response_positions.numel() > 0:
            if response_positions.min().item() < 0 or response_positions.max().item() >= seqlen:
                raise ValueError(
                    f"Response positions out of bounds for seqlen={seqlen}: "
                    f"min={response_positions.min().item()}, max={response_positions.max().item()}"
                )
        return response_positions.to(dtype=torch.long)

    @staticmethod
    def _select_response_positions(
        hidden_states: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response_start_idx is None:
            return hidden_states[:, -response_length - 1 : -1, ...]

        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=hidden_states.size(1),
        )
        if hidden_states.dim() == 2:
            return torch.gather(hidden_states, dim=1, index=response_positions)

        gather_index = response_positions.view(
            response_positions.size(0),
            response_positions.size(1),
            *([1] * (hidden_states.dim() - 2)),
        ).expand(response_positions.size(0), response_positions.size(1), *hidden_states.shape[2:])
        return torch.gather(hidden_states, dim=1, index=gather_index)

    @staticmethod
    def _build_topk_union_indices(*topk_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not topk_indices:
            raise ValueError("At least one top-k tensor is required to build a union support.")
        reference_shape = topk_indices[0].shape[:-1]
        for tensor in topk_indices:
            if tensor.shape[:-1] != reference_shape:
                raise ValueError(
                    "All top-k tensors must have the same leading shape, got "
                    f"{[tuple(item.shape) for item in topk_indices]}"
                )

        stacked = torch.cat(topk_indices, dim=-1)
        sorted_indices, _ = torch.sort(stacked, dim=-1)
        unique_mask = torch.ones_like(sorted_indices, dtype=torch.bool)
        unique_mask[..., 1:] = sorted_indices[..., 1:] != sorted_indices[..., :-1]
        support_size = unique_mask.sum(dim=-1)
        max_support = int(support_size.max().detach().item())
        ranks = torch.clamp(unique_mask.cumsum(dim=-1) - 1, min=0)
        union_indices = sorted_indices.new_zeros(*sorted_indices.shape[:-1], max_support)
        union_indices.scatter_(dim=-1, index=torch.clamp(ranks, max=max_support - 1), src=sorted_indices)
        range_index = torch.arange(max_support, device=stacked.device).view(
            *([1] * support_size.dim()),
            max_support,
        )
        support_mask = range_index < support_size.unsqueeze(-1)
        return union_indices, support_mask, support_size

    @staticmethod
    def _gather_logps_from_response_logits(
        response_logits: torch.Tensor,
        support_indices: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gathered_logits = torch.gather(response_logits, dim=-1, index=support_indices)
        logsumexp = torch.logsumexp(response_logits, dim=-1, keepdim=True)
        gathered_logps = gathered_logits - logsumexp
        if support_mask is not None:
            gathered_logps = gathered_logps.masked_fill(~support_mask, -1e9)
        return gathered_logps

    def _dump_self_distillation_log_probs(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        dump_chunks: list[dict[str, torch.Tensor]],
    ) -> None:
        dump_root = self_distillation_cfg.get("log_prob_dump_dir", None)
        if not dump_root or not dump_chunks:
            return

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return

        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.pt")

        student_log_probs = torch.cat([chunk["student_log_probs"] for chunk in dump_chunks], dim=0)
        teacher_log_probs = torch.cat([chunk["teacher_log_probs"] for chunk in dump_chunks], dim=0)

        torch.save(
            {
                "student_log_probs": student_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "global_step": int(global_step),
                "rank": rank,
                "experiment_name": experiment_name,
                "distribution_size": int(student_log_probs.shape[-1]),
                "num_valid_tokens": int(student_log_probs.shape[0]),
                "distillation_topk": self_distillation_cfg.get("distillation_topk", None),
                "distillation_add_tail": bool(self_distillation_cfg.get("distillation_add_tail", False)),
            },
            save_path,
        )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        return_response_logits: bool = False,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
                if return_response_logits is True:
                    response_logits: (bs, response_len, vocab_size)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk or return_response_logits) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")
        if return_response_logits and self.use_ulysses_sp:
            raise ValueError("return_response_logits is not implemented for Ulysses sequence parallelism.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_start_idx = micro_batch.get("response_start_idx")
            if response_start_idx is not None:
                response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)
                if response_start_idx.shape != (batch_size,):
                    raise ValueError(
                        f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
                    )
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            if response_start_idx is None:
                                full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            else:
                                response_positions = self._build_response_positions(
                                    response_start_idx=response_start_idx.to(device=topk_indices.device),
                                    response_length=response_length,
                                    seqlen=seqlen,
                                )
                                batch_indices = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1)
                                full_topk_indices[batch_indices, response_positions, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    if return_response_logits:
                        if response_start_idx is None:
                            response_positions = torch.arange(
                                seqlen - response_length - 1,
                                seqlen - 1,
                                device=input_ids.device,
                                dtype=torch.long,
                            ).unsqueeze(0).expand(batch_size, -1)
                        else:
                            response_positions = self._build_response_positions(
                                response_start_idx=response_start_idx.to(device=input_ids.device),
                                response_length=response_length,
                                seqlen=seqlen,
                            )
                        flat_positions = (
                            torch.arange(batch_size, device=input_ids.device).unsqueeze(1) * seqlen
                            + response_positions
                        )
                        flat_to_rmpad = torch.full(
                            (batch_size * seqlen,),
                            -1,
                            device=input_ids.device,
                            dtype=torch.long,
                        )
                        flat_to_rmpad[indices] = torch.arange(indices.numel(), device=input_ids.device)
                        response_rmpad_indices = flat_to_rmpad[flat_positions]
                        valid_response_positions = attention_mask.gather(1, response_positions).bool()
                        valid_response_positions = valid_response_positions & (response_rmpad_indices >= 0)
                        response_logits = logits_rmpad.new_zeros(batch_size, response_length, logits_rmpad.shape[-1])
                        response_logits[valid_response_positions] = logits_rmpad[
                            response_rmpad_indices[valid_response_positions]
                        ]

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = self._select_response_positions(
                        full_entropy.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = self._select_response_positions(
                        full_sum_pi_squared.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                log_probs = self._select_response_positions(
                    full_log_probs.squeeze(-1),
                    response_length=response_length,
                    response_start_idx=response_start_idx,
                )
                if compute_all_logps:
                    all_logps = self._select_response_positions(
                        full_all_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if use_topk:
                    topk_logps = self._select_response_positions(
                        full_topk_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    if return_topk_indices:
                        topk_indices = self._select_response_positions(
                            full_topk_indices,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = self._select_response_positions(
                        logits,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if return_response_logits:
                        response_logits = logits
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            if return_response_logits:
                outputs["response_logits"] = response_logits
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    def _precompute_va_tensors(self, data: DataProto, temperature: float, pad_token_id: int) -> dict:
        """VA-OPD (arXiv:2605.21924) pre-pass.

        Scores each student-rollout token under the teacher with the
        evidence-present view (teacher_* inputs) and the evidence-degraded view
        (teacher_minus_* inputs), computes the per-token visual advantage

            a_t = max(log p_T(y_t | x+) - log p_T(y_t | x-), 0),

        then derives (i) the per-rollout high-VA-group token mask (top
        ``va_high_fraction`` of valid tokens by VA) and (ii) rollout-level
        softmax weights from trajectory-averaged VA, z-score normalized within
        each uid sibling group across all data-parallel ranks. Results are
        attached to ``data.batch`` as ``va_high_group_mask`` (B, T) and
        ``va_rollout_weights`` (B,) so they flow through mini/micro splits.

        Returns a dict of diagnostic metrics.
        """
        cfg = self.config.self_distillation
        teacher_model = self.teacher_module or self.actor_module
        # "teacher": probe = evidence-present teacher view vs degraded teacher view.
        # "student": probe = student's own (full-image) inputs scored by the teacher
        # module vs degraded full image (teacher_minus_* inputs must then carry the
        # degraded full image so both probe conditions share the deployment view).
        probe_view = cfg.get("va_probe_view", "teacher")

        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
            batch_idx_list = None

        plus_log_probs_lst = []
        minus_log_probs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            if probe_view == "student":
                # Evidence-present probe condition = the student's own inputs
                # (full image, raw prompt), scored by the teacher module.
                teacher_inputs = {
                    "responses": model_inputs["responses"],
                    "input_ids": model_inputs["input_ids"],
                    "attention_mask": model_inputs["attention_mask"],
                    "position_ids": model_inputs["position_ids"],
                }
                if "multi_modal_inputs" in model_inputs:
                    teacher_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
            else:
                teacher_inputs = {
                    "responses": model_inputs["responses"],
                    "input_ids": model_inputs["teacher_input_ids"],
                    "attention_mask": model_inputs["teacher_attention_mask"],
                    "position_ids": model_inputs["teacher_position_ids"],
                    "response_start_idx": model_inputs["teacher_response_start_idx"],
                }
                if "teacher_multi_modal_inputs" in model_inputs:
                    teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
            teacher_minus_inputs = {
                "responses": model_inputs["responses"],
                "input_ids": model_inputs["teacher_minus_input_ids"],
                "attention_mask": model_inputs["teacher_minus_attention_mask"],
                "position_ids": model_inputs["teacher_minus_position_ids"],
                "response_start_idx": model_inputs["teacher_minus_response_start_idx"],
            }
            if "teacher_minus_multi_modal_inputs" in model_inputs:
                teacher_minus_inputs["multi_modal_inputs"] = model_inputs["teacher_minus_multi_modal_inputs"]
            with torch.no_grad():
                plus_outputs = self._forward_micro_batch(
                    teacher_inputs, temperature=temperature, calculate_entropy=False, module=teacher_model
                )
                minus_outputs = self._forward_micro_batch(
                    teacher_minus_inputs, temperature=temperature, calculate_entropy=False, module=teacher_model
                )
            plus_log_probs_lst.append(plus_outputs["log_probs"].float().cpu())
            minus_log_probs_lst.append(minus_outputs["log_probs"].float().cpu())

        plus_log_probs = torch.concat(plus_log_probs_lst, dim=0)
        minus_log_probs = torch.concat(minus_log_probs_lst, dim=0)
        if batch_idx_list is not None:
            plus_log_probs = restore_dynamic_batch(plus_log_probs, batch_idx_list)
            minus_log_probs = restore_dynamic_batch(minus_log_probs, batch_idx_list)

        response_mask = data.batch["response_mask"].float().cpu()
        sd_mask = data.batch["self_distillation_mask"].float().cpu()
        loss_mask = response_mask * sd_mask.unsqueeze(1)

        token_va = torch.clamp(plus_log_probs - minus_log_probs, min=0.0) * response_mask
        valid_counts = loss_mask.sum(dim=-1)
        traj_va = (token_va * loss_mask).sum(dim=-1) / valid_counts.clamp(min=1.0)
        active = (valid_counts > 0).float()

        if cfg.get("va_token_grouping", True):
            high_group_mask = compute_va_high_group_mask(
                token_va, loss_mask, high_fraction=float(cfg.get("va_high_fraction", 0.2))
            )
        else:
            high_group_mask = torch.zeros_like(loss_mask)

        if cfg.get("va_rollout_reweight", True):
            if "uid" not in data.non_tensor_batch:
                raise ValueError(
                    "VA-OPD rollout reweighting requires `uid` in the batch to group sibling rollouts. "
                    "Set va_rollout_reweight=False to disable, or ensure uids are propagated."
                )
            uids = [str(u) for u in data.non_tensor_batch["uid"]]
            if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                world_size = torch.distributed.get_world_size()
                gathered: list = [None] * world_size
                torch.distributed.all_gather_object(gathered, (uids, traj_va.tolist(), active.tolist()))
                global_uids = [u for part in gathered for u in part[0]]
                global_traj_va = [v for part in gathered for v in part[1]]
                global_active = [a for part in gathered for a in part[2]]
            else:
                global_uids, global_traj_va, global_active = uids, traj_va.tolist(), active.tolist()
            rollout_weights = compute_va_rollout_weights(
                local_uids=uids,
                local_traj_va=traj_va,
                local_active=active,
                global_uids=global_uids,
                global_traj_va=global_traj_va,
                global_active=global_active,
                softmax_temp=float(cfg.get("va_rollout_softmax_temp", 1.0)),
                zscore_epsilon=float(cfg.get("va_zscore_epsilon", 1e-6)),
            )
        else:
            rollout_weights = active.clone()

        data.batch["va_rollout_weights"] = rollout_weights
        data.batch["va_high_group_mask"] = high_group_mask

        num_valid_tokens = loss_mask.sum().clamp(min=1.0)
        num_active = active.sum().clamp(min=1.0)
        high_token_count = (high_group_mask * loss_mask).sum().clamp(min=1.0)
        va_mass_total = (token_va * loss_mask).sum().clamp(min=1e-12)
        va_mass_high = (token_va * high_group_mask * loss_mask).sum()
        metrics = {
            "self_distillation/va_token_mean": ((token_va * loss_mask).sum() / num_valid_tokens).item(),
            "self_distillation/va_traj_mean": ((traj_va * active).sum() / num_active).item(),
            "self_distillation/va_high_group_token_frac": (high_token_count / num_valid_tokens).item(),
            "self_distillation/va_high_group_mass_frac": (va_mass_high / va_mass_total).item(),
            "self_distillation/va_rollout_weight_local_max": float(rollout_weights.max().item()),
            "self_distillation/va_active_rollout_frac": float(active.mean().item()),
        }
        return metrics

    def _precompute_vzero_tensors(self, data: DataProto, temperature: float, pad_token_id: int) -> dict:
        """Replay positive/negative teacher views and attach V-Zero gates.

        The positive sampled-token log-probabilities are retained for the OPD
        target, so the main update loop does not repeat the positive teacher
        forward.  Sibling z-score statistics are gathered across data-parallel
        ranks because a rollout group is not guaranteed to stay rank-local.
        """

        cfg = self.config.self_distillation
        teacher_model = self.teacher_module or self.actor_module

        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
            batch_idx_list = None

        plus_log_probs_list = []
        minus_log_probs_list = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            teacher_inputs = {
                "responses": model_inputs["responses"],
                "input_ids": model_inputs["teacher_input_ids"],
                "attention_mask": model_inputs["teacher_attention_mask"],
                "position_ids": model_inputs["teacher_position_ids"],
                "response_start_idx": model_inputs["teacher_response_start_idx"],
            }
            if "teacher_multi_modal_inputs" in model_inputs:
                teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]

            teacher_minus_inputs = {
                "responses": model_inputs["responses"],
                "input_ids": model_inputs["teacher_minus_input_ids"],
                "attention_mask": model_inputs["teacher_minus_attention_mask"],
                "position_ids": model_inputs["teacher_minus_position_ids"],
                "response_start_idx": model_inputs["teacher_minus_response_start_idx"],
            }
            if "teacher_minus_multi_modal_inputs" in model_inputs:
                teacher_minus_inputs["multi_modal_inputs"] = model_inputs["teacher_minus_multi_modal_inputs"]

            with torch.no_grad():
                plus_outputs = self._forward_micro_batch(
                    teacher_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    module=teacher_model,
                )
                minus_outputs = self._forward_micro_batch(
                    teacher_minus_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    module=teacher_model,
                )
            plus_log_probs_list.append(plus_outputs["log_probs"].float().cpu())
            minus_log_probs_list.append(minus_outputs["log_probs"].float().cpu())

        plus_log_probs = torch.concat(plus_log_probs_list, dim=0)
        minus_log_probs = torch.concat(minus_log_probs_list, dim=0)
        if batch_idx_list is not None:
            plus_log_probs = restore_dynamic_batch(plus_log_probs, batch_idx_list)
            minus_log_probs = restore_dynamic_batch(minus_log_probs, batch_idx_list)

        response_mask = data.batch["response_mask"].float().cpu()
        self_distillation_mask = data.batch["self_distillation_mask"].float().cpu()
        loss_mask = response_mask * self_distillation_mask.unsqueeze(1)
        active = (loss_mask.sum(dim=-1) > 0).float()
        if "uid" not in data.non_tensor_batch:
            raise ValueError("V-Zero requires `uid` in the actor batch for sibling rollout grouping.")
        uids = [str(uid) for uid in data.non_tensor_batch["uid"]]

        local_scores, _ = compute_vzero_trajectory_scores(
            teacher_logprobs=plus_log_probs,
            teacher_neg_logprobs=minus_log_probs,
            response_mask=loss_mask,
            gamma=float(cfg.get("vzero_gate_gamma", 1.0)),
            eps=float(cfg.get("vzero_gate_epsilon", 1e-6)),
        )
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            world_size = torch.distributed.get_world_size()
            gathered: list = [None] * world_size
            torch.distributed.all_gather_object(gathered, (uids, local_scores.tolist(), active.tolist()))
            global_uids = [uid for part in gathered for uid in part[0]]
            global_scores = [score for part in gathered for score in part[1]]
            global_active = [item for part in gathered for item in part[2]]
        else:
            global_uids = uids
            global_scores = local_scores.tolist()
            global_active = active.tolist()

        weights, metrics, diagnostics = compute_vzero_evidence_weights(
            teacher_logprobs=plus_log_probs,
            teacher_neg_logprobs=minus_log_probs,
            response_mask=loss_mask,
            uids=uids,
            gamma=float(cfg.get("vzero_gate_gamma", 1.0)),
            alpha=float(cfg.get("vzero_gate_alpha", 0.5)),
            w_min=float(cfg.get("vzero_gate_min", 0.0)),
            w_max=float(cfg.get("vzero_gate_max", 2.0)),
            eps=float(cfg.get("vzero_gate_epsilon", 1e-6)),
            active=active,
            global_uids=global_uids,
            global_scores=global_scores,
            global_active=global_active,
        )
        data.batch["vzero_teacher_log_probs"] = plus_log_probs
        data.batch["vzero_rollout_weights"] = weights
        data.batch["vzero_scores"] = diagnostics["scores"]
        data.batch["vzero_advantages"] = diagnostics["advantages"]
        metrics.update(
            {
                "rew_opd/active_rollout_frac": active.mean().item(),
                "rew_opd/public_code_gamma": float(cfg.get("vzero_gate_gamma", 1.0)),
                "rew_opd/public_code_alpha": float(cfg.get("vzero_gate_alpha", 0.5)),
            }
        )
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._current_global_steps = data.meta_info.get("global_steps")
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "vopd"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        vc_target_enabled = False
        va_target_enabled = False
        vzero_target_enabled = False
        if self_distillation_enabled:
            if self_distillation_cfg is None:
                raise ValueError(f"loss_mode={loss_mode} requires actor.self_distillation config.")
            target_mode = self_distillation_cfg.get("target_mode", "full_teacher")
            vc_target_enabled = target_mode == "vc"
            va_target_enabled = target_mode == "va"
            vzero_target_enabled = target_mode == "vzero"
            raw_counterfactual_keys = self_distillation_cfg.get("counterfactual_teacher_image_keys", None)
            if raw_counterfactual_keys is None:
                counterfactual_view_count = 1
            elif isinstance(raw_counterfactual_keys, str):
                counterfactual_view_count = len(
                    [item.strip() for item in raw_counterfactual_keys.split(",") if item.strip()]
                )
            else:
                counterfactual_view_count = len(list(raw_counterfactual_keys))
            vc_direction_mode = self_distillation_cfg.get("vc_direction_mode", "single")
            use_mean_vc_direction = vc_target_enabled and vc_direction_mode == "mean" and counterfactual_view_count > 1
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "teacher_response_start_idx",
                "self_distillation_mask",
            }
            if vc_target_enabled or va_target_enabled or vzero_target_enabled:
                self_distillation_required_keys.update(
                    {
                        "teacher_minus_input_ids",
                        "teacher_minus_attention_mask",
                        "teacher_minus_position_ids",
                        "teacher_minus_response_start_idx",
                    }
                )
                if use_mean_vc_direction:
                    for view_idx in range(1, counterfactual_view_count):
                        self_distillation_required_keys.update(
                            {
                                f"teacher_minus_{view_idx}_input_ids",
                                f"teacher_minus_{view_idx}_attention_mask",
                                f"teacher_minus_{view_idx}_position_ids",
                                f"teacher_minus_{view_idx}_response_start_idx",
                            }
                        )
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
        ]
        if not self_distillation_enabled or "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        has_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_multi_modal_inputs")
        )
        has_teacher_minus_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_minus_multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_multi_modal_inputs")
        if has_teacher_minus_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_minus_multi_modal_inputs")
        if self_distillation_enabled and vc_target_enabled and use_mean_vc_direction:
            for view_idx in range(1, counterfactual_view_count):
                key = f"teacher_minus_{view_idx}_multi_modal_inputs"
                if self._has_non_empty_multi_modal_inputs(data.non_tensor_batch.get(key)):
                    non_tensor_select_keys.append(key)
        if (self.use_prefix_grouper or va_target_enabled or vzero_target_enabled) and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        va_precompute_metrics = None
        if va_target_enabled:
            # VA-OPD pre-pass: score every rollout token under the teacher's
            # evidence-present and evidence-degraded views, then attach
            # va_rollout_weights / va_high_group_mask tensors to the batch so
            # they flow through the mini/micro-batch splits below.
            va_precompute_metrics = self._precompute_va_tensors(
                data, temperature=temperature, pad_token_id=pad_token_id
            )

        vzero_precompute_metrics = None
        if vzero_target_enabled:
            vzero_precompute_metrics = self._precompute_vzero_tensors(
                data, temperature=temperature, pad_token_id=pad_token_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        if self_distillation_enabled:
            metrics["actor/grpo_loss"] = 0.0
            metrics["actor/vopd_loss"] = 0.0
            metrics["actor/vopd_loss_weighted"] = 0.0
        if va_precompute_metrics:
            metrics.update(va_precompute_metrics)
        if vzero_precompute_metrics:
            metrics.update(vzero_precompute_metrics)
        distill_dump_chunks = []
        stage_wall_time_totals = None
        if self_distillation_enabled:
            stage_wall_time_totals = {
                "timing_s/update_actor/student_forward": 0.0,
                "timing_s/update_actor/teacher_forward": 0.0,
                "timing_s/update_actor/teacher_minus_forward": 0.0,
                "timing_s/update_actor/loss_compute": 0.0,
                "timing_s/update_actor/backward": 0.0,
                "timing_s/update_actor/optimizer_step": 0.0,
                "timing_s/update_actor/teacher_ema_update": 0.0,
            }
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    policy_fallback_mask = None
                    if self_distillation_enabled and self_distillation_mask is not None:
                        policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)
                        micro_batch_metrics["actor/policy_fallback_fraction"] = (
                            policy_fallback_mask.float().mean().detach().item()
                        )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy")
                    use_trust_region_teacher = teacher_model_source == "legacy" and teacher_regularization == "trust-region"
                    if use_trust_region_teacher and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                    distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                    vc_support_mode = (
                        self_distillation_cfg.get("vc_support_mode", "student_topk")
                        if vc_target_enabled
                        else "student_topk"
                    )
                    use_three_view_union = vc_target_enabled and vc_support_mode == "three_view_union"
                    distill_support_mask = None
                    student_forward_start = time.perf_counter()
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                        return_response_logits=use_three_view_union,
                    )
                    if self_distillation_enabled:
                        student_forward_time = time.perf_counter() - student_forward_start
                        stage_wall_time_totals["timing_s/update_actor/student_forward"] += student_forward_time
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                            "response_start_idx": model_inputs["teacher_response_start_idx"],
                        }
                        if "teacher_multi_modal_inputs" in model_inputs:
                            teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                        elif (
                            va_target_enabled
                            and self_distillation_cfg.get("va_probe_view", "teacher") == "student"
                            and "multi_modal_inputs" in model_inputs
                        ):
                            teacher_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        teacher_model = self.teacher_module or self.actor_module
                        if use_trust_region_teacher and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        if vzero_target_enabled:
                            # The V-Zero pre-pass already replayed both evidence
                            # views. Reuse its positive sampled-token log-probs as
                            # the detached OPD target instead of forwarding it a
                            # second time for every PPO micro-batch.
                            teacher_outputs = None
                            teacher_log_prob = model_inputs["vzero_teacher_log_probs"]
                            teacher_all_logps = None
                            teacher_topk_logps = None
                        else:
                            with torch.no_grad():
                                teacher_forward_start = time.perf_counter()
                                teacher_outputs = self._forward_micro_batch(
                                    teacher_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=return_all_logps,
                                    distill_topk=distill_topk,
                                    topk_indices=None if use_three_view_union else student_topk_indices,
                                    return_response_logits=use_three_view_union,
                                    module=teacher_model,
                                )
                                teacher_forward_time = time.perf_counter() - teacher_forward_start
                            stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_forward_time
                            teacher_log_prob = teacher_outputs["log_probs"]
                            teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                            teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        teacher_minus_topk_logps = None
                        if vc_target_enabled:
                            if distill_topk is None:
                                raise ValueError("VC-OPD requires distillation_topk to be set.")
                            teacher_minus_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_minus_input_ids"],
                                "attention_mask": model_inputs["teacher_minus_attention_mask"],
                                "position_ids": model_inputs["teacher_minus_position_ids"],
                                "response_start_idx": model_inputs["teacher_minus_response_start_idx"],
                            }
                            if "teacher_minus_multi_modal_inputs" in model_inputs:
                                teacher_minus_inputs["multi_modal_inputs"] = model_inputs[
                                    "teacher_minus_multi_modal_inputs"
                                ]
                            with torch.no_grad():
                                teacher_minus_forward_start = time.perf_counter()
                                teacher_minus_outputs = self._forward_micro_batch(
                                    teacher_minus_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=False,
                                    distill_topk=distill_topk,
                                    topk_indices=None if use_three_view_union else student_topk_indices,
                                    return_response_logits=use_three_view_union,
                                    module=teacher_model,
                                )
                                teacher_minus_forward_time = time.perf_counter() - teacher_minus_forward_start
                                stage_wall_time_totals["timing_s/update_actor/teacher_minus_forward"] += (
                                    teacher_minus_forward_time
                                )
                                teacher_minus_outputs_list = [teacher_minus_outputs]
                                if use_mean_vc_direction:
                                    for view_idx in range(1, counterfactual_view_count):
                                        teacher_minus_inputs = {
                                            "responses": model_inputs["responses"],
                                            "input_ids": model_inputs[f"teacher_minus_{view_idx}_input_ids"],
                                            "attention_mask": model_inputs[f"teacher_minus_{view_idx}_attention_mask"],
                                            "position_ids": model_inputs[f"teacher_minus_{view_idx}_position_ids"],
                                            "response_start_idx": model_inputs[
                                                f"teacher_minus_{view_idx}_response_start_idx"
                                            ],
                                        }
                                        extra_mm_key = f"teacher_minus_{view_idx}_multi_modal_inputs"
                                        if extra_mm_key in model_inputs:
                                            teacher_minus_inputs["multi_modal_inputs"] = model_inputs[extra_mm_key]
                                        with torch.no_grad():
                                            teacher_minus_forward_start = time.perf_counter()
                                            extra_teacher_minus_outputs = self._forward_micro_batch(
                                                teacher_minus_inputs,
                                                temperature=temperature,
                                                calculate_entropy=False,
                                                return_all_logps=False,
                                                distill_topk=distill_topk,
                                                topk_indices=None if use_three_view_union else student_topk_indices,
                                                return_response_logits=use_three_view_union,
                                                module=teacher_model,
                                            )
                                            teacher_minus_forward_time = (
                                                time.perf_counter() - teacher_minus_forward_start
                                            )
                                        stage_wall_time_totals["timing_s/update_actor/teacher_minus_forward"] += (
                                            teacher_minus_forward_time
                                        )
                                        teacher_minus_outputs_list.append(extra_teacher_minus_outputs)
                                teacher_minus_topk_logps = teacher_minus_outputs.get("topk_logps")
                                if use_mean_vc_direction and not use_three_view_union:
                                    teacher_minus_topk_logps = torch.stack(
                                        [outputs_item.get("topk_logps") for outputs_item in teacher_minus_outputs_list],
                                        dim=0,
                                    )
                                if use_three_view_union:
                                    teacher_topk_indices = teacher_outputs.get("topk_indices")
                                    teacher_minus_topk_indices_list = [
                                        outputs_item.get("topk_indices") for outputs_item in teacher_minus_outputs_list
                                    ]
                                    if (
                                        student_topk_indices is None
                                        or teacher_topk_indices is None
                                        or any(item is None for item in teacher_minus_topk_indices_list)
                                    ):
                                        raise ValueError("three_view_union support requires top-k indices from all views.")
                                    union_indices, union_mask, union_size = self._build_topk_union_indices(
                                        student_topk_indices,
                                        teacher_topk_indices,
                                        *teacher_minus_topk_indices_list,
                                    )
                                    distill_support_mask = union_mask
                                    student_topk_logps = self._gather_logps_from_response_logits(
                                        outputs["response_logits"],
                                        union_indices,
                                        union_mask,
                                    )
                                    teacher_topk_logps = self._gather_logps_from_response_logits(
                                        teacher_outputs["response_logits"],
                                        union_indices,
                                        union_mask,
                                    ).detach()
                                    teacher_minus_topk_logps_list = [
                                        self._gather_logps_from_response_logits(
                                            outputs_item["response_logits"],
                                            union_indices,
                                            union_mask,
                                        ).detach()
                                        for outputs_item in teacher_minus_outputs_list
                                    ]
                                    if use_mean_vc_direction:
                                        teacher_minus_topk_logps = torch.stack(
                                            teacher_minus_topk_logps_list,
                                            dim=0,
                                        )
                                    else:
                                        teacher_minus_topk_logps = teacher_minus_topk_logps_list[0]
                                    support_loss_mask = response_mask
                                    if self_distillation_mask is not None:
                                        support_loss_mask = support_loss_mask * self_distillation_mask.unsqueeze(1)
                                    support_loss_mask = support_loss_mask.to(torch.float32)
                                    valid_support_count = support_loss_mask.sum().clamp(min=1)
                                    union_size_for_metrics = union_size.to(support_loss_mask.dtype)
                                    micro_batch_metrics["self_distillation/vc_support_three_view_union"] = 1.0
                                    micro_batch_metrics["self_distillation/vc_support_size.mean"] = (
                                        (union_size_for_metrics * support_loss_mask).sum() / valid_support_count
                                    ).detach().item()
                                    micro_batch_metrics["self_distillation/vc_support_size.max"] = (
                                        torch.where(
                                            support_loss_mask > 0,
                                            union_size_for_metrics,
                                            torch.zeros_like(union_size_for_metrics),
                                        )
                                        .max()
                                        .detach()
                                        .item()
                                    )
                                    if use_mean_vc_direction:
                                        micro_batch_metrics["self_distillation/vc_direction_mean_view_count"] = float(
                                            counterfactual_view_count
                                        )
                        if vc_target_enabled and use_three_view_union:
                            # The union support is built under teacher no-grad above, but student log-probs
                            # must be gathered outside that context so the VC loss can backpropagate.
                            student_topk_logps = self._gather_logps_from_response_logits(
                                outputs["response_logits"],
                                union_indices,
                                union_mask,
                            )
                        if self_distillation_cfg.get("log_prob_dump_dir", None):
                            if distill_topk:
                                student_distill_log_probs = student_topk_logps
                                teacher_distill_log_probs = teacher_topk_logps
                                if self_distillation_cfg.distillation_add_tail:
                                    student_distill_log_probs = self._add_tail_bucket(student_distill_log_probs)
                                    teacher_distill_log_probs = self._add_tail_bucket(teacher_distill_log_probs)
                            else:
                                student_distill_log_probs = student_all_logps
                                teacher_distill_log_probs = teacher_all_logps

                            if student_distill_log_probs is None or teacher_distill_log_probs is None:
                                raise ValueError("Missing distillation log_probs for dump.")

                            loss_mask = response_mask
                            if self_distillation_mask is not None:
                                loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                            valid_rows = loss_mask > 0
                            if valid_rows.any():
                                distill_dump_chunks.append(
                                    {
                                        "student_log_probs": student_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                        "teacher_log_probs": teacher_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                    }
                                )
                        loss_compute_start = time.perf_counter()
                        vopd_loss, vopd_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            teacher_minus_topk_log_probs=teacher_minus_topk_logps,
                            distill_support_mask=distill_support_mask,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            va_rollout_weights=model_inputs.get("va_rollout_weights"),
                            va_high_group_mask=model_inputs.get("va_high_group_mask"),
                            vzero_rollout_weights=model_inputs.get("vzero_rollout_weights"),
                            policy_config=self.config,
                            batch_num_tokens=self.config.global_batch_info.get("batch_num_tokens"),
                            global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                            loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                        )
                        loss_compute_time = time.perf_counter() - loss_compute_start
                        stage_wall_time_totals["timing_s/update_actor/loss_compute"] += loss_compute_time

                        vopd_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                        micro_batch_metrics.update(vopd_metrics)

                        if policy_fallback_mask is not None and policy_fallback_mask.any().item():
                            if advantages is None:
                                raise ValueError(
                                    "Mixed SDPO/GRPO fallback requires advantages for samples without teacher images."
                                )
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, grpo_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask * policy_fallback_mask.unsqueeze(1),
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_loss = vopd_loss + grpo_loss
                            micro_batch_metrics.update(
                                {f"actor/policy_fallback/{key.split('/', 1)[1]}": value for key, value in grpo_metrics.items()}
                            )
                        else:
                            grpo_loss = None
                            pg_loss = vopd_loss
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    backward_start = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self_distillation_enabled:
                        backward_time = time.perf_counter() - backward_start
                        stage_wall_time_totals["timing_s/update_actor/backward"] += backward_time

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    if self_distillation_enabled:
                        metrics["actor/vopd_loss"] += vopd_loss.detach().item() * loss_scale_factor
                        metrics["actor/vopd_loss_weighted"] += vopd_loss.detach().item() * loss_scale_factor
                        if grpo_loss is not None:
                            metrics["actor/grpo_loss"] += grpo_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                optimizer_step_start = time.perf_counter()
                grad_norm = self._optimizer_step()
                if self_distillation_enabled:
                    optimizer_step_time = time.perf_counter() - optimizer_step_start
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if self_distillation_enabled:
                    stage_wall_time_totals["timing_s/update_actor/optimizer_step"] += optimizer_step_time
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self_distillation_enabled and distill_dump_chunks:
            self._dump_self_distillation_log_probs(
                meta_info=data.meta_info,
                self_distillation_cfg=self_distillation_cfg,
                dump_chunks=distill_dump_chunks,
            )
        if did_update:
            teacher_update_start = time.perf_counter()
            self._update_teacher()
            if self_distillation_enabled:
                stage_wall_time_totals["timing_s/update_actor/teacher_ema_update"] += (
                    time.perf_counter() - teacher_update_start
                )
        if self_distillation_enabled:
            for key, total_time in stage_wall_time_totals.items():
                metrics[key] = Metric(aggregation=AggregationType.MAX, value=total_time)
        metric_keys_to_keep_unreduced = set(stage_wall_time_totals.keys()) if stage_wall_time_totals is not None else set()
        local_metrics_to_reduce = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, list) or (isinstance(value, Metric) and key not in metric_keys_to_keep_unreduced)
        }
        if local_metrics_to_reduce:
            reduced_local_metrics = reduce_metrics(local_metrics_to_reduce)
            metrics.update(reduced_local_metrics)
        return metrics
