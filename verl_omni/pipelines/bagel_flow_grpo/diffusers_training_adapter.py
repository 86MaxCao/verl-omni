# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
Bagel training-side adapter for diffusion RL (FlowGRPO).

Bagel is a unified multimodal model where the LLM itself performs
flow-matching diffusion. Unlike Qwen-Image (a pure DiT), Bagel's
flow-matching forward pass runs through the entire LLM with
interleaved text tokens and noisy VAE latent tokens.

Key differences from the Qwen-Image adapter:
    - The module is ``BagelForConditionalGeneration`` (a transformers
      ``PreTrainedModel``), not a diffusers ``ModelMixin``.
    - Flow-matching happens inside the LLM: noisy latent tokens are
      embedded via ``vae2llm``, processed alongside text tokens through
      the MoT (Mixture-of-Transformers) layers, and projected back to
      VAE space via ``llm2vae``.
    - The scheduler uses Bagel's timestep-shift convention and a simple
      linear sigma schedule.
    - ``prompt_embeds`` stores pre-computed text token embeddings from
      the LLM's embed_tokens layer (not from a separate text encoder).
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
from diffusers import SchedulerMixin
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    BAGEL_DEFAULT_TIMESTEP_SHIFT,
    BAGEL_LATENT_CHANNELS,
    BAGEL_LATENT_DOWNSAMPLE,
    BAGEL_LATENT_PATCH_SIZE,
    BAGEL_MAX_LATENT_SIZE,
    BAGEL_PATCH_LATENT_DIM,
    compute_bagel_shifted_sigmas,
    get_flattened_position_ids,
)

__all__ = ["BagelFlowGRPO"]

logger = logging.getLogger(__name__)


def _build_bagel_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    """Build a FlowMatchSDEDiscreteScheduler for Bagel.

    Bagel does not ship a diffusers scheduler config. We construct one
    from scratch with settings matching Bagel's flow-matching formulation.
    """
    scheduler_subfolder = os.path.join(model_path, "scheduler")
    if os.path.isdir(scheduler_subfolder):
        return FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_path,
            subfolder="scheduler",
        )

    # Bagel has no scheduler/ subfolder. Build from scratch.
    # FlowMatchEulerDiscreteScheduler base class defaults work for flow
    # matching; we just need to configure num_train_timesteps and sigma
    # behavior.
    return FlowMatchSDEDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )


def _configure_bagel_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    num_inference_steps: int,
    timestep_shift: float = BAGEL_DEFAULT_TIMESTEP_SHIFT,
    device: str,
) -> None:
    """Configure timesteps and sigmas on the scheduler for Bagel.

    Bagel uses a linear sigma schedule in [0, 1], optionally shifted:
        sigma = shift * t / (1 + (shift - 1) * t)

    We generate ``num_inference_steps`` sigma values from 1 down to 1/N.
    The diffusers scheduler internally appends sigma=0 to get N+1 values,
    and derives ``timesteps = sigmas * 1000``.
    """
    sigmas = compute_bagel_shifted_sigmas(num_inference_steps, timestep_shift)
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)


def _build_bagel_model_forward_inputs(
    module,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    micro_batch: TensorDict,
    cond_vae_embeds: Optional[torch.Tensor] = None,
    cond_vit_embeds: Optional[torch.Tensor] = None,
) -> dict:
    """Build Bagel-specific forward inputs for flow-matching prediction.

    Constructs the packed sequence with optional condition embeddings (i2i),
    text embeddings, and noisy VAE latent tokens.

    Token layout (i2i): [cond_vae | cond_vit | text | noisy_latent]
    Token layout (t2i): [text | noisy_latent]

    MoE routing: cond_vae→gen, cond_vit→und, text→und, noisy_latent→gen

    Args:
        module: The BagelForConditionalGeneration model.
        x_t: Noisy VAE latents at timestep t, shape (B, num_latent_tokens, patch_dim).
        timestep: Timestep value(s), shape (B,) with values in [0, 1].
        prompt_embeds: Pre-computed text token embeddings, shape (B, L, D).
        prompt_embeds_mask: Attention mask for text tokens, shape (B, L).
        micro_batch: Micro-batch containing Bagel-specific metadata.
        cond_vae_embeds: (B, N_vae, D) condition VAE embeddings (i2i), or None.
        cond_vit_embeds: (B, N_vit, D) condition ViT embeddings (i2i), or None.

    Returns:
        Dict with all inputs for the flow-matching forward pass.
    """
    batch_size = x_t.shape[0]
    device = x_t.device
    hidden_size = module.hidden_size

    height = tu.get_non_tensor_data(data=micro_batch, key="height", default=512)
    width = tu.get_non_tensor_data(data=micro_batch, key="width", default=512)

    latent_downsample = getattr(module, "latent_downsample", BAGEL_LATENT_DOWNSAMPLE)
    max_latent_size = getattr(module, "max_latent_size", BAGEL_MAX_LATENT_SIZE)
    use_moe = getattr(module, "use_moe", True)

    h = height // latent_downsample
    w = width // latent_downsample
    num_latent_tokens = h * w

    all_packed_sequences = []
    all_packed_und_indexes = []
    all_packed_gen_indexes = []
    all_packed_latent_indexes = []
    all_packed_position_ids = []
    sample_lens = []

    seq_offset = 0
    for b in range(batch_size):
        segments: list[torch.Tensor] = []
        und_indices: list[int] = []
        gen_indices: list[int] = []
        pos_ids: list[torch.Tensor] = []
        cursor = 0
        pos_counter = 0

        # --- Condition VAE tokens (gen MoE) ---
        if cond_vae_embeds is not None:
            n_cv = cond_vae_embeds.shape[1]
            segments.append(cond_vae_embeds[b])
            gen_indices.extend(range(seq_offset + cursor, seq_offset + cursor + n_cv))
            pos_ids.append(torch.full((n_cv,), pos_counter, device=device, dtype=torch.long))
            cursor += n_cv
            pos_counter += 1

        # --- Condition ViT tokens (und MoE) ---
        if cond_vit_embeds is not None:
            n_vit = cond_vit_embeds.shape[1]
            segments.append(cond_vit_embeds[b])
            und_indices.extend(range(seq_offset + cursor, seq_offset + cursor + n_vit))
            pos_ids.append(torch.full((n_vit,), pos_counter, device=device, dtype=torch.long))
            cursor += n_vit
            pos_counter += 1

        # --- Text tokens (und MoE) ---
        text_len = int(prompt_embeds_mask[b].sum().item())
        text_embeds_b = prompt_embeds[b, :text_len]
        segments.append(text_embeds_b)
        und_indices.extend(range(seq_offset + cursor, seq_offset + cursor + text_len))
        pos_ids.append(torch.arange(pos_counter, pos_counter + text_len, device=device))
        cursor += text_len
        pos_counter += text_len

        # --- Noisy latent tokens (gen MoE) ---
        latent_b = x_t[b]
        t_b = timestep[b:b + 1].expand(num_latent_tokens)
        vae_position_ids = get_flattened_position_ids(
            height, width, latent_downsample, max_latent_size
        ).to(device)
        latent_embeds = (
            module.vae2llm(latent_b)
            + module.time_embedder(t_b)
            + module.latent_pos_embed(vae_position_ids)
        )
        segments.append(latent_embeds.to(text_embeds_b.dtype))
        latent_global_start = seq_offset + cursor
        gen_indices.extend(range(latent_global_start, latent_global_start + num_latent_tokens))
        pos_ids.append(torch.full((num_latent_tokens,), pos_counter, device=device, dtype=torch.long))
        cursor += num_latent_tokens

        # --- Assemble ---
        packed_seq = torch.cat(segments, dim=0)
        all_packed_sequences.append(packed_seq)
        all_packed_und_indexes.append(torch.tensor(und_indices, device=device, dtype=torch.long))
        all_packed_gen_indexes.append(torch.tensor(gen_indices, device=device, dtype=torch.long))
        all_packed_latent_indexes.append(
            torch.arange(latent_global_start, latent_global_start + num_latent_tokens, device=device)
        )
        all_packed_position_ids.append(torch.cat(pos_ids))
        sample_lens.append(cursor)
        seq_offset += cursor

    packed_sequence = torch.cat(all_packed_sequences, dim=0)
    packed_und_indexes = torch.cat(all_packed_und_indexes, dim=0)
    packed_gen_indexes = torch.cat(all_packed_gen_indexes, dim=0)
    packed_latent_indexes = torch.cat(all_packed_latent_indexes, dim=0)
    packed_position_ids = torch.cat(all_packed_position_ids, dim=0)

    model_inputs = {
        "packed_sequence": packed_sequence,
        "packed_und_indexes": packed_und_indexes,
        "packed_gen_indexes": packed_gen_indexes,
        "packed_latent_indexes": packed_latent_indexes,
        "packed_position_ids": packed_position_ids,
        "sample_lens": sample_lens,
        "use_moe": use_moe,
        "num_latent_tokens": num_latent_tokens,
    }

    return model_inputs


def _bagel_flow_forward(module, model_inputs: dict) -> torch.Tensor:
    """Run Bagel's flow-matching forward pass to predict velocity.

    Takes the pre-built packed sequence and runs it through the LLM,
    then projects the latent token outputs to VAE space.

    Args:
        module: BagelForConditionalGeneration model.
        model_inputs: Dict from ``_build_bagel_model_forward_inputs``.

    Returns:
        Velocity prediction tensor of shape (B, num_latent_tokens, patch_dim).
    """
    packed_sequence = model_inputs["packed_sequence"]
    packed_und_indexes = model_inputs["packed_und_indexes"]
    packed_gen_indexes = model_inputs["packed_gen_indexes"]
    packed_latent_indexes = model_inputs["packed_latent_indexes"]
    packed_position_ids = model_inputs["packed_position_ids"]
    sample_lens = model_inputs["sample_lens"]
    use_moe = model_inputs["use_moe"]
    num_latent_tokens = model_inputs["num_latent_tokens"]

    extra_inputs = {}
    if use_moe:
        extra_inputs = dict(
            packed_und_token_indexes=packed_und_indexes,
            packed_gen_token_indexes=packed_gen_indexes,
        )

    last_hidden_state = module.language_model.model(
        packed_sequence=packed_sequence,
        sample_lens=sample_lens,
        attention_mask=None,
        packed_position_ids=packed_position_ids,
        **extra_inputs,
    )

    v_t = module.llm2vae(last_hidden_state[packed_latent_indexes])

    batch_size = len(sample_lens)
    v_t = v_t.reshape(batch_size, num_latent_tokens, -1)

    return v_t


@DiffusionModelBase.register("BagelForConditionalGeneration", algorithm="flow_grpo")
class BagelFlowGRPO(DiffusionModelBase):
    """Training adapter for Bagel flow-matching diffusion RL (FlowGRPO).

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for the Bagel unified multimodal model, providing scheduler
    configuration, model-input construction, and the forward/sampling step
    used during RL training.

    Bagel's flow-matching operates inside the LLM: noisy VAE latent tokens
    are interleaved with text tokens, processed through MoT layers, and the
    output at latent positions is projected to predict the flow velocity.

    Registered under ``"BagelForConditionalGeneration"`` so it is automatically
    selected when ``DiffusionModelConfig.architecture`` matches that name.
    """

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype) -> Optional[torch.nn.Module]:
        """Load BagelForConditionalGeneration via transformers.

        Bagel is not a diffusers model, so we cannot use diffusers AutoModel.
        We load via transformers or the custom BagelForConditionalGeneration class.

        Args:
            model_config: Model configuration.
            torch_dtype: Target dtype for the model.

        Returns:
            The loaded BagelForConditionalGeneration module.
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_config.local_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return model

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchSDEDiscreteScheduler:
        """Build and configure the SDE scheduler for Bagel.

        Args:
            model_config: Configuration for the diffusion model.

        Returns:
            FlowMatchSDEDiscreteScheduler with timesteps already set.
        """
        scheduler = _build_bagel_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ):
        """Configure timesteps and sigmas on the scheduler for Bagel.

        Args:
            scheduler: The scheduler whose timesteps will be set.
            model_config: Configuration providing num_inference_steps.
            device: The device to move timesteps to.
        """
        _configure_bagel_scheduler(
            scheduler,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            timestep_shift=BAGEL_DEFAULT_TIMESTEP_SHIFT,
            device=device,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Build Bagel-specific inputs for the flow-matching forward pass.

        For Bagel, we construct the packed sequence with optional condition
        embeddings (i2i), text embeddings, and noisy VAE latent tokens.

        Args:
            module: The BagelForConditionalGeneration model.
            model_config: Configuration providing guidance scale and other settings.
            latents: Full latent tensor of shape ``(B, T, num_tokens, patch_dim)``
                or ``(B, num_tokens, patch_dim)`` for pre-selected latents.
            timesteps: Full timestep tensor of shape ``(B, T)`` or ``(B,)``
                for pre-selected timesteps.
            prompt_embeds: Pre-computed text token embeddings of shape ``(B, L, D)``.
            prompt_embeds_mask: Attention mask for prompt_embeds of shape ``(B, L)``.
            negative_prompt_embeds: Negative prompt embeddings (unused for Bagel).
            negative_prompt_embeds_mask: Attention mask for negative embeddings.
            micro_batch: Micro-batch with metadata (height, width, etc.).
                May contain ``cond_vae_embeds`` and ``cond_vit_embeds`` for i2i.
            step: Current denoising step index.

        Returns:
            Tuple of (model_inputs, negative_model_inputs). Negative inputs
            are ``None`` since Bagel does not use CFG during training.
        """
        # Select the current step from the trajectory
        if latents.ndim == 4:
            x_t = latents[:, step]
        else:
            x_t = latents

        if timesteps.ndim == 2:
            t = timesteps[:, step]
        else:
            t = timesteps

        # Bagel timesteps are in [0, 1] (sigma space), not in [0, 1000].
        if t.max() > 1.0:
            t = t / 1000.0

        # Extract condition embeddings for i2i (if present in micro_batch)
        cond_vae_embeds = micro_batch.get("cond_vae_embeds", None)
        cond_vit_embeds = micro_batch.get("cond_vit_embeds", None)

        model_inputs = _build_bagel_model_forward_inputs(
            module=module,
            x_t=x_t,
            timestep=t,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            micro_batch=micro_batch,
            cond_vae_embeds=cond_vae_embeds,
            cond_vit_embeds=cond_vit_embeds,
        )

        # Bagel does not use CFG during training.
        return model_inputs, None

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run Bagel's flow-matching forward to predict velocity.

        Args:
            module: BagelForConditionalGeneration model.
            model_config: Model configuration.
            model_inputs: Inputs built by ``prepare_model_inputs``.
            negative_model_inputs: Not used for Bagel.

        Returns:
            Velocity prediction of shape ``(B, num_latent_tokens, patch_dim)``.
        """
        return _bagel_flow_forward(module, model_inputs)

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Run Bagel forward and sample the previous denoising step.

        Used by FlowGRPO for reverse-sampling log-probability computation.

        Args:
            module: BagelForConditionalGeneration model.
            scheduler: SDE scheduler for sampling.
            model_config: Model configuration.
            model_inputs: Inputs for the forward pass.
            negative_model_inputs: Not used for Bagel.
            scheduler_inputs: Must contain ``"all_latents"`` and ``"all_timesteps"``.
            step: Current denoising step index.

        Returns:
            Tuple of ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        # Forward pass to get velocity prediction
        noise_pred = cls.forward(module, model_config, model_inputs)

        # Flatten latent tokens for scheduler: (B, num_tokens, dim) -> (B, num_tokens * dim)
        # The scheduler expects (B, ...) tensors and computes log_prob per sample.
        noise_pred_flat = noise_pred.float().flatten(1)

        # Get current and next latents from trajectory
        current_latents = latents[:, step].float().flatten(1)
        next_latents = latents[:, step + 1].float().flatten(1)

        # Get timestep for scheduler
        current_timesteps = timesteps[:, step]

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=current_latents,
            model_output=noise_pred_flat,
            timestep=current_timesteps,
            noise_level=model_config.algo.noise_level,
            prev_sample=next_latents,
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
