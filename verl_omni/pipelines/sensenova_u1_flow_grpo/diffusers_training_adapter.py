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
SenseNova-U1 (NEOChatModel) training-side adapter for FlowGRPO.

Unlike QwenImage which uses a separate Transformer2DModel denoiser loaded via
diffusers, SenseNova-U1's Qwen3 LLM backbone IS the denoiser.  The model is
loaded via ``transformers.AutoModel`` with ``trust_remote_code=True`` and the
forward pass goes through ``language_model.model()`` + ``fm_modules["fm_head"]``.
"""

from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    apply_true_cfg,
    build_timesteps,
    compute_image_grid,
    compute_noise_scale,
    v_pred_to_velocity,
)

__all__ = ["SenseNovaU1"]


def _get_config_field(model_config: DiffusionModelConfig, key: str, default=None):
    """Read a field from the model's ``config.json`` (stored as ``transformer_config``)."""
    cfg = model_config.transformer_config
    if cfg is None:
        return default
    return cfg.get(key, default)


@DiffusionModelBase.register("NEOChatModel", algorithm="flow_grpo")
class SenseNovaU1(DiffusionModelBase):
    """Training adapter for SenseNova-U1 (NEOChatModel) with FlowGRPO."""

    # -- model loading -------------------------------------------------------

    @classmethod
    def build_module(
        cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype
    ) -> Optional[torch.nn.Module]:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_config.local_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        return model

    # -- scheduler -----------------------------------------------------------

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000,
            shift=_get_config_field(model_config, "timestep_shift", 1.0),
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ):
        height = model_config.pipeline.height
        width = model_config.pipeline.width
        patch_size = _get_config_field(model_config, "patch_size", 16)
        downsample_ratio = _get_config_field(model_config, "downsample_ratio", 0.5)
        num_steps = model_config.pipeline.num_inference_steps

        _, _, token_h, token_w = compute_image_grid(
            height, width, patch_size, downsample_ratio
        )
        image_seq_len = token_h * token_w

        cfg = model_config.transformer_config or {}
        timesteps = build_timesteps(num_steps, image_seq_len, cfg, device)
        scheduler.timesteps = timesteps
        scheduler.sigmas = 1.0 - timesteps

    # -- prepare_model_inputs ------------------------------------------------

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
        height = model_config.pipeline.height
        width = model_config.pipeline.width
        patch_size = _get_config_field(model_config, "patch_size", 16)
        downsample_ratio = _get_config_field(model_config, "downsample_ratio", 0.5)
        merge_size = int(1 / downsample_ratio)

        grid_h, grid_w, token_h, token_w = compute_image_grid(
            height, width, patch_size, downsample_ratio
        )
        image_seq_len = token_h * token_w

        # current image state from latent trajectory
        # latents shape: (B, T, 3, H, W) — pixel-space images
        image_prediction = latents[:, step]
        B = image_prediction.shape[0]
        device = image_prediction.device
        dtype = image_prediction.dtype

        # current timestep (scalar per sample)
        t = timesteps[:, step]  # (B,)

        # patchify at full resolution for z (flow target)
        z = module.patchify(image_prediction, patch_size * merge_size)  # (B, L, D_patch)

        # patchify at patch resolution for visual feature extraction
        image_input = module.patchify(
            image_prediction, patch_size, channel_first=True
        )  # (B, grid_h*grid_w, patch_size²*3)
        grid_hw = torch.tensor(
            [[grid_h, grid_w]] * B, device=device
        )
        image_embeds = module.extract_feature(
            image_input.view(B * grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=grid_hw,
        ).view(B, image_seq_len, -1)

        # timestep embeddings
        t_expanded = t.unsqueeze(1).expand(B, image_seq_len).reshape(-1)
        timestep_embeddings = module.fm_modules["timestep_embedder"](
            t_expanded
        ).view(B, image_seq_len, -1)

        # noise scale embedding
        add_noise_scale_embedding = _get_config_field(
            model_config, "add_noise_scale_embedding", False
        )
        if add_noise_scale_embedding:
            noise_scale_val = compute_noise_scale(
                grid_h=grid_h,
                grid_w=grid_w,
                merge_size=merge_size,
                noise_scale=_get_config_field(model_config, "noise_scale", 1.0),
                noise_scale_mode=_get_config_field(
                    model_config, "noise_scale_mode", "resolution"
                ),
                noise_scale_base_image_seq_len=_get_config_field(
                    model_config, "noise_scale_base_image_seq_len", 64
                ),
                noise_scale_max_value=_get_config_field(
                    model_config, "noise_scale_max_value", 8.0
                ),
            )
            noise_scale_max = _get_config_field(
                model_config, "noise_scale_max_value", 8.0
            )
            noise_scale_tensor = torch.full_like(
                t_expanded, noise_scale_val / noise_scale_max
            )
            noise_embeddings = module.fm_modules["noise_scale_embedder"](
                noise_scale_tensor
            ).view(B, image_seq_len, -1)
            timestep_embeddings = timestep_embeddings + noise_embeddings

        input_embeds = image_embeds + timestep_embeddings

        # 3D indexes for image tokens: (3, image_seq_len)
        # t-dim is set to text_len so image tokens are positioned after text
        # In training mode without KV cache, we concatenate text + image embeddings,
        # so text_len = prompt_embeds.shape[1]
        text_len = prompt_embeds.shape[1]
        indexes_image = _build_image_indexes(
            token_h, token_w, text_len, device
        )  # (3, image_seq_len)

        # build full input: concatenate text embeddings + image embeddings
        full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)  # (B, text_len + image_seq_len, D)

        # build 3D indexes for text tokens
        t_text = torch.arange(text_len, device=device, dtype=torch.long)
        h_text = torch.zeros(text_len, device=device, dtype=torch.long)
        w_text = torch.zeros(text_len, device=device, dtype=torch.long)
        indexes_text = torch.stack([t_text, h_text, w_text], dim=0)  # (3, text_len)

        # full indexes: text + image
        full_indexes = torch.cat(
            [indexes_text, indexes_image], dim=1
        )  # (3, text_len + image_seq_len)

        attention_mask = {"full_attention": None}

        model_inputs = {
            "input_embeds": full_embeds,
            "indexes": full_indexes,
            "attention_mask": attention_mask,
            "t": t,
            "z": z,
            "image_token_num": image_seq_len,
        }

        negative_model_inputs = None
        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0 and negative_prompt_embeds is not None:
            neg_text_len = negative_prompt_embeds.shape[1]
            neg_full_embeds = torch.cat(
                [negative_prompt_embeds, input_embeds], dim=1
            )
            t_neg_text = torch.arange(neg_text_len, device=device, dtype=torch.long)
            h_neg_text = torch.zeros(neg_text_len, device=device, dtype=torch.long)
            w_neg_text = torch.zeros(neg_text_len, device=device, dtype=torch.long)
            indexes_neg_text = torch.stack([t_neg_text, h_neg_text, w_neg_text], dim=0)
            indexes_neg_image = _build_image_indexes(
                token_h, token_w, neg_text_len, device
            )
            neg_full_indexes = torch.cat(
                [indexes_neg_text, indexes_neg_image], dim=1
            )
            negative_model_inputs = {
                "input_embeds": neg_full_embeds,
                "indexes": neg_full_indexes,
                "attention_mask": attention_mask,
                "t": t,
                "z": z,
                "image_token_num": image_seq_len,
            }

        return model_inputs, negative_model_inputs

    # -- forward -------------------------------------------------------------

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        v_pred = _sensenova_predict_v(module, model_config, model_inputs)
        t = model_inputs["t"]
        t_eps = _get_config_field(model_config, "t_eps", 0.05)
        velocity = v_pred_to_velocity(v_pred, t.view(-1, 1, 1), t_eps)
        return velocity

    # -- forward_and_sample_previous_step ------------------------------------

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
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)

        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0:
            assert negative_model_inputs is not None
            neg_noise_pred = cls.forward(
                module, model_config, negative_model_inputs
            )
            noise_pred = apply_true_cfg(
                noise_pred, neg_noise_pred, true_cfg_scale
            )

        # latents are in pixel space (B, T, 3, H, W); patchify to match
        # the velocity's patch space (B, L, D) before passing to scheduler.
        patch_size = _get_config_field(model_config, "patch_size", 16)
        downsample_ratio = _get_config_field(model_config, "downsample_ratio", 0.5)
        merge_size = int(1 / downsample_ratio)
        full_patch = patch_size * merge_size

        sample_z = module.patchify(latents[:, step], full_patch).float()
        prev_z = module.patchify(latents[:, step + 1], full_patch).float()

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = (
            scheduler.sample_previous_step(
                sample=sample_z,
                model_output=noise_pred.float(),
                timestep=timesteps[:, step],
                noise_level=model_config.algo.noise_level,
                prev_sample=prev_z,
                sde_type=model_config.algo.sde_type,
                return_logprobs=True,
                return_sqrt_dt=True,
            )
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_image_indexes(
    token_h: int,
    token_w: int,
    text_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build 3D indexes ``(t, h, w)`` for image tokens.

    Mirrors ``NEOChatModel._build_t2i_image_indexes``. The ``t`` dimension
    is set to ``text_len`` so image tokens are positioned after text in the
    sequence.
    """
    num_tokens = token_h * token_w
    t_image = torch.full(
        (num_tokens,), text_len, dtype=torch.long, device=device
    )
    idx = torch.arange(num_tokens, device=device, dtype=torch.long)
    h_image = idx // token_w
    w_image = idx % token_w
    return torch.stack([t_image, h_image, w_image], dim=0)


def _sensenova_predict_v(
    module,
    model_config: DiffusionModelConfig,
    model_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run the SenseNova-U1 LLM-as-denoiser forward and return ``v_pred``.

    Mirrors ``NEOChatModel._t2i_predict_v`` but runs without KV cache
    (full text + image forward in training mode).
    """
    input_embeds = model_inputs["input_embeds"]
    indexes = model_inputs["indexes"]
    attention_mask = model_inputs["attention_mask"]
    t = model_inputs["t"]
    z = model_inputs["z"]
    image_token_num = model_inputs["image_token_num"]

    B, L = z.shape[0], z.shape[1]

    image_gen_indicators = torch.ones(
        (input_embeds.shape[0], input_embeds.shape[1]),
        dtype=torch.bool,
        device=input_embeds.device,
    )

    outputs = module.language_model.model(
        inputs_embeds=input_embeds,
        image_gen_indicators=image_gen_indicators,
        indexes=indexes,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )

    hidden = outputs.last_hidden_state[:, -image_token_num:]
    hidden = hidden.view(B, L, -1)

    x_pred = module.fm_modules["fm_head"](hidden).view(B, L, -1)

    t_eps = _get_config_field(model_config, "t_eps", 0.05)
    v_pred = (x_pred - z) / (1.0 - t.view(B, 1, 1)).clamp_min(t_eps)
    return v_pred
