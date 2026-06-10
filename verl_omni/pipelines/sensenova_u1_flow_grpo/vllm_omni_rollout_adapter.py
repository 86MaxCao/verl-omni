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
SenseNova-U1 (NEOChatModel) rollout-side adapter for FlowGRPO.

Extends the vllm-omni ``SenseNovaU1Pipeline`` with SDE log-probability
collection so that FlowGRPO RL training can compute per-step rewards.
"""

from __future__ import annotations

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    COND,
    IDX_COND,
    IDX_IMG_COND,
    IDX_UNCOND,
    IMG_COND,
    IMG_START_TOKEN,
    MASK_COND,
    MASK_IMG_COND,
    MASK_UNCOND,
    SYSTEM_MESSAGE_FOR_GEN,
    THINK_OFF,
    UNCOND,
    SenseNovaU1Pipeline,
    _build_t2i_query,
    _denorm,
    _patchify,
    _to_pil,
    _unpatchify,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    clear_flash_kv_cache,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import v_pred_to_velocity

__all__ = ["SenseNovaU1PipelineWithLogProb"]


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _coalesce(*values):
    for v in values:
        if v is not None:
            return v
    return None


@VllmOmniPipelineBase.register("NEOChatModel", algorithm="flow_grpo")
class SenseNovaU1PipelineWithLogProb(SenseNovaU1Pipeline):
    """SenseNova-U1 rollout pipeline with SDE log-prob collection for FlowGRPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sde_scheduler = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000,
            shift=getattr(self.top_cfg, "timestep_shift", 1.0),
        )

    def diffuse(
        self,
        ns,
        caches: dict,
        p,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        logprobs: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SDE denoising loop with per-step log-probability collection.

        Replaces the base class ``_run_denoising_loop`` ODE step with
        ``FlowMatchSDEDiscreteScheduler.step()`` inside the SDE window.

        Returns:
            (image_prediction, all_latents, all_log_probs, all_timesteps)
        """
        merge_size = ns.merge_size
        image_prediction = ns.image_prediction

        scheduler = self.sde_scheduler
        scheduler.timesteps = ns.timesteps[:-1]
        scheduler.sigmas = 1.0 - ns.timesteps
        scheduler.set_begin_index(0)
        scheduler._step_index = 0

        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_timesteps: list[torch.Tensor] = []

        t_eps = getattr(self.top_cfg, "t_eps", 0.05)

        for step_i in range(p.num_steps):
            t = ns.timesteps[step_i]
            t_next = ns.timesteps[step_i + 1]

            z = _patchify(image_prediction, self.patch_size * merge_size)
            image_input = _patchify(
                image_prediction, self.patch_size, channel_first=True
            )
            image_embeds = self._extract_feature(
                image_input.view(p.batch_size * ns.grid_h * ns.grid_w, -1),
                gen_model=True,
                grid_hw=ns.grid_hw,
            ).view(p.batch_size, ns.token_h * ns.token_w, -1)

            t_expanded = t.expand(p.batch_size * ns.token_h * ns.token_w)
            timestep_embeddings = self.fm_modules["timestep_embedder"](
                t_expanded
            ).view(p.batch_size, ns.token_h * ns.token_w, -1)

            if self.top_cfg.add_noise_scale_embedding:
                ns_tensor = torch.full_like(
                    t_expanded,
                    ns.noise_scale / self.top_cfg.noise_scale_max_value,
                )
                ns_emb = self.fm_modules["noise_scale_embedder"](ns_tensor).view(
                    p.batch_size, ns.token_h * ns.token_w, -1
                )
                timestep_embeddings = timestep_embeddings + ns_emb

            image_embeds = image_embeds + timestep_embeddings

            v_pred = self._denoise_step(
                image_prediction, ns, t, z, image_embeds, caches, p, step_i
            )

            in_sde_window = step_i >= sde_window[0] and step_i < sde_window[1]

            if in_sde_window and step_i == sde_window[0]:
                all_latents.append(image_prediction.float())

            if in_sde_window:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            velocity = v_pred_to_velocity(v_pred, t.view(1, 1, 1), t_eps)

            if cur_noise_level > 0:
                prev_sample, log_prob, _, _ = scheduler.step(
                    model_output=velocity.float(),
                    timestep=t,
                    sample=z.float(),
                    generator=generator,
                    noise_level=cur_noise_level,
                    sde_type=sde_type,
                    return_logprobs=logprobs,
                    return_dict=False,
                )
                z_next = prev_sample.to(z.dtype)
            else:
                z_next = z + (t_next - t) * v_pred
                log_prob = None
                scheduler._step_index += 1

            image_prediction = _unpatchify(
                z_next,
                self.patch_size * merge_size,
                p.image_size[1],
                p.image_size[0],
            )

            if in_sde_window:
                all_latents.append(image_prediction.float())
                if log_prob is not None:
                    all_log_probs.append(log_prob)
                else:
                    all_log_probs.append(
                        torch.zeros(
                            z.shape[0], device=z.device, dtype=torch.float32
                        )
                    )
                timestep_value = ns.timesteps[step_i : step_i + 2].float()
                all_timesteps.append(timestep_value)

        for key in (COND, UNCOND, IMG_COND):
            if key in caches and not isinstance(caches[key], dict):
                clear_flash_kv_cache(caches[key])

        if all_latents:
            all_latents = torch.stack(all_latents, dim=1)
        else:
            all_latents = torch.empty(0)
        if all_log_probs:
            all_log_probs = torch.stack(all_log_probs, dim=1)
        else:
            all_log_probs = torch.empty(0)
        if all_timesteps:
            all_timesteps = (
                torch.stack(all_timesteps, dim=0)
                .unsqueeze(0)
                .expand(p.batch_size, -1, -1)
            )
        else:
            all_timesteps = torch.empty(0)

        return image_prediction, all_latents, all_log_probs, all_timesteps

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        """End-to-end T2I/IT2I generation with SDE log-prob collection.

        Overrides the base class ``forward`` to replace the ODE denoising
        loop with an SDE loop and collect per-step latents, log-probs, and
        timesteps for FlowGRPO RL training.

        Automatically routes to IT2I path when input images are present.
        """
        p = self._parse_request(req)
        input_images = self._extract_input_images(p.first_prompt)

        if input_images is not None:
            return self._forward_it2i_sde(p, input_images)
        return self._forward_t2i_sde(p)

    def _forward_t2i_sde(self, p) -> DiffusionOutput:
        """Text-to-image SDE generation with log-prob collection."""
        extra = p.extra_args if hasattr(p, "extra_args") else {}
        noise_level = extra.get("noise_level", 1.0)
        sde_window_size = extra.get("sde_window_size", None)
        sde_window_range = extra.get("sde_window_range", (0, p.num_steps - 1))
        sde_type = extra.get("sde_type", "sde")
        logprobs = extra.get("logprobs", True)

        ns = self._init_noise_and_schedule(p)

        think_content = THINK_OFF + IMG_START_TOKEN
        query_cond = _build_t2i_query(
            p.prompt,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text=think_content,
        )
        query_uncond = _build_t2i_query("", append_text=IMG_START_TOKEN)

        input_ids_cond, indexes_cond, mask_cond = self._build_t2i_text_inputs(
            query_cond
        )
        input_ids_uncond, indexes_uncond, mask_uncond = (
            self._build_t2i_text_inputs(query_uncond)
        )

        indexes_image_cond = self._build_t2i_image_indexes(
            ns.token_h, ns.token_w, indexes_cond.shape[1], self.device
        )
        indexes_image_uncond = self._build_t2i_image_indexes(
            ns.token_h, ns.token_w, indexes_uncond.shape[1], self.device
        )

        past_kv_cond, _ = self._t2i_prefix_forward(
            input_ids_cond, indexes_cond, mask_cond
        )
        past_kv_uncond, _ = self._t2i_prefix_forward(
            input_ids_uncond, indexes_uncond, mask_uncond
        )

        self._expand_and_prepare_kv(
            past_kv_cond, ns.token_h * ns.token_w, p.batch_size
        )
        self._expand_and_prepare_kv(
            past_kv_uncond, ns.token_h * ns.token_w, p.batch_size
        )

        caches = {
            COND: past_kv_cond,
            IDX_COND: indexes_image_cond,
            MASK_COND: {"full_attention": None},
            UNCOND: past_kv_uncond,
            IDX_UNCOND: indexes_image_uncond,
            MASK_UNCOND: {"full_attention": None},
        }

        generator = torch.Generator(device=self.device).manual_seed(p.seed)
        sde_window = self._compute_sde_window(
            p, sde_window_size, sde_window_range, generator
        )

        image_prediction, all_latents, all_log_probs, all_timesteps = (
            self.diffuse(
                ns, caches, p,
                noise_level=noise_level,
                sde_window=sde_window,
                sde_type=sde_type,
                logprobs=logprobs,
                generator=generator,
            )
        )

        prompt_embeds = self.language_model(
            input_ids=input_ids_cond, embed_only=True
        ).inputs_embeds
        prompt_embeds_mask = torch.ones(
            prompt_embeds.shape[:2],
            dtype=torch.bool,
            device=prompt_embeds.device,
        )

        negative_prompt_embeds = self.language_model(
            input_ids=input_ids_uncond, embed_only=True
        ).inputs_embeds
        negative_prompt_embeds_mask = torch.ones(
            negative_prompt_embeds.shape[:2],
            dtype=torch.bool,
            device=negative_prompt_embeds.device,
        )

        return self._build_output(
            image_prediction, all_latents, all_log_probs, all_timesteps,
            prompt_embeds, prompt_embeds_mask,
            negative_prompt_embeds, negative_prompt_embeds_mask,
        )

    def _forward_it2i_sde(self, p, input_images) -> DiffusionOutput:
        """Image-to-image SDE generation with log-prob collection."""
        extra = p.extra_args if hasattr(p, "extra_args") else {}
        noise_level = extra.get("noise_level", 1.0)
        sde_window_size = extra.get("sde_window_size", None)
        sde_window_range = extra.get("sde_window_range", (0, p.num_steps - 1))
        sde_type = extra.get("sde_type", "sde")
        logprobs = extra.get("logprobs", True)

        ns = self._init_noise_and_schedule(p)

        pixel_values, grid_hw = self._prepare_input_images(input_images)
        images_info = {"grid_hw": grid_hw, "pixel_values": pixel_values}

        # Condition: full prompt + input images
        query_cond = self._build_it2i_query(p.prompt, images_info, False)
        embeds_cond, idx_cond, mask_cond = self._build_it2i_inputs(
            query_cond, pixel_values, grid_hw
        )

        # Uncondition: empty prompt, no images
        query_uncond = _build_t2i_query("", append_text=IMG_START_TOKEN)
        embeds_uncond, idx_uncond, mask_uncond = self._build_it2i_inputs(
            query_uncond
        )

        # Prefix forward for condition (it2i uses embeds, not input_ids)
        past_kv_cond, _ = self._it2i_prefix_forward(
            embeds_cond, idx_cond, mask_cond
        )
        idx_image_cond = self._build_t2i_image_indexes(
            ns.token_h, ns.token_w,
            idx_cond[0].max().item() + 1,
            self.device,
        )

        past_kv_uncond, _ = self._it2i_prefix_forward(
            embeds_uncond, idx_uncond, mask_uncond
        )
        idx_image_uncond = self._build_t2i_image_indexes(
            ns.token_h, ns.token_w,
            idx_uncond[0].max().item() + 1,
            self.device,
        )

        self._expand_and_prepare_kv(
            past_kv_cond, ns.token_h * ns.token_w, p.batch_size
        )
        self._expand_and_prepare_kv(
            past_kv_uncond, ns.token_h * ns.token_w, p.batch_size
        )

        caches = {
            COND: past_kv_cond,
            IDX_COND: idx_image_cond,
            MASK_COND: {"full_attention": None},
            UNCOND: past_kv_uncond,
            IDX_UNCOND: idx_image_uncond,
            MASK_UNCOND: {"full_attention": None},
        }

        generator = torch.Generator(device=self.device).manual_seed(p.seed)
        sde_window = self._compute_sde_window(
            p, sde_window_size, sde_window_range, generator
        )

        image_prediction, all_latents, all_log_probs, all_timesteps = (
            self.diffuse(
                ns, caches, p,
                noise_level=noise_level,
                sde_window=sde_window,
                sde_type=sde_type,
                logprobs=logprobs,
                generator=generator,
            )
        )

        # prompt_embeds for training: the condition embeddings (with image)
        prompt_embeds = embeds_cond
        prompt_embeds_mask = torch.ones(
            prompt_embeds.shape[:2],
            dtype=torch.bool,
            device=prompt_embeds.device,
        )

        negative_prompt_embeds = embeds_uncond
        negative_prompt_embeds_mask = torch.ones(
            negative_prompt_embeds.shape[:2],
            dtype=torch.bool,
            device=negative_prompt_embeds.device,
        )

        return self._build_output(
            image_prediction, all_latents, all_log_probs, all_timesteps,
            prompt_embeds, prompt_embeds_mask,
            negative_prompt_embeds, negative_prompt_embeds_mask,
        )

    def _compute_sde_window(self, p, sde_window_size, sde_window_range, generator):
        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            return (start, start + sde_window_size)
        return (0, p.num_steps - 1)

    def _build_output(
        self, image_prediction, all_latents, all_log_probs, all_timesteps,
        prompt_embeds, prompt_embeds_mask,
        negative_prompt_embeds, negative_prompt_embeds_mask,
    ) -> DiffusionOutput:
        image = _denorm(image_prediction)
        return DiffusionOutput(
            output=_maybe_to_cpu(image),
            custom_output={
                "all_latents": _maybe_to_cpu(all_latents),
                "all_log_probs": _maybe_to_cpu(all_log_probs),
                "all_timesteps": _maybe_to_cpu(all_timesteps),
                "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": _maybe_to_cpu(
                    negative_prompt_embeds
                ),
                "negative_prompt_embeds_mask": _maybe_to_cpu(
                    negative_prompt_embeds_mask
                ),
            },
        )
