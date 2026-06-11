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
Bagel rollout adapter for vllm-omni (FlowGRPO).

Standalone PyTorch pipeline for Bagel image generation with SDE
log-probability collection.  Unlike Qwen-Image (which inherits from a
vllm-omni pipeline), this loads the full BagelForConditionalGeneration
model directly because vllm's Bagel model only supports text generation
and skips generation weights (vae2llm, llm2vae, time_embedder, etc.).

The denoising loop uses the same packed-sequence forward pass as the
training adapter (text embeddings + noisy latent tokens in a single
sequence, no KV cache).  This guarantees rollout/training log-prob
consistency at the cost of re-processing text tokens at each step.

Requires ``PYTHONPATH`` to include the vllm-omni source directory so
that ``vllm_omni.diffusion.*`` imports resolve at runtime.
"""

import logging
import os
from collections.abc import Iterable
from typing import Any, Literal, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    BAGEL_DEFAULT_TIMESTEP_SHIFT,
    BAGEL_LATENT_CHANNELS,
    BAGEL_LATENT_DOWNSAMPLE,
    BAGEL_LATENT_PATCH_SIZE,
    BAGEL_MAX_LATENT_SIZE,
    BAGEL_PATCH_LATENT_DIM,
    compute_bagel_shifted_sigmas,
    coalesce_not_none,
    get_flattened_position_ids,
    image_to_vae_input,
    maybe_to_cpu,
    patchify_for_vit,
    resize_image_to_stride,
)

__all__ = ["BagelPipelineWithLogProb"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_load_vae(model_path: str, device: torch.device, vae_config=None):
    """Attempt to load the Bagel FLUX-style VAE from ``ae.safetensors``.

    The ``AutoEncoder`` class lives in the model's remote-code module
    (``modeling_bagel.py``), not in a separate ``modeling.autoencoder``
    package.  We import it via the same trust_remote_code mechanism that
    loaded the main model.

    Args:
        model_path: Path to the Bagel model directory.
        device: Target device for the VAE.
        vae_config: Optional ``BagelVaeConfig`` from the loaded model.
            When provided, it is used directly to construct the AutoEncoder.
    """
    vae_path = os.path.join(model_path, "ae.safetensors")
    if not os.path.exists(vae_path):
        logger.warning("VAE not found at %s; image decode unavailable.", vae_path)
        return None
    try:
        from safetensors.torch import load_file as load_sft

        sd = load_sft(vae_path)

        # Import AutoEncoder from the model's trust_remote_code module.
        # The class is defined in modeling_bagel.py alongside the main model.
        AutoEncoder = None
        BagelVaeConfig = None
        try:
            import importlib

            mod = importlib.import_module("transformers_modules.modeling_bagel")
            AutoEncoder = getattr(mod, "AutoEncoder", None)
            BagelVaeConfig = getattr(mod, "BagelVaeConfig", None)
        except ImportError:
            pass

        if AutoEncoder is None:
            # Fallback: try the model directory's own modeling_bagel.
            import sys

            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            try:
                from modeling_bagel import AutoEncoder as _AE  # type: ignore

                AutoEncoder = _AE
            except ImportError:
                logger.warning(
                    "Cannot import AutoEncoder from modeling_bagel. "
                    "VAE encode/decode will be unavailable."
                )
                return None

        if vae_config is None:
            # Construct BagelVaeConfig from defaults (matching BagelVaeConfig defaults).
            if BagelVaeConfig is not None:
                vae_config = BagelVaeConfig()
            else:
                logger.warning("BagelVaeConfig not found; VAE load skipped.")
                return None

        ae = AutoEncoder(vae_config)
        ae.load_state_dict(sd, strict=False, assign=True)
        return ae.eval().to(device)
    except Exception as e:
        logger.warning("Failed to load VAE: %s", e)
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@VllmOmniPipelineBase.register("BagelForConditionalGeneration", algorithm="flow_grpo")
class BagelPipelineWithLogProb(torch.nn.Module):
    """Standalone rollout pipeline for Bagel flow-matching image generation.

    Loads the full ``BagelForConditionalGeneration`` model (including
    ``vae2llm``, ``llm2vae``, ``time_embedder``, ``latent_pos_embed``)
    since vllm's Bagel model only loads understanding weights.

    The denoising loop mirrors ``_bagel_flow_forward`` from the training
    adapter: at each step it packs text embeddings and noisy latent tokens
    into one sequence and runs a full LLM forward (no KV cache).

    The output ``DiffusionOutput.custom_output`` contains the same keys as
    the Qwen-Image adapter: ``all_latents``, ``all_log_probs``,
    ``all_timesteps``, ``prompt_embeds``, ``prompt_embeds_mask``.

    Registered as ``("BagelForConditionalGeneration", "flow_grpo")``.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.device = get_local_device()
        model_path = od_config.model

        from transformers import AutoTokenizer

        from transformers import AutoModelForCausalLM

        logger.info("Loading Bagel model from %s for FlowGRPO rollout", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = model.eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        for tok in ["<|vision_start|>", "<|vision_end|>"]:
            if tok not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([tok])

        self.new_token_ids = {
            "bos_token_id": self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
            "eos_token_id": self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "start_of_image": self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "end_of_image": self.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        }

        self.hidden_size = self.model.hidden_size
        self.latent_downsample = getattr(self.model, "latent_downsample", BAGEL_LATENT_DOWNSAMPLE)
        self.latent_patch_size = getattr(self.model, "latent_patch_size", BAGEL_LATENT_PATCH_SIZE)
        self.max_latent_size = getattr(self.model, "max_latent_size", BAGEL_MAX_LATENT_SIZE)
        self.latent_channel = getattr(self.model, "latent_channel", BAGEL_LATENT_CHANNELS)
        self.patch_latent_dim = getattr(self.model, "patch_latent_dim", BAGEL_PATCH_LATENT_DIM)
        self.use_moe = getattr(self.model, "use_moe", True)

        self.vae_model = _try_load_vae(
            model_path, self.device, vae_config=getattr(self.model.config, "vae_config", None)
        )

        # i2i: ViT-related components (from the loaded model if visual_und=True)
        self.has_vit = hasattr(self.model, "vit_model") and self.model.vit_model is not None
        if self.has_vit:
            self.vit_patch_size = getattr(self.model, "vit_patch_size", 14)
            self.vit_max_num_patch_per_side = getattr(self.model, "vit_max_num_patch_per_side", 70)
            try:
                from transformers import SiglipImageProcessor

                self.image_processor = SiglipImageProcessor.from_pretrained(model_path, local_files_only=True)
            except Exception:
                self.image_processor = None
                logger.warning("SiglipImageProcessor not loaded; i2i may not work.")
        else:
            self.image_processor = None
            logger.info("ViT model not found on BagelForConditionalGeneration; i2i disabled.")

        self.scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=1.0)

        self._interrupt = False
        self._current_timestep = None

    # ------------------------------------------------------------------
    # Weight sync (called by vllm-omni DiffusionModelRunner and initial load)
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        param_dict = dict(self.model.named_parameters())
        buf_dict = dict(self.model.named_buffers())
        all_keys = {**param_dict, **buf_dict}
        loaded = set()
        for name, tensor in weights:
            clean = name
            for pfx in ("transformer.", "module.", "model."):
                if clean.startswith(pfx):
                    clean = clean[len(pfx):]
                    break
            if clean in all_keys:
                all_keys[clean].data.copy_(tensor)
                loaded.add(clean)
        if loaded:
            logger.info("BagelPipelineWithLogProb.load_weights: updated %d params", len(loaded))
        return loaded

    # ------------------------------------------------------------------
    # Properties expected by vllm-omni pipeline interface
    # ------------------------------------------------------------------

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return None

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt_text: Optional[str] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and compute embeddings via ``embed_tokens``.

        Returns ``(prompt_embeds, prompt_embeds_mask)`` of shapes
        ``(1, max_seq_len, D)`` and ``(1, max_seq_len)`` compatible with
        the training adapter's expected input.
        """
        if prompt_ids is not None:
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            if prompt_mask is None:
                prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
            elif prompt_mask.ndim == 1:
                prompt_mask = prompt_mask.unsqueeze(0)
            prompt_ids = prompt_ids.to(self.device)
            prompt_embeds = self.model.language_model.model.embed_tokens(prompt_ids)
            return prompt_embeds, prompt_mask.to(self.device)

        assert prompt_text is not None, "Either prompt_text or prompt_ids required"

        text_ids = self.tokenizer.encode(prompt_text)
        if len(text_ids) > max_sequence_length - 2:
            text_ids = text_ids[: max_sequence_length - 2]
        text_ids = [self.new_token_ids["bos_token_id"]] + text_ids + [self.new_token_ids["eos_token_id"]]
        text_len = len(text_ids)

        padded = text_ids + [0] * (max_sequence_length - text_len)
        ids_t = torch.tensor([padded], dtype=torch.long, device=self.device)
        prompt_embeds = self.model.language_model.model.embed_tokens(ids_t)

        mask = torch.zeros(1, max_sequence_length, dtype=torch.bool, device=self.device)
        mask[0, :text_len] = True
        return prompt_embeds, mask

    # ------------------------------------------------------------------
    # Condition image encoding (i2i)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_condition_image(
        self,
        image,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a condition image into VAE latent embeddings and ViT embeddings.

        The condition image is processed through two paths:
        1. VAE encode → patchify → vae2llm + time_embedder(t=0) + latent_pos_embed
        2. ViT → connector + vit_pos_embed

        Args:
            image: PIL.Image condition image.
            height: Target height (used for VAE latent grid).
            width: Target width (used for VAE latent grid).

        Returns:
            Tuple of (cond_vae_embeds, cond_vit_embeds):
            - cond_vae_embeds: (1, num_vae_tokens, D) condition VAE embeddings
            - cond_vit_embeds: (1, num_vit_tokens, D) condition ViT embeddings,
              or None if ViT is unavailable.
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            image = PILImage.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to match output resolution (aligned to latent_downsample)
        stride = self.latent_downsample
        max_size = int(self.max_latent_size * stride)
        image = resize_image_to_stride(image, stride, max_size)
        img_w, img_h = image.size

        # --- VAE path ---
        cond_vae_embeds = None
        if self.vae_model is not None:
            img_tensor = image_to_vae_input(image).to(self.device)  # (C, H, W)
            vae_dtype = next(self.vae_model.parameters()).dtype
            latent = self.vae_model.encode(img_tensor.unsqueeze(0).to(vae_dtype))  # (1, C, h*p, w*p)

            # Patchify: (1, C, h*p, w*p) → (1, h*w, p²*C)
            h = img_h // stride
            w = img_w // stride
            p = self.latent_patch_size
            c = self.latent_channel
            latent = latent[0]  # (C, h*p, w*p)
            latent = latent[:, :h * p, :w * p].reshape(c, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(h * w, p * p * c)

            # Embed: vae2llm + time(0) + pos
            vae_pos_ids = get_flattened_position_ids(
                img_h, img_w, stride, self.max_latent_size
            ).to(self.device)
            zero_t = torch.zeros(h * w, device=self.device, dtype=torch.float32)
            model_dtype = next(self.model.parameters()).dtype
            cond_vae_embeds = (
                self.model.vae2llm(latent.to(model_dtype))
                + self.model.time_embedder(zero_t)
                + self.model.latent_pos_embed(vae_pos_ids)
            ).unsqueeze(0)  # (1, h*w, D)

        # --- ViT path ---
        cond_vit_embeds = None
        if self.has_vit:
            if self.image_processor is not None:
                vit_pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values[0]  # (C, H_vit, W_vit)
            else:
                import torchvision.transforms.functional as TF
                vit_pixel_values = TF.to_tensor(image)

            # Patchify for ViT
            vit_dtype = next(self.model.vit_model.parameters()).dtype
            vit_tokens = patchify_for_vit(vit_pixel_values, self.vit_patch_size).to(
                device=self.device, dtype=vit_dtype
            )
            num_vit_tokens = vit_tokens.shape[0]

            # Position IDs for ViT
            vit_h = vit_pixel_values.shape[1]
            vit_w = vit_pixel_values.shape[2]
            vit_pos_ids = get_flattened_position_ids(
                vit_h, vit_w, self.vit_patch_size, self.vit_max_num_patch_per_side
            ).to(self.device)

            # Run ViT
            cu_seqlens = torch.tensor([0, num_vit_tokens], dtype=torch.int32, device=self.device)
            vit_out = self.model.vit_model(
                packed_pixel_values=vit_tokens,
                packed_flattened_position_ids=vit_pos_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=num_vit_tokens,
            )

            # Connector + position embed
            vit_out = self.model.connector(vit_out)
            vit_out = vit_out + self.model.vit_pos_embed(vit_pos_ids)
            cond_vit_embeds = vit_out.unsqueeze(0)  # (1, num_vit_tokens, D)

        return cond_vae_embeds, cond_vit_embeds

    # ------------------------------------------------------------------
    # Velocity prediction (matches training adapter's _bagel_flow_forward)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_velocity(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        height: int,
        width: int,
        cond_vae_embeds: Optional[torch.Tensor] = None,
        cond_vit_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict flow velocity via packed-sequence LLM forward.

        Packs condition embeddings (if i2i), text embeddings, and noisy VAE
        latent tokens into a single sequence per sample, runs the LLM, and
        projects latent outputs to VAE space via ``llm2vae``.

        Token layout (i2i): [cond_vae | cond_vit | text | noisy_latent]
        Token layout (t2i): [text | noisy_latent]

        MoE routing:
        - cond_vae → gen, cond_vit → und, text → und, noisy_latent → gen

        Args:
            x_t: ``(B, num_tokens, patch_dim)`` noisy latent tokens.
            timestep: ``(B,)`` timestep values in ``[0, 1]``.
            prompt_embeds: ``(B, L, D)`` text embeddings.
            prompt_embeds_mask: ``(B, L)`` boolean mask.
            height, width: image size in pixels.
            cond_vae_embeds: ``(B, N_vae, D)`` condition image VAE embeddings, or None.
            cond_vit_embeds: ``(B, N_vit, D)`` condition image ViT embeddings, or None.

        Returns:
            ``(B, num_tokens, patch_dim)`` velocity prediction.
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        num_latent_tokens = x_t.shape[1]

        vae_pos_ids = get_flattened_position_ids(
            height, width, self.latent_downsample, self.max_latent_size
        ).to(device)

        all_packed: list[torch.Tensor] = []
        all_und_idx: list[torch.Tensor] = []
        all_gen_idx: list[torch.Tensor] = []
        all_pos_ids: list[torch.Tensor] = []
        all_latent_idx: list[torch.Tensor] = []
        sample_lens: list[int] = []
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
            text_emb = prompt_embeds[b, :text_len]
            segments.append(text_emb)
            und_indices.extend(range(seq_offset + cursor, seq_offset + cursor + text_len))
            pos_ids.append(torch.arange(pos_counter, pos_counter + text_len, device=device))
            cursor += text_len
            pos_counter += text_len

            # --- Noisy latent tokens (gen MoE) ---
            t_b = timestep[b : b + 1].expand(num_latent_tokens)
            lat_emb = (
                self.model.vae2llm(x_t[b])
                + self.model.time_embedder(t_b)
                + self.model.latent_pos_embed(vae_pos_ids)
            )
            segments.append(lat_emb.to(text_emb.dtype))
            latent_global_start = seq_offset + cursor
            gen_indices.extend(range(latent_global_start, latent_global_start + num_latent_tokens))
            pos_ids.append(torch.full((num_latent_tokens,), pos_counter, device=device, dtype=torch.long))
            cursor += num_latent_tokens

            # --- Assemble ---
            packed = torch.cat(segments, dim=0)
            all_packed.append(packed)
            all_und_idx.append(torch.tensor(und_indices, device=device, dtype=torch.long))
            all_gen_idx.append(torch.tensor(gen_indices, device=device, dtype=torch.long))
            all_latent_idx.append(torch.arange(latent_global_start, latent_global_start + num_latent_tokens, device=device))
            all_pos_ids.append(torch.cat(pos_ids))
            sample_lens.append(cursor)
            seq_offset += cursor

        packed_seq = torch.cat(all_packed)
        und_indexes = torch.cat(all_und_idx)
        gen_indexes = torch.cat(all_gen_idx)
        latent_indexes = torch.cat(all_latent_idx)
        position_ids = torch.cat(all_pos_ids)

        extra: dict[str, Any] = {}
        if self.use_moe:
            extra = dict(
                packed_und_token_indexes=und_indexes,
                packed_gen_token_indexes=gen_indexes,
            )

        hidden = self.model.language_model.model(
            packed_sequence=packed_seq,
            sample_lens=sample_lens,
            attention_mask=None,
            packed_position_ids=position_ids,
            **extra,
        )

        v_t = self.model.llm2vae(hidden[latent_indexes])
        return v_t.reshape(batch_size, num_latent_tokens, -1)

    # ------------------------------------------------------------------
    # SDE denoising loop
    # ------------------------------------------------------------------

    def diffuse(
        self,
        x_t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        height: int,
        width: int,
        timesteps: torch.Tensor,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: Optional[torch.Generator],
        logprobs: bool,
        cond_vae_embeds: Optional[torch.Tensor] = None,
        cond_vit_embeds: Optional[torch.Tensor] = None,
    ):
        """Run the full flow-matching denoising loop with SDE collection.

        Iterates over ``timesteps`` (from the configured scheduler). Within
        the SDE window, Gaussian noise is injected and per-step latents /
        log-probabilities are collected.

        Args:
            x_t: ``(B, num_tokens, patch_dim)`` initial noise.
            prompt_embeds: ``(B, L, D)``.
            prompt_embeds_mask: ``(B, L)``.
            height, width: image size in pixels.
            timesteps: 1-D tensor from ``scheduler.timesteps`` (values ∈
                ``[0, 1000]``).
            noise_level: SDE noise magnitude inside the window.
            sde_window: ``(start, end)`` step indices.
            sde_type: ``"sde"`` or ``"cps"``.
            generator: optional RNG for reproducibility.
            logprobs: whether to compute per-step log-probabilities.
            cond_vae_embeds: ``(B, N, D)`` condition VAE embeddings (i2i), or None.
            cond_vit_embeds: ``(B, N, D)`` condition ViT embeddings (i2i), or None.

        Returns:
            ``(x_t, all_latents, all_log_probs, all_timesteps)``
        """
        batch_size = x_t.shape[0]

        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_timesteps: list[torch.Tensor] = []
        self.scheduler.set_begin_index(0)

        for i, ts_val in enumerate(timesteps):
            if self.interrupt:
                continue

            # Determine per-step noise level
            if i < sde_window[0]:
                cur_noise = 0.0
            elif i == sde_window[0]:
                cur_noise = noise_level
                all_latents.append(x_t.float())
            elif i < sde_window[1]:
                cur_noise = noise_level
            else:
                cur_noise = 0.0

            self._current_timestep = ts_val

            # Bagel sigma ∈ [0, 1]; scheduler timestep = sigma * 1000
            sigma = (ts_val / 1000.0) if isinstance(ts_val, (int, float)) else (ts_val.item() / 1000.0)
            sigma_t = torch.tensor([sigma], device=self.device, dtype=torch.float32).expand(batch_size)

            v_t = self._forward_velocity(
                x_t, sigma_t, prompt_embeds, prompt_embeds_mask, height, width,
                cond_vae_embeds=cond_vae_embeds,
                cond_vit_embeds=cond_vit_embeds,
            )

            # Flatten for scheduler: (B, T, D) → (B, T*D)
            x_flat = x_t.float().flatten(1)
            v_flat = v_t.float().flatten(1)

            x_flat, log_prob, _, _ = self.scheduler.step(
                model_output=v_flat,
                timestep=ts_val,
                sample=x_flat,
                generator=generator,
                noise_level=cur_noise,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            x_t = x_flat.reshape(batch_size, -1, self.patch_latent_dim).to(prompt_embeds.dtype)

            if sde_window[0] <= i < sde_window[1]:
                all_latents.append(x_t.float())
                all_log_probs.append(log_prob)
                all_timesteps.append(ts_val)

        # Stack: latents → (B, W+1, T, D), log_probs → (B, W), timesteps → (B, W)
        all_latents_t = torch.stack(all_latents, dim=1)
        all_log_probs_t = (
            torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        )
        all_timesteps_t = torch.stack(all_timesteps).unsqueeze(0).expand(batch_size, -1)

        return x_t, all_latents_t, all_log_probs_t, all_timesteps_t

    # ------------------------------------------------------------------
    # VAE decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_latents(self, x_t: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Decode patchified latents to images via the FLUX-style VAE.

        Args:
            x_t: ``(B, num_tokens, patch_dim)`` final denoised latents.
            height, width: image size in pixels.

        Returns:
            ``(B, 3, H, W)`` images in ``[0, 1]``, or raw latents when
            the VAE is unavailable.
        """
        if self.vae_model is None:
            logger.warning("VAE not loaded; returning raw latents.")
            return x_t

        B = x_t.shape[0]
        h = height // self.latent_downsample
        w = width // self.latent_downsample
        p = self.latent_patch_size
        c = self.latent_channel

        # (B, h*w, p²*c) → (B, c, h*p, w*p)
        latent = x_t.reshape(B, h, w, p, p, c)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(B, c, h * p, w * p)

        vae_dtype = next(self.vae_model.parameters()).dtype
        image = self.vae_model.decode(latent.to(vae_dtype))
        return (image * 0.5 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------
    # End-to-end forward (vllm-omni pipeline interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        true_cfg_scale: float = 1.0,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 24,
        max_sequence_length: int = 256,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        noise_level: float = 1.2,
        sde_window_size: Optional[int] = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
        timestep_shift: float = BAGEL_DEFAULT_TIMESTEP_SHIFT,
    ) -> DiffusionOutput:
        """End-to-end Bagel image generation with rollout data collection.

        Encodes the prompt, prepares initial latent noise, runs the SDE
        denoising loop via :meth:`diffuse`, optionally decodes through the
        VAE, and returns a :class:`DiffusionOutput` with all rollout data.

        Sampling parameters from ``req.sampling_params`` take precedence
        over keyword arguments.

        Returns:
            ``DiffusionOutput`` with ``custom_output`` keys:
            ``"all_latents"``, ``"all_log_probs"``, ``"all_timesteps"``,
            ``"prompt_embeds"``, ``"prompt_embeds_mask"``.
        """
        # -- Extract prompt and condition image ----------------------------
        custom_prompt = req.prompts[0] if req.prompts else {}
        prompt_text: Optional[str] = None
        condition_image = None
        if isinstance(custom_prompt, str):
            prompt_text = custom_prompt
        elif isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            prompt_text = custom_prompt.get("prompt_text", prompt_text)
            # Extract condition image for i2i
            multi_modal_data = custom_prompt.get("multi_modal_data", None)
            if multi_modal_data is None:
                extra = custom_prompt.get("extra_args", {})
                if isinstance(extra, dict):
                    multi_modal_data = extra.get("multi_modal_data", {})
            if multi_modal_data:
                condition_image = multi_modal_data.get("image") or multi_modal_data.get("img2img")
                if isinstance(condition_image, list):
                    condition_image = condition_image[0] if condition_image else None

        # -- Override defaults from sampling_params ------------------------
        sp = req.sampling_params
        height = coalesce_not_none(getattr(sp, "height", None), height)
        width = coalesce_not_none(getattr(sp, "width", None), width)
        num_inference_steps = coalesce_not_none(getattr(sp, "num_inference_steps", None), num_inference_steps)
        max_sequence_length = coalesce_not_none(getattr(sp, "max_sequence_length", None), max_sequence_length)

        extra_args: dict = getattr(sp, "extra_args", None) or {}
        noise_level = coalesce_not_none(extra_args.get("noise_level"), noise_level)
        sde_window_size = coalesce_not_none(extra_args.get("sde_window_size"), sde_window_size)
        sde_window_range = coalesce_not_none(extra_args.get("sde_window_range"), sde_window_range)
        sde_type = coalesce_not_none(extra_args.get("sde_type"), sde_type)
        logprobs = coalesce_not_none(extra_args.get("logprobs"), logprobs)

        generator = getattr(sp, "generator", None) or generator
        seed = getattr(sp, "seed", None)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        true_cfg_scale = coalesce_not_none(getattr(sp, "true_cfg_scale", None), true_cfg_scale)
        req_n = getattr(sp, "num_outputs_per_prompt", None)
        if req_n and req_n > 0:
            num_images_per_prompt = req_n

        self._interrupt = False
        self._current_timestep = None

        # -- Encode prompt -------------------------------------------------
        if prompt_ids is not None or prompt_text is not None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt_text=prompt_text,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                max_sequence_length=max_sequence_length,
            )
        else:
            return DiffusionOutput(output=None, custom_output={})

        batch_size = prompt_embeds.shape[0]
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            prompt_embeds_mask = prompt_embeds_mask.repeat(num_images_per_prompt, 1)
            batch_size *= num_images_per_prompt

        # -- Encode condition image (i2i) ----------------------------------
        cond_vae_embeds = None
        cond_vit_embeds = None
        if condition_image is not None:
            cond_vae_embeds, cond_vit_embeds = self.encode_condition_image(
                condition_image, height, width
            )
            # Expand to batch
            if cond_vae_embeds is not None and cond_vae_embeds.shape[0] == 1 and batch_size > 1:
                cond_vae_embeds = cond_vae_embeds.expand(batch_size, -1, -1)
            if cond_vit_embeds is not None and cond_vit_embeds.shape[0] == 1 and batch_size > 1:
                cond_vit_embeds = cond_vit_embeds.expand(batch_size, -1, -1)

        # -- Prepare initial noise -----------------------------------------
        h = height // self.latent_downsample
        w = width // self.latent_downsample
        num_tokens = h * w
        x_t = torch.randn(
            batch_size,
            num_tokens,
            self.patch_latent_dim,
            device=self.device,
            generator=generator,
            dtype=prompt_embeds.dtype,
        )

        # -- Configure scheduler -------------------------------------------
        sigmas = compute_bagel_shifted_sigmas(num_inference_steps, timestep_shift)
        self.scheduler.set_timesteps(num_inference_steps, device=str(self.device), sigmas=sigmas)
        timesteps = self.scheduler.timesteps

        # -- SDE window ----------------------------------------------------
        if sde_window_size is not None:
            hi = max(sde_window_range[1] - sde_window_size + 1, sde_window_range[0] + 1)
            start = torch.randint(sde_window_range[0], hi, (1,), generator=generator, device=self.device).item()
            sde_window = (start, start + sde_window_size)
        else:
            sde_window = (0, len(timesteps) - 1)

        # -- Denoising loop ------------------------------------------------
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            x_t, all_latents, all_log_probs, all_timesteps = self.diffuse(
                x_t,
                prompt_embeds,
                prompt_embeds_mask,
                height,
                width,
                timesteps,
                noise_level,
                sde_window,
                sde_type,
                generator,
                logprobs,
                cond_vae_embeds=cond_vae_embeds,
                cond_vit_embeds=cond_vit_embeds,
            )

        self._current_timestep = None

        # -- Decode --------------------------------------------------------
        if output_type == "latent":
            image = x_t
        else:
            image = self._decode_latents(x_t, height, width)

        custom_output = {
            "all_latents": maybe_to_cpu(all_latents),
            "all_log_probs": maybe_to_cpu(all_log_probs),
            "all_timesteps": maybe_to_cpu(all_timesteps),
            "prompt_embeds": maybe_to_cpu(prompt_embeds),
            "prompt_embeds_mask": maybe_to_cpu(prompt_embeds_mask),
        }
        if cond_vae_embeds is not None:
            custom_output["cond_vae_embeds"] = maybe_to_cpu(cond_vae_embeds)
        if cond_vit_embeds is not None:
            custom_output["cond_vit_embeds"] = maybe_to_cpu(cond_vit_embeds)

        return DiffusionOutput(output=maybe_to_cpu(image), custom_output=custom_output)
