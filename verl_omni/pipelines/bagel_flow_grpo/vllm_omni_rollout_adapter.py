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

NOTE: This is a scaffolding implementation. The vllm Bagel model currently
only supports text generation (understanding), not image generation. Full
Bagel image generation rollout requires:

1. A vllm-omni diffusion pipeline that can run Bagel's multi-step
   flow-matching image generation with SDE noise injection and
   log-probability collection.

2. Integration with Bagel's KV-cache-based inference flow:
   - Step 1: Prefill text prompt into KV cache
   - Step 2: Run flow-matching denoising loop with SDE noise
   - Step 3: Decode final latents through VAE
   - Step 4: Collect per-step latents, log-probs, and timesteps

3. The pipeline must use the MoT-aware forward pass for generation
   (``mode="gen"``), not the understanding mode.

The existing vllm Bagel model (``vllm.model_executor.models.bagel``)
explicitly skips generation weights (moe_gen, vae2llm, llm2vae,
time_embedder, etc.), so it cannot be used directly for image
generation rollout.

Possible approaches:
    A. Extend vllm-omni with a native Bagel diffusion pipeline that
       loads the full model weights (including generation components).
    B. Use a standalone PyTorch-based rollout (outside vllm) that
       runs BagelForConditionalGeneration directly with KV caching.
    C. Use a two-stage rollout: vllm for text AR generation (thinking
       tokens), then a separate flow-matching loop for image denoising.
"""

import logging
from typing import Any

import torch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase

__all__ = ["BagelPipelineWithLogProb"]

logger = logging.getLogger(__name__)


# NOTE: This registration is commented out until the rollout adapter is
# fully implemented. Uncomment when the vllm-omni Bagel diffusion
# pipeline is available.
#
# @VllmOmniPipelineBase.register("BagelForConditionalGeneration", algorithm="flow_grpo")
class BagelPipelineWithLogProb:
    """Rollout pipeline for Bagel flow-matching image generation with log-probs.

    This class will extend a Bagel diffusion pipeline (from vllm-omni or a
    custom implementation) to collect per-step SDE rollout data for FlowGRPO
    training.

    The rollout output should contain:
        - ``all_latents``: All intermediate latents from the SDE window,
          shape ``(B, W+1, num_tokens, patch_dim)``.
        - ``all_log_probs``: Per-step log-probabilities within the SDE window,
          shape ``(B, W)`` or ``None``.
        - ``all_timesteps``: Timesteps for each step in the SDE window,
          shape ``(B, W)``.
        - ``prompt_embeds``: Pre-computed text token embeddings from the LLM's
          embed_tokens layer, shape ``(B, L, D)``.
        - ``prompt_embeds_mask``: Attention mask for prompt embeddings,
          shape ``(B, L)``.

    Unlike Qwen-Image where prompt_embeds come from a separate text encoder,
    Bagel's prompt_embeds are the LLM's own token embeddings. During training,
    these embeddings are re-computed by the model's embed_tokens layer, so the
    rollout only needs to store the tokenized prompt and its length to
    reconstruct the forward pass.
    """

    def __init__(self, **kwargs: Any):
        raise NotImplementedError(
            "BagelPipelineWithLogProb rollout adapter is not yet implemented. "
            "See module docstring for implementation requirements and "
            "possible approaches."
        )
