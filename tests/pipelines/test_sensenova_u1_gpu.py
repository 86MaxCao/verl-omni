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
"""GPU tests for SenseNova-U1 model loading and forward pass.

Run with: CUDA_VISIBLE_DEVICES=2 pytest tests/pipelines/test_sensenova_u1_gpu.py -v -s
"""

import json
from unittest.mock import patch

import pytest
import torch

MODEL_PATH = "/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_config(load_tokenizer=False):
    """Build a DiffusionModelConfig for SenseNova-U1 without Hydra."""
    from verl_omni.workers.config.diffusion.model import DiffusionModelConfig

    cfg = DiffusionModelConfig.__new__(DiffusionModelConfig)
    cfg.path = MODEL_PATH
    cfg.architecture = "NEOChatModel"
    cfg.algorithm = "flow_grpo"
    cfg.model_type = "diffusion_model"
    cfg.local_path = None
    cfg.tokenizer_path = None
    cfg.local_tokenizer_path = None
    cfg.transformer_config = None
    cfg.use_shm = False
    cfg.trust_remote_code = True
    cfg.custom_chat_template = None
    cfg.external_lib = None
    cfg.enable_gradient_checkpointing = True
    cfg.attn_backend = "native"
    cfg.lora_rank = 0
    cfg.lora_alpha = 64
    cfg.lora_init_weights = "gaussian"
    cfg.target_modules = "all-linear"
    cfg.target_parameters = None
    cfg.exclude_modules = None
    cfg.lora = {}
    cfg.lora_adapter_path = None
    cfg.policy_state_adapters = ("default",)
    cfg.lora_dtype = None
    cfg.mtp = None
    cfg.fsdp_layer_prefixes = ["language_model.model.layers."]
    cfg.config_path = None
    cfg.transformer_subfolder = ""
    cfg.load_tokenizer = load_tokenizer

    from verl_omni.workers.config.diffusion.rollout import (
        DiffusionPipelineConfig,
        DiffusionRolloutAlgoConfig,
    )

    cfg.pipeline = DiffusionPipelineConfig(
        height=512, width=512, num_inference_steps=4,
        true_cfg_scale=1.0, max_sequence_length=256,
    )
    cfg.algo = DiffusionRolloutAlgoConfig(
        noise_level=1.0, sde_type="sde",
    )

    cfg.__post_init__()
    return cfg


# ---------------------------------------------------------------------------
# Module-scoped fixtures (model loaded once for all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_config():
    return _make_model_config()


@pytest.fixture(scope="module")
def model(model_config):
    import verl_omni.pipelines  # noqa: F401 — trigger registration
    from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

    module = SenseNovaU1.build_module(model_config, torch_dtype=torch.bfloat16)
    module = module.cuda().eval()
    return module


@pytest.fixture(scope="module")
def scheduler(model_config):
    from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

    return SenseNovaU1.build_scheduler(model_config)


# ===========================================================================
# Test Group 1: Registry & Loading
# ===========================================================================


class TestRegistryAndLoading:
    def test_registry_lookup(self, model_config):
        import verl_omni.pipelines  # noqa: F401
        from verl_omni.pipelines.model_base import DiffusionModelBase
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

        cls = DiffusionModelBase.get_class(model_config)
        assert cls is SenseNovaU1

    def test_build_module_type(self, model):
        assert model is not None
        assert hasattr(model, "language_model")
        assert hasattr(model, "fm_modules")
        assert hasattr(model, "patchify")
        assert hasattr(model, "extract_feature")

    def test_fm_modules_keys(self, model):
        fm = model.fm_modules
        assert "timestep_embedder" in fm
        assert "fm_head" in fm
        assert "noise_scale_embedder" in fm

    def test_build_scheduler(self, scheduler):
        from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

        assert isinstance(scheduler, FlowMatchSDEDiscreteScheduler)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) > 0

    def test_scheduler_timesteps_monotonic(self, scheduler):
        ts = scheduler.timesteps
        diffs = ts[1:] - ts[:-1]
        assert (diffs > 0).all()


# ===========================================================================
# Test Group 2: Forward Pass
# ===========================================================================


class TestForwardPass:
    B = 1
    H = W = 512
    TEXT_LEN = 8
    NUM_STEPS = 4

    @pytest.fixture(scope="class")
    def dummy_inputs(self, model, model_config, scheduler):
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1
        from verl_omni.pipelines.sensenova_u1_flow_grpo.common import build_timesteps, compute_image_grid

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        patch_size = 16
        _, _, token_h, token_w = compute_image_grid(self.H, self.W, patch_size, 0.5)
        image_seq_len = token_h * token_w
        hidden_dim = model.language_model.config.hidden_size

        latents = torch.randn(
            self.B, self.NUM_STEPS + 1, 3, self.H, self.W,
            device=device, dtype=dtype,
        )

        cfg = model_config.transformer_config or {}
        ts_1d = build_timesteps(self.NUM_STEPS, image_seq_len, cfg, device)
        timesteps = ts_1d.unsqueeze(0).expand(self.B, -1)

        prompt_embeds = torch.randn(
            self.B, self.TEXT_LEN, hidden_dim, device=device, dtype=dtype,
        )
        prompt_mask = torch.ones(self.B, self.TEXT_LEN, device=device, dtype=torch.bool)
        neg_embeds = torch.randn_like(prompt_embeds)
        neg_mask = torch.ones_like(prompt_mask)

        return {
            "latents": latents,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "neg_embeds": neg_embeds,
            "neg_mask": neg_mask,
            "image_seq_len": image_seq_len,
            "hidden_dim": hidden_dim,
        }

    def test_prepare_model_inputs_shapes(self, model, model_config, dummy_inputs):
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

        step = 0
        model_inputs, neg_inputs = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_inputs["latents"],
            timesteps=dummy_inputs["timesteps"],
            prompt_embeds=dummy_inputs["prompt_embeds"],
            prompt_embeds_mask=dummy_inputs["prompt_mask"],
            negative_prompt_embeds=dummy_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        assert "input_embeds" in model_inputs
        assert "indexes" in model_inputs
        assert "t" in model_inputs
        assert "z" in model_inputs
        assert "image_token_num" in model_inputs

        seq_len = self.TEXT_LEN + dummy_inputs["image_seq_len"]
        assert model_inputs["input_embeds"].shape == (
            self.B, seq_len, dummy_inputs["hidden_dim"],
        )
        assert model_inputs["indexes"].shape == (3, seq_len)
        assert model_inputs["t"].shape == (self.B,)

        # true_cfg_scale == 1.0, so neg_inputs should be None
        assert neg_inputs is None

    @torch.no_grad()
    def test_forward_output_shape(self, model, model_config, dummy_inputs):
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

        step = 0
        model_inputs, _ = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_inputs["latents"],
            timesteps=dummy_inputs["timesteps"],
            prompt_embeds=dummy_inputs["prompt_embeds"],
            prompt_embeds_mask=dummy_inputs["prompt_mask"],
            negative_prompt_embeds=dummy_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        velocity = SenseNovaU1.forward(model, model_config, model_inputs)
        z_shape = model_inputs["z"].shape
        assert velocity.shape == z_shape
        assert torch.isfinite(velocity).all()

    @torch.no_grad()
    def test_forward_and_sample_previous_step(self, model, model_config, scheduler, dummy_inputs):
        from tensordict import TensorDict
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

        step = 0
        model_inputs, neg_inputs = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_inputs["latents"],
            timesteps=dummy_inputs["timesteps"],
            prompt_embeds=dummy_inputs["prompt_embeds"],
            prompt_embeds_mask=dummy_inputs["prompt_mask"],
            negative_prompt_embeds=dummy_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        scheduler_inputs = {
            "all_latents": dummy_inputs["latents"],
            "all_timesteps": dummy_inputs["timesteps"],
        }

        log_prob, prev_sample_mean, std_dev_t, sqrt_dt = (
            SenseNovaU1.forward_and_sample_previous_step(
                module=model,
                scheduler=scheduler,
                model_config=model_config,
                model_inputs=model_inputs,
                negative_model_inputs=neg_inputs,
                scheduler_inputs=scheduler_inputs,
                step=step,
            )
        )

        assert log_prob.shape == (self.B,)
        assert torch.isfinite(log_prob).all()
        assert prev_sample_mean is not None
        assert std_dev_t is not None
        assert sqrt_dt is not None
