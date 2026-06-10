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
"""GPU tests for SenseNova-U1 it2i (image-to-image) support in the training adapter.

Validates that the training adapter correctly handles variable-length prompt_embeds
that result from the it2i rollout path (where prompt_embeds contain image embeddings).

Run with: CUDA_VISIBLE_DEVICES=2 pytest tests/pipelines/test_sensenova_u1_it2i_gpu.py -v -s
"""

import pytest
import torch

MODEL_PATH = "/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT"


# ---------------------------------------------------------------------------
# Fixtures (shared across all tests; model loaded once)
# ---------------------------------------------------------------------------


def _make_model_config():
    """Build a DiffusionModelConfig for SenseNova-U1."""
    from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
    from verl_omni.workers.config.diffusion.rollout import (
        DiffusionPipelineConfig,
        DiffusionRolloutAlgoConfig,
    )

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
    cfg.load_tokenizer = False
    cfg.pipeline = DiffusionPipelineConfig(
        height=512, width=512, num_inference_steps=4,
        true_cfg_scale=1.0, max_sequence_length=256,
    )
    cfg.algo = DiffusionRolloutAlgoConfig(noise_level=1.0, sde_type="sde")
    cfg.__post_init__()
    return cfg


@pytest.fixture(scope="module")
def model_config():
    return _make_model_config()


@pytest.fixture(scope="module")
def model(model_config):
    import verl_omni.pipelines  # noqa: F401
    from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

    module = SenseNovaU1.build_module(model_config, torch_dtype=torch.bfloat16)
    module = module.cuda().eval()
    return module


@pytest.fixture(scope="module")
def scheduler(model_config):
    from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

    return SenseNovaU1.build_scheduler(model_config)


# ===========================================================================
# Test Group: it2i training adapter with longer prompt_embeds
# ===========================================================================


class TestIt2iTrainingAdapter:
    """Test training adapter with it2i-shaped inputs.

    In it2i mode, the rollout adapter produces prompt_embeds that contain
    image token embeddings (from the input image), making them longer than
    in t2i mode. The training adapter must handle this correctly.
    """

    B = 1
    H = W = 512
    NUM_STEPS = 4
    # In it2i mode, prompt includes text tokens + image tokens (e.g., 256 image tokens for a 512x512 input)
    IT2I_TEXT_LEN = 64  # text portion
    IT2I_IMG_TOKENS = 256  # image tokens embedded into the prompt
    IT2I_PROMPT_LEN = IT2I_TEXT_LEN + IT2I_IMG_TOKENS  # total prompt_embeds length

    @pytest.fixture(scope="class")
    def dummy_it2i_inputs(self, model, model_config, scheduler):
        from verl_omni.pipelines.sensenova_u1_flow_grpo.common import (
            build_timesteps,
            compute_image_grid,
        )

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        hidden_dim = model.language_model.config.hidden_size

        patch_size = 16
        _, _, token_h, token_w = compute_image_grid(self.H, self.W, patch_size, 0.5)
        image_seq_len = token_h * token_w

        # Latents trajectory: (B, num_steps+1, 3, H, W)
        latents = torch.randn(
            self.B, self.NUM_STEPS + 1, 3, self.H, self.W,
            device=device, dtype=dtype,
        )

        # Timesteps: build from config
        cfg = model_config.transformer_config or {}
        ts_1d = build_timesteps(self.NUM_STEPS, image_seq_len, cfg, device)
        timesteps = ts_1d.unsqueeze(0).expand(self.B, -1)

        # it2i prompt_embeds: longer than t2i (includes image embeddings)
        prompt_embeds = torch.randn(
            self.B, self.IT2I_PROMPT_LEN, hidden_dim,
            device=device, dtype=dtype,
        )
        prompt_mask = torch.ones(
            self.B, self.IT2I_PROMPT_LEN, device=device, dtype=torch.bool
        )

        # Negative prompt (typically shorter, text-only)
        neg_text_len = 16
        neg_embeds = torch.randn(
            self.B, neg_text_len, hidden_dim, device=device, dtype=dtype,
        )
        neg_mask = torch.ones(self.B, neg_text_len, device=device, dtype=torch.bool)

        return {
            "latents": latents,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "neg_embeds": neg_embeds,
            "neg_mask": neg_mask,
            "hidden_dim": hidden_dim,
            "image_seq_len": image_seq_len,
        }

    def test_prepare_model_inputs_it2i_shapes(self, model, model_config, dummy_it2i_inputs):
        """prepare_model_inputs works with it2i-length prompt_embeds."""
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

        step = 0
        model_inputs, neg_inputs = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_it2i_inputs["latents"],
            timesteps=dummy_it2i_inputs["timesteps"],
            prompt_embeds=dummy_it2i_inputs["prompt_embeds"],
            prompt_embeds_mask=dummy_it2i_inputs["prompt_mask"],
            negative_prompt_embeds=dummy_it2i_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_it2i_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        expected_seq_len = self.IT2I_PROMPT_LEN + dummy_it2i_inputs["image_seq_len"]
        assert model_inputs["input_embeds"].shape == (
            self.B, expected_seq_len, dummy_it2i_inputs["hidden_dim"],
        ), f"Expected seq_len {expected_seq_len}, got {model_inputs['input_embeds'].shape[1]}"

        # 3D indexes should match the full sequence length
        assert model_inputs["indexes"].shape == (3, expected_seq_len)

        # Image token count should be just the denoising image tokens
        assert model_inputs["image_token_num"] == dummy_it2i_inputs["image_seq_len"]

        print(f"\nit2i model_inputs shape: {model_inputs['input_embeds'].shape}")
        print(f"text_len (with image tokens): {self.IT2I_PROMPT_LEN}")
        print(f"denoising image tokens: {dummy_it2i_inputs['image_seq_len']}")
        print(f"total seq_len: {expected_seq_len}")

    @torch.no_grad()
    def test_forward_it2i_output_shape(self, model, model_config, dummy_it2i_inputs):
        """forward() produces correct velocity shape with it2i inputs."""
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

        step = 0
        model_inputs, _ = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_it2i_inputs["latents"],
            timesteps=dummy_it2i_inputs["timesteps"],
            prompt_embeds=dummy_it2i_inputs["prompt_embeds"],
            prompt_embeds_mask=dummy_it2i_inputs["prompt_mask"],
            negative_prompt_embeds=dummy_it2i_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_it2i_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        velocity = SenseNovaU1.forward(model, model_config, model_inputs)

        # velocity should have same shape as z (patchified latent)
        z_shape = model_inputs["z"].shape
        assert velocity.shape == z_shape, f"Expected {z_shape}, got {velocity.shape}"
        assert torch.isfinite(velocity).all(), "velocity contains non-finite values"

        print(f"\nit2i velocity shape: {velocity.shape}")
        print(f"velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")

    @torch.no_grad()
    def test_forward_and_sample_it2i(self, model, model_config, scheduler, dummy_it2i_inputs):
        """forward_and_sample_previous_step works with it2i inputs."""
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1

        step = 0
        model_inputs, neg_inputs = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_it2i_inputs["latents"],
            timesteps=dummy_it2i_inputs["timesteps"],
            prompt_embeds=dummy_it2i_inputs["prompt_embeds"],
            prompt_embeds_mask=dummy_it2i_inputs["prompt_mask"],
            negative_prompt_embeds=dummy_it2i_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_it2i_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        scheduler_inputs = {
            "all_latents": dummy_it2i_inputs["latents"],
            "all_timesteps": dummy_it2i_inputs["timesteps"],
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

        assert log_prob.shape == (self.B,), f"Expected ({self.B},), got {log_prob.shape}"
        assert torch.isfinite(log_prob).all(), f"log_prob non-finite: {log_prob}"
        assert prev_sample_mean is not None
        assert std_dev_t is not None
        assert sqrt_dt is not None

        print(f"\nit2i log_prob: {log_prob}")
        print(f"it2i prev_sample_mean shape: {prev_sample_mean.shape}")

    @torch.no_grad()
    def test_it2i_vs_t2i_different_output(self, model, model_config, dummy_it2i_inputs):
        """it2i and t2i inputs produce different velocity outputs."""
        from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import SenseNovaU1
        from verl_omni.pipelines.sensenova_u1_flow_grpo.common import (
            build_timesteps,
            compute_image_grid,
        )

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        hidden_dim = model.language_model.config.hidden_size

        # t2i inputs (short prompt_embeds)
        t2i_text_len = 8
        t2i_prompt = torch.randn(
            self.B, t2i_text_len, hidden_dim, device=device, dtype=dtype,
        )
        t2i_mask = torch.ones(self.B, t2i_text_len, device=device, dtype=torch.bool)

        step = 0
        model_inputs_t2i, _ = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_it2i_inputs["latents"],
            timesteps=dummy_it2i_inputs["timesteps"],
            prompt_embeds=t2i_prompt,
            prompt_embeds_mask=t2i_mask,
            negative_prompt_embeds=dummy_it2i_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_it2i_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        model_inputs_it2i, _ = SenseNovaU1.prepare_model_inputs(
            module=model,
            model_config=model_config,
            latents=dummy_it2i_inputs["latents"],
            timesteps=dummy_it2i_inputs["timesteps"],
            prompt_embeds=dummy_it2i_inputs["prompt_embeds"],
            prompt_embeds_mask=dummy_it2i_inputs["prompt_mask"],
            negative_prompt_embeds=dummy_it2i_inputs["neg_embeds"],
            negative_prompt_embeds_mask=dummy_it2i_inputs["neg_mask"],
            micro_batch=None,
            step=step,
        )

        vel_t2i = SenseNovaU1.forward(model, model_config, model_inputs_t2i)
        vel_it2i = SenseNovaU1.forward(model, model_config, model_inputs_it2i)

        # Different prompt lengths should produce different outputs
        assert not torch.allclose(vel_t2i, vel_it2i, atol=1e-3), (
            "t2i and it2i should produce different velocities"
        )

        print(f"\nt2i input_embeds shape: {model_inputs_t2i['input_embeds'].shape}")
        print(f"it2i input_embeds shape: {model_inputs_it2i['input_embeds'].shape}")
        print(f"Velocity difference norm: {(vel_t2i - vel_it2i).norm():.4f}")
