"""Test Bagel FlowGRPO rollout adapter with i2i (condition image).

Loads the full BAGEL-7B-MoT model on a single GPU and runs:
1. encode_condition_image (VAE + ViT paths)
2. A short diffuse loop (3 steps) with condition embeddings

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_bagel_rollout_i2i.py
"""

import os
import sys
import time

import torch
from PIL import Image

# Ensure verl-omni is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ensure vllm-omni is importable
VLLM_OMNI_PATH = "/home/ximeng.czq/caoziqi/code/experiment/vllm-omni"
if VLLM_OMNI_PATH not in sys.path:
    sys.path.insert(0, VLLM_OMNI_PATH)

MODEL_PATH = "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT/"


def make_dummy_request(prompt_text: str, condition_image: Image.Image, height: int, width: int):
    """Create a minimal OmniDiffusionRequest-like object for testing."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class FakeSamplingParams:
        height: int = None
        width: int = None
        num_inference_steps: int = None
        max_sequence_length: int = None
        extra_args: dict = field(default_factory=dict)
        true_cfg_scale: float = None
        num_outputs_per_prompt: int = None
        generator: Any = None
        seed: int = None

    @dataclass
    class FakeRequest:
        prompts: list = field(default_factory=list)
        sampling_params: Any = None

    sp = FakeSamplingParams(
        height=height,
        width=width,
        num_inference_steps=3,
        max_sequence_length=64,
        extra_args={
            "noise_level": 1.0,
            "sde_window_range": (0, 2),
            "sde_type": "sde",
            "logprobs": True,
        },
        seed=42,
    )

    prompt_dict = {
        "prompt_text": prompt_text,
        "multi_modal_data": {"image": condition_image},
    }

    return FakeRequest(prompts=[prompt_dict], sampling_params=sp)


def main():
    print("=" * 60)
    print("TEST: Bagel FlowGRPO Rollout Adapter (i2i)")
    print("=" * 60)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Create a dummy condition image (256x256 RGB)
    print("\n[1/4] Creating dummy condition image...")
    cond_img = Image.new("RGB", (256, 256), color=(128, 64, 200))

    # Build the OmniDiffusionConfig-like object
    from dataclasses import dataclass

    @dataclass
    class FakeODConfig:
        model: str = MODEL_PATH

    od_config = FakeODConfig()

    # Load the pipeline
    print("[2/4] Loading BagelPipelineWithLogProb (this may take a while)...")
    t0 = time.time()
    from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb
    pipeline = BagelPipelineWithLogProb(od_config=od_config)
    print(f"       Model loaded in {time.time() - t0:.1f}s")
    print(f"       has_vit={pipeline.has_vit}, vae_model={'loaded' if pipeline.vae_model else 'None'}")

    # Test encode_condition_image
    print("[3/4] Testing encode_condition_image...")
    t0 = time.time()
    height, width = 256, 256
    cond_vae_embeds, cond_vit_embeds = pipeline.encode_condition_image(cond_img, height, width)
    print(f"       encode_condition_image done in {time.time() - t0:.2f}s")
    if cond_vae_embeds is not None:
        print(f"       cond_vae_embeds: {cond_vae_embeds.shape}, dtype={cond_vae_embeds.dtype}")
    else:
        print("       cond_vae_embeds: None (VAE not loaded)")
    if cond_vit_embeds is not None:
        print(f"       cond_vit_embeds: {cond_vit_embeds.shape}, dtype={cond_vit_embeds.dtype}")
    else:
        print("       cond_vit_embeds: None (ViT not available)")

    # Test full forward (end-to-end with condition image)
    print("[4/4] Testing full forward (3-step diffuse with i2i)...")
    req = make_dummy_request("a cat sitting on a red chair", cond_img, height, width)
    t0 = time.time()
    output = pipeline.forward(req)
    elapsed = time.time() - t0
    print(f"       forward done in {elapsed:.2f}s")

    co = output.custom_output
    print(f"       all_latents shape: {co['all_latents'].shape if co.get('all_latents') is not None else 'None'}")
    print(f"       all_log_probs shape: {co['all_log_probs'].shape if co.get('all_log_probs') is not None else 'None'}")
    print(f"       all_timesteps shape: {co['all_timesteps'].shape if co.get('all_timesteps') is not None else 'None'}")
    print(f"       prompt_embeds shape: {co['prompt_embeds'].shape}")
    if "cond_vae_embeds" in co:
        print(f"       cond_vae_embeds in output: {co['cond_vae_embeds'].shape}")
    if "cond_vit_embeds" in co:
        print(f"       cond_vit_embeds in output: {co['cond_vit_embeds'].shape}")

    # Verify shapes are consistent
    h = height // pipeline.latent_downsample
    w = width // pipeline.latent_downsample
    expected_tokens = h * w
    print(f"\n       Expected latent tokens: {expected_tokens} ({h}x{w})")
    if co.get("all_latents") is not None:
        assert co["all_latents"].shape[2] == expected_tokens, \
            f"Latent tokens mismatch: {co['all_latents'].shape[2]} != {expected_tokens}"
        print("       Shape consistency check PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
