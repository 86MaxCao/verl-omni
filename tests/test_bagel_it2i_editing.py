"""Test Bagel FlowGRPO it2i (image+text-to-image editing).

Verifies that:
  A) Condition image influences rollout output
  B) Text instruction influences rollout output
  C) Visual sanity check (decoded image saved for inspection)
  D) Training adapter packed-sequence correctness with real embeddings
  E) Condition vs no-condition velocity divergence (training side)

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_bagel_it2i_editing.py
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VLLM_OMNI_PATH = "/home/ximeng.czq/caoziqi/code/experiment/vllm-omni"
if VLLM_OMNI_PATH not in sys.path:
    sys.path.insert(0, VLLM_OMNI_PATH)

MODEL_PATH = "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT/"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

HEIGHT, WIDTH = 256, 256
NUM_STEPS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gradient_image(color_start, color_end, size=256):
    """Create a horizontal gradient image from color_start to color_end."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(size):
        t = x / (size - 1)
        arr[:, x] = (
            int(color_start[0] * (1 - t) + color_end[0] * t),
            int(color_start[1] * (1 - t) + color_end[1] * t),
            int(color_start[2] * (1 - t) + color_end[2] * t),
        )
    return Image.fromarray(arr)


def make_split_image(left_color, right_color, size=256):
    """Create a left/right split image."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, : size // 2] = left_color
    arr[:, size // 2 :] = right_color
    return Image.fromarray(arr)


def make_request(prompt_text, condition_image, height=HEIGHT, width=WIDTH, seed=42):
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
        num_inference_steps=NUM_STEPS,
        max_sequence_length=64,
        extra_args={
            "noise_level": 1.0,
            "sde_window_range": (0, 2),
            "sde_type": "sde",
            "logprobs": True,
        },
        seed=seed,
    )

    return FakeRequest(
        prompts=[{"prompt_text": prompt_text, "multi_modal_data": {"image": condition_image}}],
        sampling_params=sp,
    )


def make_text_only_request(prompt_text, height=HEIGHT, width=WIDTH, seed=42):
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
        num_inference_steps=NUM_STEPS,
        max_sequence_length=64,
        extra_args={
            "noise_level": 1.0,
            "sde_window_range": (0, 2),
            "sde_type": "sde",
            "logprobs": True,
        },
        seed=seed,
    )

    return FakeRequest(prompts=[prompt_text], sampling_params=sp)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_a_condition_image_influences_output(pipeline):
    """Different condition images → different latents (same text, same seed)."""
    print("\n[A] Condition image influences output...")

    red_gradient = make_gradient_image((200, 30, 30), (255, 100, 50))
    blue_gradient = make_gradient_image((30, 30, 200), (50, 100, 255))
    text = "make it brighter"

    out_red = pipeline.forward(make_request(text, red_gradient, seed=123))
    out_blue = pipeline.forward(make_request(text, blue_gradient, seed=123))

    latents_red = out_red.custom_output["all_latents"]
    latents_blue = out_blue.custom_output["all_latents"]

    assert latents_red.shape == latents_blue.shape, \
        f"Shape mismatch: {latents_red.shape} vs {latents_blue.shape}"

    diff = (latents_red - latents_blue).abs().mean().item()
    max_diff = (latents_red - latents_blue).abs().max().item()
    print(f"       mean abs diff: {diff:.4f}, max abs diff: {max_diff:.4f}")

    assert diff > 1e-6, f"Condition image had no effect! mean diff={diff}"
    print("       PASSED: different condition images produce different outputs")


def test_b_text_influences_output(pipeline):
    """Different text instructions → different latents (same condition, same seed)."""
    print("\n[B] Text instruction influences output...")

    cond_img = make_split_image((200, 50, 50), (50, 50, 200))
    text_a = "make everything warm and orange"
    text_b = "make everything cold and blue"

    out_a = pipeline.forward(make_request(text_a, cond_img, seed=456))
    out_b = pipeline.forward(make_request(text_b, cond_img, seed=456))

    latents_a = out_a.custom_output["all_latents"]
    latents_b = out_b.custom_output["all_latents"]

    diff = (latents_a - latents_b).abs().mean().item()
    max_diff = (latents_a - latents_b).abs().max().item()
    print(f"       mean abs diff: {diff:.4f}, max abs diff: {max_diff:.4f}")

    assert diff > 1e-6, f"Text instruction had no effect! mean diff={diff}"
    print("       PASSED: different text instructions produce different outputs")


def test_c_visual_sanity_check(pipeline):
    """Decode an it2i result and save for visual inspection."""
    print("\n[C] Visual sanity check (decoding + saving)...")

    cond_img = make_split_image((220, 40, 40), (40, 40, 220))
    text = "turn it into a green landscape"

    out = pipeline.forward(make_request(text, cond_img, seed=789))
    final_latent = out.custom_output["all_latents"][:, -1]  # last step

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save condition image
    cond_img.save(os.path.join(OUTPUT_DIR, "it2i_condition.png"))

    # Decode via VAE
    if pipeline.vae_model is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = pipeline._decode_latents(
                final_latent.to(pipeline.device), HEIGHT, WIDTH
            )
        img_np = (decoded[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        save_path = os.path.join(OUTPUT_DIR, "it2i_edit_green.png")
        img_pil.save(save_path)
        print(f"       Decoded image saved: {save_path}")
    else:
        print("       VAE not loaded, skipping decode. Saving latent stats only.")

    # Also run t2i for comparison (no condition image)
    out_t2i = pipeline.forward(make_text_only_request(text, seed=789))
    final_latent_t2i = out_t2i.custom_output["all_latents"][:, -1]

    if pipeline.vae_model is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            decoded_t2i = pipeline._decode_latents(
                final_latent_t2i.to(pipeline.device), HEIGHT, WIDTH
            )
        img_np_t2i = (decoded_t2i[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np_t2i).save(os.path.join(OUTPUT_DIR, "it2i_t2i_comparison.png"))
        print(f"       T2I comparison image saved: {OUTPUT_DIR}/it2i_t2i_comparison.png")

    diff_it2i_t2i = (final_latent - final_latent_t2i).abs().mean().item()
    print(f"       it2i vs t2i latent diff: {diff_it2i_t2i:.4f}")
    assert diff_it2i_t2i > 1e-6, "it2i and t2i produced identical latents!"
    print("       PASSED: it2i output differs from t2i output")


def test_d_training_packed_sequence_real_embeddings(pipeline):
    """Training adapter packed-sequence with real condition + text embeddings."""
    print("\n[D] Training adapter: packed sequence with real embeddings...")

    from tensordict import TensorDict
    from verl_omni.pipelines.bagel_flow_grpo.common import (
        BAGEL_LATENT_DOWNSAMPLE, BAGEL_PATCH_LATENT_DIM,
    )
    from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import (
        _build_bagel_model_forward_inputs,
        _bagel_flow_forward,
    )

    device = pipeline.device
    model = pipeline.model

    cond_img = make_gradient_image((180, 60, 20), (60, 180, 220))

    with torch.no_grad():
        cond_vae_embeds, cond_vit_embeds = pipeline.encode_condition_image(
            cond_img, HEIGHT, WIDTH
        )
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt_text="change the color to blue",
            max_sequence_length=32,
        )

    h = HEIGHT // BAGEL_LATENT_DOWNSAMPLE
    w = WIDTH // BAGEL_LATENT_DOWNSAMPLE
    num_latent_tokens = h * w
    x_t = torch.randn(1, num_latent_tokens, BAGEL_PATCH_LATENT_DIM, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], device=device, dtype=torch.float32)

    micro_batch = TensorDict({}, batch_size=[1])
    micro_batch.set_non_tensor("height", HEIGHT)
    micro_batch.set_non_tensor("width", WIDTH)

    n_cond_vae = cond_vae_embeds.shape[1]
    n_cond_vit = cond_vit_embeds.shape[1]
    text_len = int(prompt_embeds_mask.sum().item())

    model_inputs = _build_bagel_model_forward_inputs(
        module=model,
        x_t=x_t,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        micro_batch=micro_batch,
        cond_vae_embeds=cond_vae_embeds,
        cond_vit_embeds=cond_vit_embeds,
    )

    expected_total = n_cond_vae + n_cond_vit + text_len + num_latent_tokens
    actual_total = model_inputs["packed_sequence"].shape[0]
    print(f"       cond_vae={n_cond_vae}, cond_vit={n_cond_vit}, text={text_len}, latent={num_latent_tokens}")
    print(f"       total tokens: {actual_total} (expected {expected_total})")
    assert actual_total == expected_total

    n_und = model_inputs["packed_und_indexes"].shape[0]
    n_gen = model_inputs["packed_gen_indexes"].shape[0]
    expected_und = n_cond_vit + text_len
    expected_gen = n_cond_vae + num_latent_tokens
    print(f"       und_indexes: {n_und} (expected {expected_und})")
    print(f"       gen_indexes: {n_gen} (expected {expected_gen})")
    assert n_und == expected_und
    assert n_gen == expected_gen

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v_t = _bagel_flow_forward(model, model_inputs)

    expected_shape = (1, num_latent_tokens, BAGEL_PATCH_LATENT_DIM)
    assert v_t.shape == expected_shape, f"v_t shape: {v_t.shape} != {expected_shape}"
    assert not torch.isnan(v_t).any(), "v_t contains NaN!"
    assert not torch.isinf(v_t).any(), "v_t contains Inf!"
    print("       PASSED: real embeddings produce correct forward pass")


def test_e_condition_divergence(pipeline):
    """Velocity predictions diverge with vs without condition (training side)."""
    print("\n[E] Condition vs no-condition velocity divergence...")

    from tensordict import TensorDict
    from verl_omni.pipelines.bagel_flow_grpo.common import (
        BAGEL_LATENT_DOWNSAMPLE, BAGEL_PATCH_LATENT_DIM,
    )
    from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import (
        _build_bagel_model_forward_inputs,
        _bagel_flow_forward,
    )

    device = pipeline.device
    model = pipeline.model

    cond_img = make_split_image((200, 50, 50), (50, 200, 50))

    with torch.no_grad():
        cond_vae_embeds, cond_vit_embeds = pipeline.encode_condition_image(
            cond_img, HEIGHT, WIDTH
        )
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt_text="edit this image",
            max_sequence_length=16,
        )

    h = HEIGHT // BAGEL_LATENT_DOWNSAMPLE
    w = WIDTH // BAGEL_LATENT_DOWNSAMPLE
    num_latent_tokens = h * w
    x_t = torch.randn(1, num_latent_tokens, BAGEL_PATCH_LATENT_DIM, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], device=device, dtype=torch.float32)

    micro_batch = TensorDict({}, batch_size=[1])
    micro_batch.set_non_tensor("height", HEIGHT)
    micro_batch.set_non_tensor("width", WIDTH)

    # With condition
    inputs_cond = _build_bagel_model_forward_inputs(
        module=model,
        x_t=x_t,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        micro_batch=micro_batch,
        cond_vae_embeds=cond_vae_embeds,
        cond_vit_embeds=cond_vit_embeds,
    )

    # Without condition
    inputs_nocond = _build_bagel_model_forward_inputs(
        module=model,
        x_t=x_t,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        micro_batch=micro_batch,
        cond_vae_embeds=None,
        cond_vit_embeds=None,
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v_cond = _bagel_flow_forward(model, inputs_cond)
        v_nocond = _bagel_flow_forward(model, inputs_nocond)

    diff = (v_cond - v_nocond).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        v_cond.flatten(), v_nocond.flatten(), dim=0
    ).item()

    print(f"       mean abs diff: {diff:.4f}")
    print(f"       cosine similarity: {cos_sim:.4f}")

    assert diff > 1e-4, f"Condition had negligible effect on velocity! diff={diff}"
    assert cos_sim < 0.999, f"Condition barely changed velocity direction! cos_sim={cos_sim}"
    print("       PASSED: condition meaningfully changes velocity prediction")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("TEST: Bagel FlowGRPO it2i (Image+Text Editing)")
    print("=" * 60)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load pipeline once
    print("\nLoading BagelPipelineWithLogProb...")
    t0 = time.time()
    from dataclasses import dataclass

    @dataclass
    class FakeODConfig:
        model: str = MODEL_PATH

    from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb
    pipeline = BagelPipelineWithLogProb(od_config=FakeODConfig())
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"  has_vit={pipeline.has_vit}, vae_model={'loaded' if pipeline.vae_model else 'None'}")

    test_a_condition_image_influences_output(pipeline)
    test_b_text_influences_output(pipeline)
    test_c_visual_sanity_check(pipeline)
    test_d_training_packed_sequence_real_embeddings(pipeline)
    test_e_condition_divergence(pipeline)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
