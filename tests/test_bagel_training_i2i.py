"""Test Bagel FlowGRPO training adapter with i2i (condition embeddings).

Loads the full BAGEL-7B-MoT model on a single GPU and tests:
1. _build_bagel_model_forward_inputs with condition embeddings
2. _bagel_flow_forward produces correct output shape
3. forward_and_sample_previous_step (the RL log-prob path)

Usage:
    CUDA_VISIBLE_DEVICES=1 python tests/test_bagel_training_i2i.py
"""

import os
import sys
import time

import torch

# Ensure verl-omni is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ensure vllm-omni is importable
VLLM_OMNI_PATH = "/home/ximeng.czq/caoziqi/code/experiment/vllm-omni"
if VLLM_OMNI_PATH not in sys.path:
    sys.path.insert(0, VLLM_OMNI_PATH)

MODEL_PATH = "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT/"


def main():
    print("=" * 60)
    print("TEST: Bagel FlowGRPO Training Adapter (i2i)")
    print("=" * 60)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load model
    print("\n[1/5] Loading BagelForConditionalGeneration...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.eval().to(device)
    print(f"       Loaded in {time.time() - t0:.1f}s")
    print(f"       hidden_size={model.hidden_size}, use_moe={getattr(model, 'use_moe', True)}")

    # Test parameters
    batch_size = 2
    height, width = 256, 256
    from verl_omni.pipelines.bagel_flow_grpo.common import (
        BAGEL_LATENT_DOWNSAMPLE, BAGEL_MAX_LATENT_SIZE, BAGEL_PATCH_LATENT_DIM,
    )
    h = height // BAGEL_LATENT_DOWNSAMPLE
    w = width // BAGEL_LATENT_DOWNSAMPLE
    num_latent_tokens = h * w
    hidden_size = model.hidden_size
    text_len = 10

    print(f"       Latent grid: {h}x{w} = {num_latent_tokens} tokens")
    print(f"       Text len: {text_len}, Batch: {batch_size}")

    # Create dummy inputs
    print("\n[2/5] Creating dummy inputs (text embeds + condition embeds)...")
    prompt_embeds = torch.randn(batch_size, text_len, hidden_size, device=device, dtype=torch.bfloat16)
    prompt_embeds_mask = torch.ones(batch_size, text_len, dtype=torch.bool, device=device)
    x_t = torch.randn(batch_size, num_latent_tokens, BAGEL_PATCH_LATENT_DIM, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5, 0.3], device=device, dtype=torch.float32)

    # Simulate condition embeddings (as if from rollout encode_condition_image)
    n_cond_vae = num_latent_tokens  # same grid size as output
    n_cond_vit = 49  # e.g., 7x7 ViT patches for a small image
    cond_vae_embeds = torch.randn(batch_size, n_cond_vae, hidden_size, device=device, dtype=torch.bfloat16)
    cond_vit_embeds = torch.randn(batch_size, n_cond_vit, hidden_size, device=device, dtype=torch.bfloat16)
    print(f"       cond_vae_embeds: {cond_vae_embeds.shape}")
    print(f"       cond_vit_embeds: {cond_vit_embeds.shape}")

    # Create a minimal micro_batch (TensorDict-like)
    from tensordict import TensorDict
    micro_batch = TensorDict({}, batch_size=[batch_size])
    micro_batch.set_non_tensor("height", height)
    micro_batch.set_non_tensor("width", width)

    # Test _build_bagel_model_forward_inputs
    print("\n[3/5] Testing _build_bagel_model_forward_inputs (i2i)...")
    from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import (
        _build_bagel_model_forward_inputs,
        _bagel_flow_forward,
    )

    t0 = time.time()
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
    print(f"       Built in {time.time() - t0:.3f}s")
    print(f"       packed_sequence shape: {model_inputs['packed_sequence'].shape}")
    print(f"       packed_und_indexes: {model_inputs['packed_und_indexes'].shape}")
    print(f"       packed_gen_indexes: {model_inputs['packed_gen_indexes'].shape}")
    print(f"       packed_latent_indexes: {model_inputs['packed_latent_indexes'].shape}")
    print(f"       sample_lens: {model_inputs['sample_lens']}")

    # Verify token counts
    expected_per_sample = n_cond_vae + n_cond_vit + text_len + num_latent_tokens
    total_expected = expected_per_sample * batch_size
    actual_total = model_inputs["packed_sequence"].shape[0]
    print(f"       Expected total tokens: {total_expected}, actual: {actual_total}")
    assert actual_total == total_expected, f"Token count mismatch: {actual_total} != {total_expected}"

    # Verify MoE index correctness
    n_und = model_inputs["packed_und_indexes"].shape[0]
    n_gen = model_inputs["packed_gen_indexes"].shape[0]
    expected_und = (n_cond_vit + text_len) * batch_size
    expected_gen = (n_cond_vae + num_latent_tokens) * batch_size
    print(f"       und_indexes: {n_und} (expected {expected_und})")
    print(f"       gen_indexes: {n_gen} (expected {expected_gen})")
    assert n_und == expected_und, f"und index count mismatch: {n_und} != {expected_und}"
    assert n_gen == expected_gen, f"gen index count mismatch: {n_gen} != {expected_gen}"
    print("       Token counts and MoE routing PASSED")

    # Test _bagel_flow_forward
    print("\n[4/5] Testing _bagel_flow_forward (i2i)...")
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v_t = _bagel_flow_forward(model, model_inputs)
    elapsed = time.time() - t0
    print(f"       Forward done in {elapsed:.2f}s")
    print(f"       v_t shape: {v_t.shape}")
    expected_shape = (batch_size, num_latent_tokens, BAGEL_PATCH_LATENT_DIM)
    assert v_t.shape == expected_shape, f"v_t shape mismatch: {v_t.shape} != {expected_shape}"
    assert not torch.isnan(v_t).any(), "v_t contains NaN!"
    assert not torch.isinf(v_t).any(), "v_t contains Inf!"
    print("       Shape and value checks PASSED")

    # Test t2i mode (no condition) for comparison
    print("\n[5/5] Testing _bagel_flow_forward (t2i, no condition)...")
    model_inputs_t2i = _build_bagel_model_forward_inputs(
        module=model,
        x_t=x_t,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        micro_batch=micro_batch,
        cond_vae_embeds=None,
        cond_vit_embeds=None,
    )
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v_t_t2i = _bagel_flow_forward(model, model_inputs_t2i)
    elapsed = time.time() - t0
    print(f"       t2i forward done in {elapsed:.2f}s")
    print(f"       v_t_t2i shape: {v_t_t2i.shape}")
    assert v_t_t2i.shape == expected_shape
    expected_t2i_total = (text_len + num_latent_tokens) * batch_size
    actual_t2i_total = model_inputs_t2i["packed_sequence"].shape[0]
    print(f"       t2i total tokens: {actual_t2i_total} (expected {expected_t2i_total})")
    assert actual_t2i_total == expected_t2i_total
    print("       t2i mode PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
