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
Utility functions specific to SenseNova-U1 FlowGRPO training.
"""

import math

import torch


def compute_noise_scale(
    grid_h: int,
    grid_w: int,
    merge_size: int,
    noise_scale: float,
    noise_scale_mode: str,
    noise_scale_base_image_seq_len: int,
    noise_scale_max_value: float,
) -> float:
    """Compute resolution-dependent noise scale for SenseNova-U1.

    Mirrors the logic in ``NEOChatModel.t2i_generate``.
    """
    if noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        base = float(noise_scale_base_image_seq_len)
        scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
        result = scale * float(noise_scale)
        if noise_scale_mode == "dynamic_sqrt":
            result = math.sqrt(result)
    else:
        result = noise_scale
    return min(result, noise_scale_max_value)


def compute_image_grid(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: float,
) -> tuple[int, int, int, int]:
    """Compute grid dimensions and token counts for SenseNova-U1.

    Returns:
        (grid_h, grid_w, token_h, token_w) where grid_h/w are full-resolution
        patch counts and token_h/w are the merged (downsampled) token counts.
    """
    grid_h = height // patch_size
    grid_w = width // patch_size
    merge_size = int(1 / downsample_ratio)
    token_h = height // (patch_size * merge_size)
    token_w = width // (patch_size * merge_size)
    return grid_h, grid_w, token_h, token_w


def apply_time_schedule(
    timesteps: torch.Tensor,
    image_seq_len: int,
    timestep_shift: float,
    time_schedule: str,
    time_shift_type: str,
    base_shift: float,
    max_shift: float,
    base_image_seq_len: int,
    max_image_seq_len: int,
) -> torch.Tensor:
    """Apply SenseNova-U1's custom timestep shifting.

    Mirrors ``NEOChatModel._apply_time_schedule``. Supports the ``standard``
    schedule with ``exponential`` and ``linear`` shift types.
    """
    if time_schedule == "standard":
        if time_shift_type == "exponential":
            mu = _calculate_dynamic_mu(
                image_seq_len, base_shift, max_shift, base_image_seq_len, max_image_seq_len
            )
            timesteps = _exponential_shift(timesteps, mu)
        elif time_shift_type == "linear":
            shift = timestep_shift
            timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        else:
            raise ValueError(f"Unknown time_shift_type: {time_shift_type}")
    elif time_schedule != "none":
        raise ValueError(f"Unknown time_schedule: {time_schedule}")
    return timesteps


def _calculate_dynamic_mu(
    image_seq_len: int,
    base_shift: float,
    max_shift: float,
    base_image_seq_len: int,
    max_image_seq_len: int,
) -> float:
    """Linear interpolation of shift factor based on image sequence length."""
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return m * image_seq_len + b


def _exponential_shift(timesteps: torch.Tensor, mu: float) -> torch.Tensor:
    """Apply exponential sigma shifting: sigma' = exp(mu) * sigma / (1 + (exp(mu) - 1) * sigma)."""
    exp_mu = math.exp(mu)
    return exp_mu * timesteps / (1 + (exp_mu - 1) * timesteps)


def build_timesteps(
    num_steps: int,
    image_seq_len: int,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    """Build the shifted timestep schedule for SenseNova-U1.

    Args:
        num_steps: Number of denoising steps.
        image_seq_len: Number of image tokens (token_h * token_w).
        config: Model config dict (from config.json).
        device: Target device.

    Returns:
        Tensor of shape (num_steps + 1,) with shifted timesteps from 0 to 1.
    """
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    timesteps = apply_time_schedule(
        timesteps,
        image_seq_len=image_seq_len,
        timestep_shift=config.get("timestep_shift", 1.0),
        time_schedule=config.get("time_schedule", "standard"),
        time_shift_type=config.get("time_shift_type", "exponential"),
        base_shift=config.get("base_shift", 0.5),
        max_shift=config.get("max_shift", 1.15),
        base_image_seq_len=config.get("base_image_seq_len", 64),
        max_image_seq_len=config.get("max_image_seq_len", 4096),
    )
    return timesteps


def v_pred_to_velocity(
    v_pred: torch.Tensor,
    t: torch.Tensor,
    t_eps: float,
) -> torch.Tensor:
    """Convert SenseNova-U1's v_pred to standard flow-matching velocity.

    SenseNova-U1 predicts ``v_pred = (x_pred - z) / (1 - t)``, i.e. a
    normalized displacement. The standard flow-matching velocity used by
    ``FlowMatchSDEDiscreteScheduler`` is ``dx/dt = v_pred * (1 - t)``
    (the unnormalized direction in the ODE). This function undoes the
    normalization so the scheduler receives the correct velocity.
    """
    scale = (1.0 - t).clamp_min(t_eps)
    return v_pred * scale


def apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Apply True-CFG guidance with norm rescaling."""
    comb_pred = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)
