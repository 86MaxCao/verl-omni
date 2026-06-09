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

"""Bagel-specific constants and utilities for FlowGRPO training."""

import torch

# Bagel VAE scale factor: downsample=8 (from BagelVaeConfig default)
BAGEL_VAE_DOWNSAMPLE = 8

# Bagel latent patch size: 2 (from BagelConfig default)
BAGEL_LATENT_PATCH_SIZE = 2

# Effective downsampling: VAE downsample * latent patch size = 16
BAGEL_LATENT_DOWNSAMPLE = BAGEL_VAE_DOWNSAMPLE * BAGEL_LATENT_PATCH_SIZE

# VAE latent channels (from BagelVaeConfig default)
BAGEL_LATENT_CHANNELS = 16

# Patchified latent dimension: patch_size^2 * z_channels = 4 * 16 = 64
BAGEL_PATCH_LATENT_DIM = BAGEL_LATENT_PATCH_SIZE**2 * BAGEL_LATENT_CHANNELS

# Maximum latent grid size (from BagelConfig default, corrected to 64)
BAGEL_MAX_LATENT_SIZE = 64

# Default timestep shift (from BagelConfig)
BAGEL_DEFAULT_TIMESTEP_SHIFT = 1.0


def maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def coalesce_not_none(value, default):
    return default if value is None else value


def compute_bagel_shifted_sigmas(num_inference_steps: int, timestep_shift: float = 1.0):
    """Compute Bagel's shifted sigma schedule for the diffusers scheduler.

    Bagel uses flow matching with an optional timestep shift:
        t_shifted = shift * t / (1 + (shift - 1) * t)

    The diffusers ``FlowMatchEulerDiscreteScheduler.set_timesteps`` expects
    exactly ``num_inference_steps`` sigma values (it appends sigma=0 internally).
    We generate linearly spaced values from 1 to ``1/N``, then apply the shift.

    Args:
        num_inference_steps: Number of inference steps.
        timestep_shift: Timestep shift factor. 1.0 = no shift. Official Bagel
            uses 3.0 for generation inference, 1.0 for training.

    Returns:
        numpy array of sigmas with shape ``(num_inference_steps,)``, compatible
        with ``FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...)``.
    """
    import numpy as np

    # Generate linear sigmas: 1.0, (N-1)/N, ..., 1/N
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

    if timestep_shift != 1.0:
        sigmas_t = torch.from_numpy(sigmas).float()
        sigmas_shifted = timestep_shift * sigmas_t / (1.0 + (timestep_shift - 1.0) * sigmas_t)
        sigmas = sigmas_shifted.numpy()

    return sigmas


def get_flattened_position_ids(H, W, patch_size, max_num_patches_per_side):
    """Compute flattened position IDs for a 2D grid (extrapolate mode).

    This mirrors ``_get_flattened_position_ids_extrapolate`` in the Bagel model.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        patch_size: Effective patch size (latent_downsample for VAE tokens).
        max_num_patches_per_side: Maximum number of patches per side.

    Returns:
        Flattened position IDs tensor of shape (h * w,).
    """
    h = H // patch_size
    w = W // patch_size
    h_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
    w_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
    position_ids = h_ids * max_num_patches_per_side + w_ids
    return position_ids.flatten()
