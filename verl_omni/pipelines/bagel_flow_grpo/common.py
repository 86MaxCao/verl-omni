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

import numpy as np
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


def patchify_for_vit(image_tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchify an image tensor for ViT input.

    Converts a (C, H, W) tensor into (num_patches, patch_size^2 * C) patches.

    Args:
        image_tensor: Image tensor of shape (C, H, W).
        patch_size: Size of each patch.

    Returns:
        Flattened patches of shape (num_patches, patch_size^2 * C).
    """
    C, H, W = image_tensor.shape
    h = H // patch_size
    w = W // patch_size
    patches = image_tensor[:, :h * patch_size, :w * patch_size]
    patches = patches.reshape(C, h, patch_size, w, patch_size)
    patches = patches.permute(1, 3, 2, 4, 0)  # (h, w, p, p, C)
    patches = patches.reshape(h * w, patch_size * patch_size * C)
    return patches


def resize_image_to_stride(image, stride: int, max_size: int, min_size: int = 256):
    """Resize a PIL image so dimensions are multiples of stride within bounds.

    Args:
        image: PIL.Image input.
        stride: Target stride alignment.
        max_size: Maximum allowed dimension.
        min_size: Minimum allowed dimension.

    Returns:
        Resized PIL.Image.
    """
    from PIL import Image as PILImage

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    scale = min(max_size / max(w, h), 1.0)
    scale = max(scale, min_size / min(w, h))
    new_w = max(stride, int(round(w * scale / stride) * stride))
    new_h = max(stride, int(round(h * scale / stride) * stride))
    new_w = min(new_w, max_size)
    new_h = min(new_h, max_size)
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), PILImage.BICUBIC)
    return image


def image_to_vae_input(image) -> torch.Tensor:
    """Convert a PIL image to VAE input tensor in [-1, 1].

    Args:
        image: PIL.Image in RGB mode.

    Returns:
        Tensor of shape (C, H, W) in [-1, 1].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
    return arr.permute(2, 0, 1)


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
