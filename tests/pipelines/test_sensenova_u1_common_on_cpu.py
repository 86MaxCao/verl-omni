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
"""CPU tests for SenseNova-U1 common utility functions."""

import math

import pytest
import torch

from verl_omni.pipelines.sensenova_u1_flow_grpo.common import (
    apply_time_schedule,
    apply_true_cfg,
    build_timesteps,
    compute_image_grid,
    compute_noise_scale,
    v_pred_to_velocity,
)
from verl_omni.pipelines.sensenova_u1_flow_grpo.diffusers_training_adapter import (
    _build_image_indexes,
)


# ---------------------------------------------------------------------------
# compute_noise_scale
# ---------------------------------------------------------------------------


class TestComputeNoiseScale:
    def test_resolution_mode(self):
        # grid 32x32, merge 2 -> seq_len = 16*16 = 256, base 64
        # scale = sqrt(32*32 / 4 / 64) = sqrt(4) = 2.0
        # result = 2.0 * 1.0 = 2.0
        result = compute_noise_scale(
            grid_h=32, grid_w=32, merge_size=2,
            noise_scale=1.0, noise_scale_mode="resolution",
            noise_scale_base_image_seq_len=64, noise_scale_max_value=8.0,
        )
        assert result == pytest.approx(2.0)

    def test_dynamic_sqrt_mode(self):
        result = compute_noise_scale(
            grid_h=32, grid_w=32, merge_size=2,
            noise_scale=1.0, noise_scale_mode="dynamic_sqrt",
            noise_scale_base_image_seq_len=64, noise_scale_max_value=8.0,
        )
        assert result == pytest.approx(math.sqrt(2.0))

    def test_clamped_by_max_value(self):
        result = compute_noise_scale(
            grid_h=64, grid_w=64, merge_size=2,
            noise_scale=10.0, noise_scale_mode="resolution",
            noise_scale_base_image_seq_len=64, noise_scale_max_value=3.0,
        )
        assert result == pytest.approx(3.0)

    def test_static_mode(self):
        result = compute_noise_scale(
            grid_h=32, grid_w=32, merge_size=2,
            noise_scale=1.5, noise_scale_mode="static",
            noise_scale_base_image_seq_len=64, noise_scale_max_value=8.0,
        )
        assert result == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# compute_image_grid
# ---------------------------------------------------------------------------


class TestComputeImageGrid:
    def test_512x512(self):
        grid_h, grid_w, token_h, token_w = compute_image_grid(
            height=512, width=512, patch_size=16, downsample_ratio=0.5,
        )
        assert grid_h == 32
        assert grid_w == 32
        assert token_h == 16
        assert token_w == 16

    def test_256x512(self):
        grid_h, grid_w, token_h, token_w = compute_image_grid(
            height=256, width=512, patch_size=16, downsample_ratio=0.5,
        )
        assert grid_h == 16
        assert grid_w == 32
        assert token_h == 8
        assert token_w == 16


# ---------------------------------------------------------------------------
# apply_time_schedule / build_timesteps
# ---------------------------------------------------------------------------


class TestTimeSchedule:
    def test_exponential_shift_changes_timesteps(self):
        ts = torch.linspace(0, 1, 11)
        shifted = apply_time_schedule(
            ts, image_seq_len=256,
            timestep_shift=1.0, time_schedule="standard",
            time_shift_type="exponential",
            base_shift=0.5, max_shift=1.15,
            base_image_seq_len=64, max_image_seq_len=4096,
        )
        assert shifted.shape == ts.shape
        assert shifted[0] == pytest.approx(0.0, abs=1e-6)
        assert shifted[-1] == pytest.approx(1.0, abs=1e-6)
        assert not torch.allclose(shifted[1:-1], ts[1:-1])

    def test_linear_shift(self):
        ts = torch.linspace(0, 1, 11)
        shift = 3.0
        shifted = apply_time_schedule(
            ts, image_seq_len=256,
            timestep_shift=shift, time_schedule="standard",
            time_shift_type="linear",
            base_shift=0.5, max_shift=1.15,
            base_image_seq_len=64, max_image_seq_len=4096,
        )
        expected = shift * ts / (1 + (shift - 1) * ts)
        assert torch.allclose(shifted, expected)

    def test_none_schedule_passthrough(self):
        ts = torch.linspace(0, 1, 11)
        shifted = apply_time_schedule(
            ts, image_seq_len=256,
            timestep_shift=1.0, time_schedule="none",
            time_shift_type="exponential",
            base_shift=0.5, max_shift=1.15,
            base_image_seq_len=64, max_image_seq_len=4096,
        )
        assert torch.allclose(shifted, ts)

    def test_unknown_schedule_raises(self):
        ts = torch.linspace(0, 1, 11)
        with pytest.raises(ValueError, match="Unknown time_schedule"):
            apply_time_schedule(
                ts, image_seq_len=256,
                timestep_shift=1.0, time_schedule="bogus",
                time_shift_type="exponential",
                base_shift=0.5, max_shift=1.15,
                base_image_seq_len=64, max_image_seq_len=4096,
            )

    def test_unknown_shift_type_raises(self):
        ts = torch.linspace(0, 1, 11)
        with pytest.raises(ValueError, match="Unknown time_shift_type"):
            apply_time_schedule(
                ts, image_seq_len=256,
                timestep_shift=1.0, time_schedule="standard",
                time_shift_type="bogus",
                base_shift=0.5, max_shift=1.15,
                base_image_seq_len=64, max_image_seq_len=4096,
            )

    def test_build_timesteps_shape_and_monotonic(self):
        config = {
            "timestep_shift": 1.0,
            "time_schedule": "standard",
            "time_shift_type": "exponential",
            "base_shift": 0.5,
            "max_shift": 1.15,
            "base_image_seq_len": 64,
            "max_image_seq_len": 4096,
        }
        ts = build_timesteps(num_steps=20, image_seq_len=256, config=config, device="cpu")
        assert ts.shape == (21,)
        assert ts[0] == pytest.approx(0.0, abs=1e-6)
        assert ts[-1] == pytest.approx(1.0, abs=1e-6)
        diffs = ts[1:] - ts[:-1]
        assert (diffs > 0).all(), "timesteps must be strictly increasing"


# ---------------------------------------------------------------------------
# v_pred_to_velocity
# ---------------------------------------------------------------------------


class TestVPredToVelocity:
    def test_basic_conversion(self):
        v_pred = torch.randn(2, 256, 768)
        t = torch.tensor([0.3, 0.7]).view(2, 1, 1)
        velocity = v_pred_to_velocity(v_pred, t, t_eps=0.05)
        expected = v_pred * (1.0 - t)
        assert torch.allclose(velocity, expected)

    def test_near_t1_clamp(self):
        v_pred = torch.ones(1, 4, 8)
        t = torch.tensor([0.99]).view(1, 1, 1)
        velocity = v_pred_to_velocity(v_pred, t, t_eps=0.05)
        # (1 - 0.99) = 0.01, but clamp_min(0.05) -> 0.05
        expected = v_pred * 0.05
        assert torch.allclose(velocity, expected)


# ---------------------------------------------------------------------------
# apply_true_cfg
# ---------------------------------------------------------------------------


class TestApplyTrueCfg:
    def test_cfg_scale_1_is_identity(self):
        noise_pred = torch.randn(2, 256, 768)
        negative = torch.randn(2, 256, 768)
        result = apply_true_cfg(noise_pred, negative, true_cfg_scale=1.0)
        # scale=1 -> comb = neg + 1*(pos-neg) = pos, then norm rescale to same norm
        assert torch.allclose(result, noise_pred, atol=1e-5)

    def test_output_shape(self):
        noise_pred = torch.randn(2, 256, 768)
        negative = torch.randn(2, 256, 768)
        result = apply_true_cfg(noise_pred, negative, true_cfg_scale=2.0)
        assert result.shape == noise_pred.shape


# ---------------------------------------------------------------------------
# _build_image_indexes
# ---------------------------------------------------------------------------


class TestBuildImageIndexes:
    def test_shape_and_values(self):
        token_h, token_w = 16, 16
        text_len = 50
        indexes = _build_image_indexes(token_h, token_w, text_len, device="cpu")
        assert indexes.shape == (3, token_h * token_w)
        # t dimension should all be text_len
        assert (indexes[0] == text_len).all()
        # h dimension should be in [0, token_h)
        assert indexes[1].min() == 0
        assert indexes[1].max() == token_h - 1
        # w dimension should be in [0, token_w)
        assert indexes[2].min() == 0
        assert indexes[2].max() == token_w - 1
