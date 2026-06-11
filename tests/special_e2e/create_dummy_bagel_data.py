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
Create synthetic parquet datasets for Bagel FlowGRPO e2e testing.

Supports two modes:
  - t2i:  text-only prompts (same format as create_dummy_diffusion_data.py)
  - it2i: prompts with condition images for image editing

For it2i, colored PNG files are generated under DATA_DIR/images/ and
referenced in prompts using the qwen-vl multi-content message format
so that ``qwen_vl_utils.process_vision_info`` can extract them as PIL
images at runtime.

The dataset uses the jpeg_compressibility reward (a self-contained
rule-based reward that needs no external reward model).
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)

T2I_USER_PROMPTS = [
    "A red circle on a white background",
    "A blue square on a black background",
    "A green triangle next to an orange rectangle",
    "The word HELLO written in bold letters",
    "A yellow star above a purple crescent moon",
    "Two overlapping circles, one red and one blue",
    "A gradient from dark blue to light blue",
    "A checkerboard pattern of black and white squares",
]

IT2I_EDIT_INSTRUCTIONS = [
    "make it warmer and more orange",
    "turn the background into a gradient of blue tones",
    "add more green and make it look like a forest",
    "increase the contrast and make the colors more vivid",
    "shift all colors toward purple and pink hues",
    "make everything cooler with blue and teal tones",
    "convert to a warm sunset palette with red and gold",
    "brighten the entire image and add yellow highlights",
]

CONDITION_IMAGE_COLORS = [
    ((200, 30, 30), (255, 150, 50)),     # red-orange gradient
    ((30, 30, 200), (50, 150, 255)),     # blue gradient
    ((30, 180, 30), (150, 255, 50)),     # green gradient
    ((180, 180, 30), (255, 255, 100)),   # yellow gradient
    ((150, 30, 180), (255, 100, 255)),   # purple gradient
    ((30, 150, 150), (100, 255, 255)),   # teal gradient
    ((200, 100, 30), (255, 200, 100)),   # warm gradient
    ((100, 100, 100), (200, 200, 200)),  # gray gradient
]


def _make_gradient_image(color_start, color_end, size=256):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(size):
        t = x / (size - 1)
        arr[:, x] = (
            int(color_start[0] * (1 - t) + color_end[0] * t),
            int(color_start[1] * (1 - t) + color_end[1] * t),
            int(color_start[2] * (1 - t) + color_end[2] * t),
        )
    return Image.fromarray(arr)


def generate_condition_images(image_dir, n):
    os.makedirs(image_dir, exist_ok=True)
    paths = []
    for i in range(n):
        colors = CONDITION_IMAGE_COLORS[i % len(CONDITION_IMAGE_COLORS)]
        img = _make_gradient_image(colors[0], colors[1])
        path = os.path.join(image_dir, f"cond_{i:04d}.png")
        img.save(path)
        paths.append(path)
    return paths


def build_t2i_rows(split, n):
    rows = []
    for i in range(n):
        prompt_text = T2I_USER_PROMPTS[i % len(T2I_USER_PROMPTS)]
        rows.append(
            {
                "data_source": "jpeg_compressibility",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": " "},
                ],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def build_it2i_rows(split, n, image_paths):
    rows = []
    for i in range(n):
        edit_text = IT2I_EDIT_INSTRUCTIONS[i % len(IT2I_EDIT_INSTRUCTIONS)]
        img_path = image_paths[i % len(image_paths)]
        rows.append(
            {
                "data_source": "jpeg_compressibility",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image> {edit_text}"},
                ],
                "images": [{"image": img_path}],
                "negative_prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": " "},
                ],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate dummy Bagel parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_bagel"),
        help="Directory to write parquet and image files",
    )
    parser.add_argument(
        "--mode",
        choices=["t2i", "it2i", "both"],
        default="both",
        help="Data mode: t2i (text-to-image), it2i (image-to-image), or both",
    )
    parser.add_argument("--train_size", type=int, default=32, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=8, help="Number of validation samples")
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)

    if args.mode in ("t2i", "both"):
        t2i_dir = os.path.join(args.local_save_dir, "t2i")
        os.makedirs(t2i_dir, exist_ok=True)

        train_df = pd.DataFrame(build_t2i_rows("train", args.train_size))
        val_df = pd.DataFrame(build_t2i_rows("test", args.val_size))

        train_path = os.path.join(t2i_dir, "train.parquet")
        val_path = os.path.join(t2i_dir, "test.parquet")
        train_df.to_parquet(train_path)
        val_df.to_parquet(val_path)
        print(f"[t2i] Wrote {len(train_df)} train samples to {train_path}")
        print(f"[t2i] Wrote {len(val_df)} val samples to {val_path}")

    if args.mode in ("it2i", "both"):
        it2i_dir = os.path.join(args.local_save_dir, "it2i")
        os.makedirs(it2i_dir, exist_ok=True)
        image_dir = os.path.join(it2i_dir, "images")

        total_images = max(args.train_size, args.val_size)
        image_paths = generate_condition_images(image_dir, total_images)

        train_df = pd.DataFrame(build_it2i_rows("train", args.train_size, image_paths))
        val_df = pd.DataFrame(build_it2i_rows("test", args.val_size, image_paths))

        train_path = os.path.join(it2i_dir, "train.parquet")
        val_path = os.path.join(it2i_dir, "test.parquet")
        train_df.to_parquet(train_path)
        val_df.to_parquet(val_path)
        print(f"[it2i] Generated {len(image_paths)} condition images in {image_dir}")
        print(f"[it2i] Wrote {len(train_df)} train samples to {train_path}")
        print(f"[it2i] Wrote {len(val_df)} val samples to {val_path}")


if __name__ == "__main__":
    main()
