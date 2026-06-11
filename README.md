<div align="center">

# VeRL-Omni

### Easy, fast, and stable RL training for diffusion and omni-modality models

[![Docs](https://img.shields.io/badge/docs-Read%20the%20Docs-8A2BE2)](https://verl-omni.readthedocs.io/en/latest/index.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE) <a href="docs/assets/WeChat.jpg"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

## BAGEL FlowGRPO Adaptation

This branch adds **FlowGRPO RL training support for [BAGEL-7B-MoT](https://github.com/ByteDance-Seed/Bagel)**, a unified multimodal model with Mixture-of-Transformers (MoT) architecture that jointly supports image understanding and generation.

### Key Features

- **Text-to-Image (t2i)**: Generate images from text prompts with RL-based optimization via FlowGRPO.
- **Image-to-Image (it2i)**: Edit images with text instructions — the condition image is encoded via VAE + ViT and packed into the sequence alongside text and noisy latent tokens at each denoising step.
- **Rollout/Training Consistency**: Both rollout and training use identical packed-sequence forward passes (no KV cache), ensuring log-probability consistency for RL.

### Architecture

Unlike pure DiT models (e.g., Qwen-Image), Bagel's flow-matching diffusion runs inside the LLM itself:

- **Token layout (t2i)**: `[text | noisy_latent]`
- **Token layout (it2i)**: `[cond_vae | cond_vit | text | noisy_latent]`
- **MoE routing**: `cond_vae → gen`, `cond_vit → und`, `text → und`, `noisy_latent → gen`

### Quick Start

```bash
# 1. Install dependencies
pip install "vllm==0.20.2"
pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@c7178d89bb7a70817f239febc84c3b21a714dae7"
pip install "verl==0.8.0"
pip install -e .

# 2. Run Bagel FlowGRPO training (t2i)
NUM_GPUS=4 bash examples/flowgrpo_trainer/run_bagel_gen_debug.sh

# 3. Run e2e tests
bash tests/special_e2e/run_flowgrpo_bagel_t2i.sh   # text-to-image
bash tests/special_e2e/run_flowgrpo_bagel_it2i.sh   # image-to-image editing
```

### Added Files

| File | Description |
|---|---|
| `verl_omni/pipelines/bagel_flow_grpo/common.py` | Bagel constants and utilities (patchify, sigma schedule, position IDs) |
| `verl_omni/pipelines/bagel_flow_grpo/vllm_omni_rollout_adapter.py` | Rollout adapter: standalone pipeline with SDE log-prob collection |
| `verl_omni/pipelines/bagel_flow_grpo/diffusers_training_adapter.py` | Training adapter: packed-sequence forward, `forward_and_sample_previous_step` |
| `examples/flowgrpo_trainer/run_bagel_gen_debug.sh` | Example launch script for Bagel FlowGRPO training |
| `tests/special_e2e/run_flowgrpo_bagel_t2i.sh` | E2E smoke test for t2i |
| `tests/special_e2e/run_flowgrpo_bagel_it2i.sh` | E2E smoke test for it2i |
| `tests/special_e2e/create_dummy_bagel_data.py` | Synthetic data generator for e2e tests |

---

`VeRL-Omni` is a general RL training framework focused on multimodal generative models, built on top of [`verl`](https://github.com/verl-project/verl).

It originated from the multi-modal generation RL effort in `verl`, and now has a dedicated home so it can evolve in a more focused way.

## Why `VeRL-Omni`

Multimodal generative RL training differs from text-only LLM RL not only in model structure, but also in I/O patterns, compute characteristics, and runtime bottlenecks. As this space grows, it deserves a dedicated training repository that can evolve quickly around its own constraints.

### Scope

`VeRL-Omni` targets RL post-training for three families of generative models:

1. **Diffusion generative models** for image, video, and audio — e.g., Qwen-Image, Wan2.2.
2. **Unified multimodal understanding + generation models** — e.g., BAGEL, HunyuanImage-3.0.
3. **Omni-modality models** that jointly handle text, image, audio, and video — e.g., Qwen3-Omni.

### What we focus on

- **Specialized rollout** via [`vLLM-Omni`](https://github.com/vllm-project/vllm-omni) for high-throughput diffusion and multimodal generation.
- **Flexible reward pipelines** spanning rule-based rewards, model-based rewards, and multimodal reward computation.
- **Modular training backends** that plug into existing parallelism (FSDP, USP) and other optimizations rather than rebuilding the stack from scratch.
- **End-to-end examples and benchmarks** validating co-located sync and fully-async RL on the model families above.
- **High training throughput** — on our reference Qwen-Image FlowGRPO setup, `VeRL-Omni` achieves **~25% higher end-to-end throughput** than the diffusers-based [`flow_grpo`](https://github.com/yifan123/flow_grpo) implementation, driven by `vLLM-Omni` rollout, FSDP training, and overlapped reward computation (asynchronous).


<div align="center">
  <img src="docs/assets/arch.png" alt="verl-omni architecture diagram" width="70%">
</div>


## Getting Started  🚀

Visit our documentation to learn more.

- [Installation](https://verl-omni.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl-omni.readthedocs.io/en/latest/start/flowgrpo_quickstart.html)

## Model and Algorithm Support 🎨

<table>
  <tr>
    <th>Model</th>
    <th>Category</th>
    <th>Modality</th>
    <th>Algorithm</th>
    <th>Status</th>
  </tr>
  <tr>
    <td rowspan="5">Qwen-Image</td>
    <td rowspan="5">Diffusion generator</td>
    <td rowspan="5">Text → Image</td>
    <td>FlowGRPO</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>MixGRPO</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>GRPO-Guard</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>DiffusionNFT</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>DPO</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Wan2.2</td>
    <td>Diffusion generator</td>
    <td>Text → Video</td>
    <td>DanceGRPO</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>LTX2.3</td>
    <td>Diffusion generator</td>
    <td>Text → Video + Audio</td>
    <td>FlowGRPO</td>
    <td>WIP</td>
  </tr>
  <tr>
    <td>BAGEL</td>
    <td>Unified understand + gen</td>
    <td>Text + Image</td>
    <td>FlowGRPO</td>
    <td>WIP</td>
  </tr>
  <tr>
    <td rowspan="2">HunyuanImage-3.0</td>
    <td rowspan="2">Unified understand + gen</td>
    <td rowspan="2">Text + Image</td>
    <td>MixGRPO</td>
    <td>Planned</td>
  </tr>
  <tr>
    <td>SRPO</td>
    <td>Planned</td>
  </tr>
  <tr>
    <td>Qwen3-Omni-Thinker</td>
    <td>Omni-modality</td>
    <td>Text / Image / Video / Audio</td>
    <td>GSPO</td>
    <td>WIP</td>
  </tr>
  <tr>
    <td>SD3.5</td>
    <td>Diffusion generator</td>
    <td>Text → Image</td>
    <td>DPO</td>
    <td>✅</td>
  </tr>
</table>


## Ascend NPU Support 💠

`VeRL-Omni` now supports Ascend NPU. For instructions on how to install and get started with FlowGRPO training on Ascend NPU, please refer to our [Ascend NPU Quickstart Guide](https://verl-omni.readthedocs.io/en/latest/start/flowgrpo_quickstart_npu.html).


## Roadmap 🗺

Future work is tracked here:

- [RFC: Multi-modal Generation RL 2026Q2 Roadmap](https://github.com/verl-project/verl/issues/5755)

## Contributing 🤝

Contributions are welcome.

See the [contribution guide](CONTRIBUTING.md).

## Acknowledgement 🌟

`verl-omni` builds on the engineering foundations developed in [`verl`](https://github.com/verl-project/verl) and is closely aligned with multimodal inference systems such as [`vLLM-Omni`](https://github.com/vllm-project/vllm-omni).

## Citation 📚

If you find the project helpful, please cite:

```bibtex
@misc{verlomni_github,
  title        = {{VeRL-Omni: Easy, Fast, and Stable RL Training for Diffusion and Omni-Modality Models}},
  author       = {Yongxiang Huang and Cheung Kawai and Jingan Zhou and Yingshu Chen and {openYuanrong Team} and Xibin Wu},
  year         = {2026},
  howpublished = {\url{https://github.com/verl-project/verl-omni}},
  urldate      = {2026-04-28}
}
```
