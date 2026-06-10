#!/bin/bash
# SenseNova-U1 it2i (image-to-image) FlowGRPO — minimal e2e test
#
# Tests the full training pipeline with thinkmorph interleaved data.
# Uses 2 GPUs (GPU 2,3) with minimal batch sizes for validation.
#
# Prerequisites:
#   - conda activate sensenova_u1
#   - PYTHONPATH includes vllm-omni-sensenova
#   - Test data at /tmp/thinkmorph_test_32_it2i.jsonl
#
# Usage:
#   bash examples/flowgrpo_trainer/run_sensenova_u1_it2i_test.sh

set -ex

export CUDA_VISIBLE_DEVICES=2,3
SITE_PKGS="/mnt/nas-tbt/caoziqi/micromamba/envs/sensenova_u1/lib/python3.11/site-packages"
TORCH_LIB="${SITE_PKGS}/torch/lib"
export LD_LIBRARY_PATH="${SITE_PKGS}/nvidia/cu13/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# vllm-omni source
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-/home/ximeng.czq/caoziqi/code/experiment/vllm-omni-sensenova}"
export PYTHONPATH="${VLLM_OMNI_DIR}:${PYTHONPATH:-}"

# Python from sensenova_u1 env
PYTHON="${PYTHON:-/mnt/nas-tbt/caoziqi/micromamba/envs/sensenova_u1/bin/python3}"

# Pre-import pyarrow to avoid jemalloc/CUDA runtime segfault
export PYARROW_IGNORE_TIMEZONE=1
$PYTHON -c "import pyarrow" 2>/dev/null || true

# Data
TRAIN_DATA="/tmp/thinkmorph_test_32_it2i_ready.jsonl"
MODEL_PATH="/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT"

# Simple reward: JPEG compressibility (no external model needed)
REWARD_FN="verl_omni/utils/reward_score/jpeg_compressibility.py"

NUM_GPUS=2
IMAGE_RESOLUTION=512

$PYTHON -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=2 \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    data.image_key=images \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.architecture=NEOChatModel \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.transformer_subfolder="" \
    actor_rollout_ref.model.fsdp_layer_prefixes='["language_model.model.layers."]' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=$NUM_GPUS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,3]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    reward.num_workers=$NUM_GPUS \
    reward.custom_reward_function.path=$REWARD_FN \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name=sensenova_u1_it2i_test \
    trainer.experiment_name=it2i_e2e_debug \
    trainer.log_val_generations=2 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 "$@"
