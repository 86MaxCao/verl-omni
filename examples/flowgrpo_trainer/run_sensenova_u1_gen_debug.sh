#!/bin/bash
# SenseNova-U1 Image Generation RL (FlowGRPO) — debug launch script
#
# This script demonstrates how to launch SenseNova-U1 FlowGRPO training.
# It is a debug configuration; for production runs adjust batch sizes,
# learning rate, GPU counts, and reward model settings accordingly.
#
# Prerequisites:
#   - SenseNova-U1-8B-MoT model weights with trust_remote_code support
#   - Training data in parquet format with text prompts
#   - A reward model or reward function (e.g., HPSv3, aesthetic score)
#   - transformers >= 5.8.0 (required by the model's custom code)
#
# Key differences from Qwen-Image:
#   - architecture must be set explicitly (no model_index.json for transformers models)
#   - fsdp_layer_prefixes target the LLM decoder layers (language_model.model.layers.)
#   - transformer_subfolder is empty (SenseNova-U1 is a single-model checkpoint)
#   - trust_remote_code=True is required (custom NEOChatModel)
#   - No VAE — the model operates in pixel/patch space
#
# Key differences from Bagel:
#   - Architecture is NEOChatModel (not BagelForConditionalGeneration)
#   - Dense Qwen3 backbone (not MoE), 17.55B parameters
#   - Uses fm_modules (timestep_embedder, fm_head, noise_scale_embedder)
#   - Custom v_pred formulation with noise_scale_embedding support
#
# Usage:
#   NUM_GPUS=4 bash examples/flowgrpo_trainer/run_sensenova_u1_gen_debug.sh

set -x

WORKSPACE=${WORKSPACE:-$HOME}

# vllm-omni source (needed by the rollout adapter)
VLLM_OMNI_DIR=${VLLM_OMNI_DIR:-$HOME/caoziqi/code/experiment/vllm-omni-sensenova}
export PYTHONPATH="${VLLM_OMNI_DIR}:${PYTHONPATH:-}"

# Pre-import pyarrow to avoid jemalloc segfault when vllm.config loads
python3 -c "import pyarrow" 2>/dev/null || true
export PYTHON_STARTUP_IMPORT="pyarrow"

# Data paths (adjust to your training data)
train_data_path=${TRAIN_DATA:-/mnt/nas-tbt/caoziqi/data/sensenova_u1_gen/train.parquet}
test_data_path=${TEST_DATA:-/mnt/nas-tbt/caoziqi/data/sensenova_u1_gen/test.parquet}

# Model paths
model_name=${MODEL_PATH:-/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT}
reward_function_path=verl_omni/utils/reward_score/hpsv3_reward.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS:-4}
NUM_NODES=${NUM_NODES:-1}
ACTOR_SP=1
ROLLOUT_TP=1
REWARD_TP=1
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}

# NOTE: rollout engine is set to vllm_omni but the SenseNova-U1 rollout adapter
# is not yet implemented. For initial testing of the training adapter,
# use pre-collected rollout data or implement the rollout adapter first.
ENGINE=vllm_omni
REWARD_ENGINE=vllm

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$train_data_path \
    data.val_files=$test_data_path \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.architecture=NEOChatModel \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.transformer_subfolder="" \
    actor_rollout_ref.model.fsdp_layer_prefixes='["language_model.model.layers."]' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=24 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=24 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name=sensenova_u1_flow_grpo \
    trainer.experiment_name=sensenova_u1_gen_debug \
    trainer.log_val_generations=4 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / NUM_NODES)) \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=100 "$@"
