#!/bin/bash
# eval.sh — Run RoboTwin evaluation with a lerobot-finetuned pi0_fast model.
#
# Usage:
#   bash policy/pi0_lerobot/eval.sh <task_name> <task_config> <seed> <gpu_id>
#   bash policy/pi0_lerobot/eval.sh <task_name> <task_config> <model_path> <seed> <gpu_id>
#
# Arguments:
#   task_name    RoboTwin task name  (e.g. "stack_blocks")
#   task_config  Task config file    (e.g. "stack_blocks_d0")
#   model_path   Absolute or relative path to the lerobot checkpoint directory
#                (the folder containing config.json and model.safetensors)
#   seed         Random seed for evaluation
#   gpu_id       CUDA device index to use

# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
# export SAPIEN_RENDERER="none"
# export DISPLAY=:0  # if no display
# export PYOPENGL_PLATFORM=egl
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/data/tej/HF_CACHE"
export PHYSX_NUM_THREADS=4

policy_name=pi0_lerobot_server
task_name=${1}
task_config=${2}
model_override=()
if [ "$#" -eq 5 ]; then
    model_path=${3}
    seed=${4}
    gpu_id=${5}
    model_override=(--model_path "${model_path}")
else
    seed=${3}
    gpu_id=${4}
fi

# bash eval.sh beat_block_hammer demo_randomized 0 7
# bash eval.sh beat_block_hammer demo_randomized /path/to/pretrained_model 0 7

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
if [ "$#" -eq 5 ]; then
    echo -e "\033[36mmodel_path override: ${model_path}\033[0m"
fi

cd ../.. # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
SAPIEN_HEADLESS=1 \
python script/eval_policy.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${policy_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    "${model_override[@]}"
