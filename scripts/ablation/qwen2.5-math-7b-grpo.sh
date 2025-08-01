# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH

N_GPU=8
N_SAMPLE=8
SAVE_STEPS=100
CLIPRANGE=0.2


# Qwen-Math template
python train_zero_math_gmpo.py \
    --critic_type grpo \
    --gpus $N_GPU \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --cliprange $CLIPRANGE \
    --oracle_type reward \
    --oracle math \
    --pretrain Qwen/Qwen2.5-Math-7B \
    --prompt_template qwen_math \
    --verifier_version math_verify \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data understand_r1_zero_main/datasets/train/math_lvl3to5_8k \
    --train_split train \
    --input_key problem \
    --output_key answer \
    --max-train 9999999 \
    --num_prompt_epoch 20 \
    --prompt_max_length 1024 \
    --num_samples $N_SAMPLE \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 3000 \
    --save_steps $SAVE_STEPS \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device $((128 / N_GPU)) \
    --pi_buffer_maxlen_per_device $((128 * N_SAMPLE / N_GPU)) \
    --eval_batch_size 200 \
    --eval_steps 100 \
    --eval_temperature 0.6 \
    --eval_n 16 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 3000 \
    --eval_data understand_r1_zero_main/datasets/evaluation_suite \
    --eval_input_key input \
    --use-wb \
    --wb_project oat-zero \
    --wb-run-name qwen2.5-Math-7b-drgrpo-qwenmathtemplate \
    --critic_type_modify grpo


