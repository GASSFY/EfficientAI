#!/usr/bin/env bash
# Manual MAS-Quant train / eval helper for Qwen2.5-VL.
#
# Prerequisites:
#   - act_scales/<model>-text-vision-128.pt   (from generate_act_scale_shift.py) for training
#   - outputs/...-2epochs-.../mas_parameters.pth for eval-only (--epochs 0 --resume)
# Outputs go under --output_dir (default ./outputs); see ARTIFACTS.md.
# Prefer --gen_max_new_tokens 16 for MCQ tasks and 128 for ocrbench (via main.py flag).
# For multi-config overnight runs, use run_overnight.sh instead.

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"   # 改成实际路径
# TASKS="mmmu_val,realworldqa,ocrbench,ai2d"
TASKS="ai2d"
export inference_mode="split_scales"
export API_TYPE=dummy

# 测FP16
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --model "${MODEL_PATH}" \
#   --mode train \
#   --epochs 0 \
#   --wbits 16 --abits 16 \
#   --dataset-type text-vision \
#   --nsamples 128 \
#   --output_dir ./outputs_vl/fp16 \
#   --tasks_multimodal "${TASKS}" \
#   --limit_multimodal 1.0

# 先生成scale
# python generate_act_scale_shift.py \
#     --model "${MODEL_PATH}" \
#     --dataset-type text-vision \
#     --nsamples 128

# 开始训练，产物放在output之下
# python main.py \
#     --model "${MODEL_PATH}" \
#     --mode train \
#     --epochs 2 \
#     --wbits 4 --abits 16 \
#     --let \
#     --loss_multi_modal_mae_alpha \
#     --dataset-type text-vision \
#     --nsamples 128 \
#     --output_dir ./outputs \
#     --symmetric \
#     --group_size 0

# 上面放的是量化产物的位置
RESUME=/root/autodl-tmp/quantization/EfficientAI/masquant/outputs/Qwen2.5-VL-7B-Instruct-2epochs-w4a16--0801-173551.325879-split_scales/mas_parameters.pth

# 这是评测的
python main.py \
    --model "${MODEL_PATH}" \
    --mode train \
    --epochs 0 \
    --wbits 4 --abits 16 \
    --let \
    --resume "${RESUME}" \
    --dataset-type text-vision \
    --nsamples 128 \
    --output_dir ./outputs \
    --symmetric \
    --group_size 0 \
    --tasks_multimodal "${TASKS}"