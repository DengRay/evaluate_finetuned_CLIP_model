#!/bin/bash

# Usage: see example script below.
# bash run_scripts/zeroshot_eval.sh 0 \
#     ${path_to_dataset} ${dataset_name} \
#     ViT-B-16 RoBERTa-wwm-ext-base-chinese \
#     ${ckpt_path}

# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

num=${1}
path=${2}
dataset=${3}
datapath=${path}/${dataset}/${num}
savedir=${path}/save_predictions
vision_model=${4} # ViT-B-16
text_model=${5}
resume=${6}
label_file=${path}/anno/${num}.txt
index=${7:-}

mkdir -p ${savedir}
#for i in 0 1 4 5 6 7 8 9 10 11 12
for i in 0
do
    python -u cn_clip/eval/zeroshot_evaluation_2.py \
    --datapath="/home/dengyiru/zero_shot/co_ture/$i" \
    --label-file="/home/dengyiru/zero_shot/color/$i.txt" \
    --save-dir="/home/dengyiru/zero_shot/save_predictions" \
    --dataset="/home/dengyiru/zero_shot/co_ture" \
    --index=${index} \
    --img-batch-size=64 \
    --resume='/home/dengyiru/Chinese-CLIP-master/data_path/pretrained_weights/clip_cn_vit-b-16.pt' \
    --vision-model='ViT-B-16' \
    --text-model='RoBERTa-wwm-ext-base-chinese' \
    --turns=$i
done

#--resume='/home/dengyiru/Chinese-CLIP-master/data_path/experiments/farfetch_finetune_vit-b-16_roberta-base_bs128_4gpu/checkpoints/epoch_latest.pt' \
#--resume='/home/dengyiru/Chinese-CLIP-master/data_path/experiments/vip_finetune_vit-b-16_roberta-base_bs128_4gpu/checkpoints/epoch_latest.pt' \
    
