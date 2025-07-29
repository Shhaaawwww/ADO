#!/bin/bash

# 激活conda环境
source ~/.bashrc
conda activate vtoff

# 进入项目目录
cd /home/zhangchi/AdverCat/CatVTON-edited/

# 创建输出目录
mkdir -p ./swanlog

echo "=== 开始多模特对抗优化 ==="
echo "使用5个模特轮流优化，每个模特100步，总共500步"

# 运行多模特对抗优化
CUDA_VISIBLE_DEVICES=2 python adv_inference_mutiperson.py \
    --pair_file "../vitonhd/pair_mutiperson.txt" \
    --attack_steps 2000 \
    --attack_lr 0.05 \
    --k 0.2 \
    --output_dir "output/mutiperson_attack" \


echo "所有任务已完成!"
echo "结果保存在: ./output/mutiperson_attack/"
echo "各轮结果保存在: ./optclothmutiperson/round_X/"
