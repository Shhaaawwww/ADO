#!/bin/bash

# Activate conda environment
source ~/.bashrc
conda activate vtoff

# Enter project directory
cd /home/zhangchi/AdverCat/CatVTON-edited/

# Create output directory
mkdir -p ./swanlog

echo "=== Starting multi-person adversarial optimization ==="
echo "Using 5 models for alternating optimization, 100 steps per model, 500 steps total"

# Run multi-person adversarial optimization
CUDA_VISIBLE_DEVICES=2 python adv_inference_mutiperson.py \
    --pair_file "../vitonhd/pair_mutiperson.txt" \
    --attack_steps 2000 \
    --attack_lr 0.05 \
    --k 0.2 \
    --output_dir "output/mutiperson_attack" \


echo "All tasks completed!"
echo "Results saved in: ./output/mutiperson_attack/"
echo "Round results saved in: ./optclothmutiperson/round_X/"
