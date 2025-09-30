#!/bin/bash
MODEL_NAME="joint"
MODE="joint"
DATASETS=("cifar100" "cub" "imagenet-r" "t-imagenet" "cars")

# =====================================
# Run for backbone: dinov2
# =====================================
BACKBONE="dinov2"
for DATASET in "${DATASETS[@]}"; do
    echo "training joint on ${DATASET} with backbone ${BACKBONE}"
    python run.py \
        --dataset_name "${DATASET}" \
        --model_name "${MODEL_NAME}" \
        --mode "${MODE}" \
        --backbone "${BACKBONE}" \
        --seed 0 \
        --num_epochs 20 \
        --learning_rate 1e-3 \
        --lora_learning_rate 1e-4 \
        --learning_rate_decay 1e-1
done

# =====================================
# Run for backbone: mocov3
# =====================================
BACKBONE="mocov3"
for DATASET in "${DATASETS[@]}"; do
    echo "training joint on ${DATASET} with backbone ${BACKBONE}"
    python run.py \
        --dataset_name "${DATASET}" \
        --model_name "${MODEL_NAME}" \
        --mode "${MODE}" \
        --backbone "${BACKBONE}" \
        --seed 0 \
        --num_epochs 20 \
        --learning_rate 1e-2 \
        --lora_learning_rate 1e-3 \
        --learning_rate_decay 1e-2
done
