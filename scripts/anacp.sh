#!/bin/bash
MODEL_NAME="anacp"
MODE="cil"
DATASETS=("cifar100" "cub" "imagenet-r" "t-imagenet" "cars")
SEEDS=(0 1 2)

# =====================================
# Run for backbone: dinov2
# =====================================
BACKBONE="dinov2"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running acp on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --num_heads 3 \
            --training_method "aper" \
            --num_epochs 10 \
            --learning_rate 1e-3 \
            --lora_learning_rate 1e-4 \
            --shared_cov
    done
done

# =====================================
# Run for backbone: mocov3
# =====================================
BACKBONE="mocov3"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running acp on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --num_heads 3 \
            --training_method "aper" \
            --num_epochs 10 \
            --learning_rate 1e-2 \
            --lora_learning_rate 1e-3 \
            --shared_cov
    done
done
