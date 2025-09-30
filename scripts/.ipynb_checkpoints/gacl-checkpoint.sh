#!/bin/bash
MODEL_NAME="ranpac"
MODE="cil"
DATASETS=("t-imagenet") # ("cifar100" "cub" "imagenet-r" "cars")
SEEDS=(0 1 2)

# =====================================
# Run for backbone: dinov2
# =====================================
BACKBONE="dinov2"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running racpac on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e4 \
            --training_method "slca" \
            --num_epochs 10 \
            --learning_rate 1e-3 \
            --backbone_learning_rate 1e-5 
    done
done

# =====================================
# Run for backbone: mocov3
# =====================================
BACKBONE="mocov3"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running ranpac on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --D 5000 \
            --reg 1e2 \
            --training_method "slca" \
            --num_epochs 10 \
            --learning_rate 1e-2 \
            --backbone_learning_rate 1e-4 
    done
done
