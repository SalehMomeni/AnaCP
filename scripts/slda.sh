#!/bin/bash
MODEL_NAME="slda"
MODE="cil"
DATASETS=("cifar100" "cub" "imagenet-r" "t-imagenet" "cars")
SEEDS=(0 1 2)

# =====================================
# Run for backbone: dinov2
# =====================================
BACKBONE="dinov2"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running slda on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --reg 1e-4 \
            --training_method "None"
    done
done

# =====================================
# Run for backbone: mocov3
# =====================================
BACKBONE="mocov3"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running slda on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --reg 1e-4 \
            --training_method "None"
    done
done
