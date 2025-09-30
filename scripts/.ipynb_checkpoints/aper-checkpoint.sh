#!/bin/bash
MODEL_NAME="fecam"
MODE="cil"
DATASETS=("cifar100" "cub" "imagenet-r" "cars")
SEEDS=(0 1 2)

# =====================================
# Run for backbone: dinov2
# =====================================
BACKBONE="dinov2"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "running fecam on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --lambd 0.5 \
            --gamma1 1 \
            --gamma2 1 \
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
        echo "running fecam on ${DATASET} with seed ${SEED} and backbone ${BACKBONE}"
        python run.py \
            --dataset_name "${DATASET}" \
            --model_name "${MODEL_NAME}" \
            --mode "${MODE}" \
            --backbone "${BACKBONE}" \
            --seed "${SEED}" \
            --lambd 0.5 \
            --gamma1 1 \
            --gamma2 1 \
            --training_method "slca" \
            --num_epochs 10 \
            --learning_rate 1e-2 \
            --backbone_learning_rate 1e-4 
    done
done
