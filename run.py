import torch
import argparse
from trainer import train_incremental, peft

def parse_args():
    parser = argparse.ArgumentParser(description="Run training for CIL or Joint mode.")

    # Required
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['cil', 'joint'], required=True)

    # Training config
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone', type=str, default="dinov2")
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--training_method', type=str, default="none")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lora_learning_rate', type=float, default=1e-4)
    parser.add_argument('--backbone_learning_rate', type=float, default=1e-5)
    parser.add_argument('--learning_rate_decay', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--D', type=int, default=5000)
    parser.add_argument('--reg', type=float, default=1e2)

    # AnaCP
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--samples_per_class', type=int, default=100)
    parser.add_argument('--shared_cov', action='store_true')

    #KLDA
    parser.add_argument('--gamma', type=float, default=1e-4)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "cil":
        train_incremental(args)
    elif args.mode == "joint":
        peft(args)
