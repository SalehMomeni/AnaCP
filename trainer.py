import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from models import get_model
from typing import List
from sklearn.metrics import accuracy_score
from data_loader import UnifiedDataLoader
from backbone import Backbone, BackboneLORA, LoRALinear, get_processor


def train_incremental(args):
    device = torch.device(args.device)
    backbone = Backbone(args)
    processor = get_processor(args.backbone)

    train_loader = UnifiedDataLoader(
        dataset_name=args.dataset_name,
        split="train",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=True,
        n_tasks=args.n_tasks,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    ).get_dataloaders()

    test_loader = UnifiedDataLoader(
        dataset_name=args.dataset_name,
        split="test",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=True,
        n_tasks=args.n_tasks,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    ).get_dataloaders()

    model = get_model(args)

    n_tasks = len(train_loader)
    A = np.zeros((n_tasks, n_tasks))
    test_features = [None] * n_tasks  # Cache test features
    concat_features = args.model_name.lower() == 'aper'

    for t in range(n_tasks):
        if t == 0 and args.training_method.lower() != 'none':
            print("Training the model on first task...")
            backbone.finetune(train_loader[t])
        print(f"\nLearning task {t + 1}/{n_tasks}...")
        print("Extracting Features...")
        X_train, Y_train = backbone.get_features(train_loader[t], concat_features=concat_features)

        print("Updating model...")
        model.update(X_train, Y_train)

        # Evaluate on all seen tasks so far
        print("Evaluating...")
        for i in range(t + 1):
            if test_features[i] is None:
                test_features[i] = backbone.get_features(test_loader[i], concat_features=concat_features)
            X_test, Y_test = test_features[i]
            Y_pred = model.predict(X_test)
            A[t, i] = accuracy_score(Y_test, Y_pred)

        row_avg = np.mean(A[t][:t+1])
        print(f"Accuracy after learning task {t + 1}: {row_avg:.4f}")

    # Compute final metrics
    row_averages = [np.mean(row[row > 0]) for row in A]
    avg_acc = np.mean(row_averages)
    last_acc = row_averages[-1]

    # Logging
    log_dir = os.path.join(args.log_dir, args.dataset_name, args.backbone, str(args.seed))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, args.model_name)

    with open(log_file, "w") as f:
        f.write("Accuracy Matrix (A[t][i]):\n")
        f.write(str(A) + "\n\n")
        f.write(f"Average Accuracy: {avg_acc:.4f}\n")
        f.write(f"Last Accuracy: {last_acc:.4f}\n")


def peft(args):
    device = torch.device(args.device)
    processor = get_processor(args.backbone)

    train_loader = UnifiedDataLoader(
        dataset_name=args.dataset_name,
        split="train",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=False,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    ).get_dataloaders()

    test_loader = UnifiedDataLoader(
        dataset_name=args.dataset_name,
        split="test",
        transform=processor,
        data_dir=args.data_dir,
        is_class_incremental=False,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    ).get_dataloaders()

    num_classes = len(set(train_loader.dataset.targets))
    backbone = BackboneLORA(args.backbone, num_classes, args.rank, device)

    # Parameter groups for LoRA and classifier
    lora_params = []
    classifier_params = list(backbone.classifier.parameters())

    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend(list(module.lora_down.parameters()))
            lora_params.extend(list(module.lora_up.parameters()))

    def CosineAnnealingLR(optimizer, base_lrs, decay_factor, num_epochs):
        def cosine_fn(base_lr):
            min_lr = decay_factor * base_lr
            return lambda epoch: min_lr / base_lr + 0.5 * (1 - min_lr / base_lr) * (1 + math.cos(math.pi * epoch / num_epochs))
    
        lr_lambdas = [cosine_fn(lr) for lr in base_lrs]
        return LambdaLR(optimizer, lr_lambda=lr_lambdas)

    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': args.lora_learning_rate},
        {'params': classifier_params, 'lr': args.learning_rate}
    ])

    scheduler = CosineAnnealingLR(
        optimizer,
        base_lrs=[args.lora_learning_rate, args.learning_rate],
        decay_factor=args.learning_rate_decay,
        num_epochs=args.num_epochs
    )

    criterion = nn.CrossEntropyLoss()
    log_path = os.path.join(args.log_dir, args.dataset_name, args.backbone)
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "peft.txt")

    with open(log_file, 'w') as f:
        for epoch in range(args.num_epochs):
            backbone.train()
            total_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = backbone(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)

            scheduler.step()
            avg_loss = total_loss / len(train_loader.dataset)
            f.write(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}\n")
            f.flush()

        # Evaluation
        backbone.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.to(device)
                outputs = backbone(inputs)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        acc = 100. * correct / total
        f.write(f"Final Test Accuracy = {acc:.2f}%\n")
        f.flush()
