"""
Training pipeline for colorectal histopathology classification.

Supports mixed-precision training (torch.amp) for faster throughput on
modern GPUs (e.g., NVIDIA B200), label-smoothing cross-entropy loss,
cosine annealing LR scheduling, and early stopping with checkpointing.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

from src.dataset import load_pathmnist, create_dataloaders
from src.models import get_model, count_parameters
from src.evaluate import (
    evaluate_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Label-smoothing loss
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing (Szegedy et al., 2016).

    Label smoothing prevents the model from becoming over-confident by
    distributing a small probability mass (epsilon) uniformly across all
    non-target classes.  This acts as a regulariser and often improves
    calibration and generalisation.

    Args:
        epsilon: Smoothing factor in [0, 1) (default 0.1).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, epsilon: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Hard-target loss
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Uniform smoothing loss
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.epsilon) * nll + self.epsilon * smooth
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ---------------------------------------------------------------------------
# Single-epoch training / validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch using mixed-precision (AMP).

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: torch.device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    """Evaluate the model on a validation / test split.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: torch.device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.squeeze().long().to(device, non_blocking=True)

            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_model(model_name: str, config: dict):
    """End-to-end training pipeline for a single model.

    Steps:
        1. Set seed and create data loaders.
        2. Instantiate model, loss (label smoothing), Adam optimiser,
           cosine LR schedule, and AMP GradScaler.
        3. Train with early stopping; checkpoint the best validation-loss model.
        4. Evaluate on test set; save metrics, confusion matrix, ROC curves,
           and training history plots.

    Args:
        model_name: 'se_cnn' or 'efficientnet_b0'.
        config: Parsed YAML configuration dict.

    Returns:
        Tuple of (trained_model, history_dict).
    """
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Data ---------------------------------------------------------------
    data_dir    = config["dataset"]["data_dir"]
    image_size  = config["dataset"]["image_size"]
    batch_size  = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    num_classes = config["dataset"]["num_classes"]

    train_ds, val_ds, test_ds = load_pathmnist(data_dir, image_size)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size, num_workers
    )
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # ---- Model --------------------------------------------------------------
    model = get_model(model_name, num_classes=num_classes).to(device)
    total_p, trainable_p = count_parameters(model)
    print(f"Model: {model_name}  |  Total params: {total_p:,}  |  Trainable: {trainable_p:,}")

    # ---- Loss / optimiser / scheduler / scaler ------------------------------
    criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    num_epochs = config["training"]["num_epochs"]
    scheduler  = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler     = GradScaler()          # AMP gradient scaler

    # ---- Directories --------------------------------------------------------
    results_dir = config["output"]["results_dir"]
    model_dir   = config["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---- Training loop ------------------------------------------------------
    patience       = config["training"]["patience"]
    best_val_loss  = float("inf")
    epochs_no_imp  = 0
    best_ckpt      = os.path.join(model_dir, f"best_{model_name}.pth")
    history        = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Training  {model_name}  (up to {num_epochs} epochs, patience={patience})")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1:>3}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"LR: {lr_now:.2e}  |  {time.time()-t0:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_imp = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}.")
                break

    # ---- Test evaluation ----------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    labels, preds, probs = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs, num_classes)

    print(f"\n{'='*60}")
    print(f"Test results — {model_name}")
    print(f"{'='*60}")
    for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"  {k:<12}: {metrics[k]:.4f}")
    print(f"\n{metrics['classification_report']}")

    # ---- Save artefacts -----------------------------------------------------
    plot_confusion_matrix(
        labels, preds,
        save_path=os.path.join(results_dir, f"{model_name}_confusion_matrix.png"),
    )
    plot_roc_curves(
        labels, probs, num_classes,
        save_path=os.path.join(results_dir, f"{model_name}_roc_curves.png"),
    )
    plot_training_history(
        history,
        save_path=os.path.join(results_dir, f"{model_name}_training_history.png"),
    )

    metrics_out = {k: v for k, v in metrics.items()}
    with open(os.path.join(results_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    return model, history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train colorectal histopathology classification models"
    )
    parser.add_argument(
        "--model", type=str, default="se_cnn",
        choices=["se_cnn", "efficientnet_b0", "both"],
        help="Model to train",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output"]["results_dir"], exist_ok=True)

    models_to_train = ["se_cnn", "efficientnet_b0"] if args.model == "both" else [args.model]
    for name in models_to_train:
        train_model(name, config)


if __name__ == "__main__":
    main()
