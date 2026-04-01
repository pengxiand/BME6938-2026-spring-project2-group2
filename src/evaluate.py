"""
Evaluation module for computing metrics and generating visualizations.

Provides functions for computing classification metrics (accuracy, precision,
recall, F1, ROC-AUC) and generating confusion matrices and ROC curves.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.dataset import CLASS_NAMES


def evaluate_model(model, dataloader, device):
    """Run inference and collect predictions and true labels.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for evaluation data.
        device: torch device (cpu or cuda).

    Returns:
        Tuple of (all_labels, all_preds, all_probs) as numpy arrays.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.squeeze().long()

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_labels.append(labels.numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    return all_labels, all_preds, all_probs


def compute_metrics(labels, preds, probs, num_classes=9):
    """Compute classification metrics.

    Args:
        labels: Ground truth labels.
        preds: Predicted labels.
        probs: Predicted probabilities (num_samples x num_classes).
        num_classes: Number of classes.

    Returns:
        Dictionary of computed metrics.
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # One-vs-rest AUC
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    try:
        auc_score = roc_auc_score(labels_bin, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc_score = float("nan")

    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc_score,
        "classification_report": report,
    }


def plot_confusion_matrix(labels, preds, save_path=None):
    """Plot and optionally save confusion matrix.

    Args:
        labels: Ground truth labels.
        preds: Predicted labels.
        save_path: If provided, save the figure to this path.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(labels, probs, num_classes=9, save_path=None):
    """Plot one-vs-rest ROC curves for each class.

    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities.
        num_classes: Number of classes.
        save_path: If provided, save the figure to this path.
    """
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    plt.figure(figsize=(12, 10))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
