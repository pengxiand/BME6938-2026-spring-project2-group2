"""
Model definitions for colorectal histopathology classification.

Two architectures are provided:
1. SEAttentionCNN  — a 5-block CNN with Squeeze-and-Excitation attention, trained from scratch
2. FinetunedEfficientNetB0 — EfficientNet-B0 pretrained on ImageNet, fine-tuned for 9-class tissue classification
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention block (Hu et al., 2018).

    Recalibrates channel feature responses by explicitly modelling
    inter-channel dependencies, letting the network emphasise informative
    channels and suppress less useful ones.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck FC layers (default 16).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# Convolutional block with integrated SE attention
# ---------------------------------------------------------------------------

class ConvSEBlock(nn.Module):
    """Conv → BN → ReLU → SE attention block.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        pool: If True, apply MaxPool2d(2) after SE (default True).
        reduction: SE reduction ratio.
    """

    def __init__(self, in_ch, out_ch, pool=True, reduction=16):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch, reduction=reduction),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# SEAttentionCNN — trained from scratch
# ---------------------------------------------------------------------------

class SEAttentionCNN(nn.Module):
    """5-block CNN with Squeeze-and-Excitation channel attention, trained from scratch.

    Architecture:
        Block 1: ConvSE(3  → 32,  pool)
        Block 2: ConvSE(32 → 64,  pool)
        Block 3: ConvSE(64 → 128, pool)
        Block 4: ConvSE(128→ 256, pool)
        Block 5: ConvSE(256→ 512, AdaptiveAvgPool to 1×1)
        Classifier: FC(512→256) → ReLU → Dropout(0.4) → FC(256→num_classes)

    SE attention allows the network to dynamically reweight feature channels,
    which is particularly valuable for histopathology where subtle staining
    differences distinguish tissue subtypes.

    Args:
        num_classes: Number of output classes (default 9).
    """

    def __init__(self, num_classes=9):
        super().__init__()

        self.features = nn.Sequential(
            ConvSEBlock(3,   32,  pool=True),
            ConvSEBlock(32,  64,  pool=True),
            ConvSEBlock(64,  128, pool=True),
            ConvSEBlock(128, 256, pool=True),
            # Final block: pool to 1×1 via adaptive pooling instead of MaxPool
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# FinetunedEfficientNetB0 — transfer learning
# ---------------------------------------------------------------------------

class FinetunedEfficientNetB0(nn.Module):
    """EfficientNet-B0 pretrained on ImageNet, fine-tuned for tissue classification.

    EfficientNet-B0 uses compound scaling to balance depth, width, and
    resolution, achieving strong accuracy with fewer parameters than ResNet
    variants. The final classifier is replaced with a two-layer head suited
    to the 9-class PathMNIST task.

    Args:
        num_classes: Number of output classes (default 9).
        freeze_backbone: If True, freeze all layers except the classifier
                         head (useful for limited compute or very small datasets).
    """

    def __init__(self, num_classes=9, freeze_backbone=False):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Factory and utilities
# ---------------------------------------------------------------------------

def get_model(model_name: str, num_classes: int = 9) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_name: One of 'se_cnn' or 'efficientnet_b0'.
        num_classes: Number of output classes.

    Returns:
        Initialised nn.Module.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "se_cnn":
        return SEAttentionCNN(num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        return FinetunedEfficientNetB0(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose 'se_cnn' or 'efficientnet_b0'."
        )


def count_parameters(model: nn.Module):
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
