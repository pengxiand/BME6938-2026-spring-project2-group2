"""
Dataset module for loading and augmenting PathMNIST histopathology images.

PathMNIST is a 9-class colorectal tissue classification benchmark derived
from the NCT-CRC-HE-100K dataset (Kather et al., 2019).  Images are
28 × 28 px RGB patches, resized to 224 × 224 for compatibility with
pretrained backbones.

Augmentation strategy:
  - Training : geometric (flip, rotate, perspective) + colour (jitter, grayscale)
               + ImageNet normalisation
  - Val / Test: resize + centre-crop + ImageNet normalisation only
"""

import os

import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Statistics from ImageNet — used to normalise inputs for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# PathMNIST tissue class labels (9 classes)
CLASS_NAMES = [
    "Adipose",
    "Background",
    "Debris",
    "Lymphocytes",
    "Mucus",
    "Smooth Muscle",
    "Normal Colon Mucosa",
    "Cancer-Associated Stroma",
    "Colorectal Adenocarcinoma Epithelium",
]


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_transforms(augment: bool = True, target_size: int = 224) -> transforms.Compose:
    """Build a transform pipeline for training or evaluation.

    Training augmentations include:
      - Random horizontal and vertical flips
      - Random rotation (±20 °)
      - Random perspective distortion (simulates slide scanning variation)
      - Colour jitter (brightness, contrast, saturation, hue)
      - Random grayscale (p=0.05) to improve stain-robustness
      - ImageNet normalisation

    Evaluation transforms apply only resizing, centre-crop, and normalisation
    to ensure deterministic, unbiased assessment.

    Args:
        augment: If True, apply training augmentations; otherwise evaluation only.
        target_size: Spatial resolution expected by the model (default 224).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if augment:
        pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            normalize,
        ])
    else:
        pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size), antialias=True),
            normalize,
        ])

    return pipeline


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_pathmnist(
    data_dir: str = "./data",
    image_size: int = 224,
):
    """Download (if needed) and load PathMNIST with appropriate transforms.

    The dataset is loaded at its native 28 × 28 resolution to minimise disk
    usage, then up-sampled to `image_size` inside the transform pipeline.

    Args:
        data_dir: Root directory for storing the downloaded dataset.
        image_size: Target spatial resolution (default 224).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    os.makedirs(data_dir, exist_ok=True)

    train_ds = PathMNIST(
        split="train",
        transform=get_transforms(augment=True, target_size=image_size),
        download=True,
        root=data_dir,
        size=28,
    )
    val_ds = PathMNIST(
        split="val",
        transform=get_transforms(augment=False, target_size=image_size),
        download=True,
        root=data_dir,
        size=28,
    )
    test_ds = PathMNIST(
        split="test",
        transform=get_transforms(augment=False, target_size=image_size),
        download=True,
        root=data_dir,
        size=28,
    )

    return train_ds, val_ds, test_ds


def get_raw_dataset(
    data_dir: str = "./data",
    image_size: int = 224,
):
    """Load PathMNIST without normalisation for EDA and visualisation.

    Pixel values are in [0, 1] (ToTensor only) so that images display
    correctly without needing to reverse normalisation.

    Args:
        data_dir: Root directory for the dataset.
        image_size: Target spatial resolution.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    os.makedirs(data_dir, exist_ok=True)

    vis_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
    ])

    train_ds = PathMNIST(
        split="train", transform=vis_transform,
        download=True, root=data_dir, size=28,
    )
    val_ds = PathMNIST(
        split="val", transform=vis_transform,
        download=True, root=data_dir, size=28,
    )
    test_ds = PathMNIST(
        split="test", transform=vis_transform,
        download=True, root=data_dir, size=28,
    )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Wrap datasets in DataLoaders.

    pin_memory=True is enabled to accelerate host-to-GPU transfers on CUDA
    devices.  persistent_workers=True avoids re-spawning worker processes
    between epochs, reducing overhead on large datasets.

    Args:
        train_dataset: Training split dataset.
        val_dataset: Validation split dataset.
        test_dataset: Test split dataset.
        batch_size: Samples per batch (default 64).
        num_workers: Number of data-loading worker processes (default 4).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
