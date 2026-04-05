# Colorectal Cancer Histopathology Classification with CNNs

Automated 9-class classification of colorectal cancer tissue types using deep learning on the PathMNIST dataset.

## Clinical Context

Colorectal cancer is the third most commonly diagnosed cancer and the second leading cause of cancer-related mortality worldwide. Accurate histopathological tissue classification is essential for diagnosis, treatment planning, and prognosis. This project develops deep learning models to automatically classify colorectal tissue into 9 categories from H&E-stained histopathology patches, potentially assisting pathologists in faster and more consistent slide analysis.

**Target users:** Pathologists and clinical laboratories seeking computer-aided diagnosis tools for colorectal tissue classification.

**Tissue classes:** Adipose, Background, Debris, Lymphocytes, Mucus, Smooth Muscle, Normal Colon Mucosa, Cancer-Associated Stroma, Colorectal Adenocarcinoma Epithelium.

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- GPU recommended (NVIDIA CUDA-compatible), CPU also supported

### Installation

```bash
git clone https://github.com/wq2581/wq2581-BME6938-2026-spring-project2-group1.git
cd wq2581-BME6938-2026-spring-project2-group1
pip install -r requirements.txt
```

### Training

```bash
# Train both models (SEAttentionCNN + EfficientNet-B0)
python -m src.train --model both --config configs/config.yaml

# Train only SEAttentionCNN
python -m src.train --model se_cnn --config configs/config.yaml

# Train only EfficientNet-B0
python -m src.train --model efficientnet_b0 --config configs/config.yaml
```

Expected runtime: ~15–20 minutes per model on GPU.

## Usage Guide

1. **Explore the dataset**: Open `notebooks/EDA.ipynb` to review class distributions, sample images, pixel statistics, and augmentation previews
2. **Train models**: Run the training script as shown above — the dataset downloads automatically
3. **Evaluate and compare**: Open `notebooks/demo.ipynb` to load trained models, run inference, and compare performance with metrics and visualisations

## Data Description

- **Dataset**: PathMNIST from the MedMNIST v2 collection
- **Source**: Derived from the NCT-CRC-HE-100K dataset (Kather et al., 2019)
- **Original resolution**: 28×28 pixels, resized to 224×224 via bilinear interpolation
- **Format**: RGB (3-channel)
- **Splits**: Train (89,996) / Validation (10,004) / Test (7,180)
- **Classes**: 9 tissue types
- **License**: CC BY 4.0
- **Citation**: Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification," Scientific Data, 2023.

The dataset is automatically downloaded via the `medmnist` Python package when training or notebooks are run.

## Methods

### Model Architectures

| Model | Description | Parameters |
|---|---|---|
| SEAttentionCNN | 5-block CNN with Squeeze-and-Excitation channel attention, trained from scratch | ~1.9M |
| EfficientNet-B0 | ImageNet-pretrained EfficientNet-B0 with fine-tuned classification head | ~4.1M |

### Training Strategy

- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **LR scheduler**: Cosine Annealing (η_min=1e-6)
- **Loss**: Label Smoothing Cross-Entropy (ε=0.1)
- **Early stopping**: patience=7, checkpointing best model
- **Mixed precision**: torch.amp (AMP) for faster GPU training
- **Batch size**: 64
- **Reproducibility**: Fixed random seed (42)

## Results Summary

| Metric | SEAttentionCNN | EfficientNet-B0 |
|---|---|---|
| Accuracy | 0.9331 | 0.9513 |
| Precision (weighted) | 0.9345 | 0.9510 |
| Recall (weighted) | 0.9331 | 0.9513 |
| F1-Score (weighted) | 0.9326 | 0.9503 |
| ROC-AUC (weighted) | 0.9960 | 0.9965 |

EfficientNet-B0 outperforms SEAttentionCNN across all metrics, demonstrating the benefit of ImageNet pretraining. Both models achieve ROC-AUC > 0.996. Cancer-Associated Stroma is the most challenging class for both models (F1: 0.71–0.74), consistent with its morphological overlap with Smooth Muscle tissue.

## Project Structure

```
wq2581-BME6938-2026-spring-project2-group1/
├── README.md                    # Project overview and instructions
├── requirements.txt             # Python dependencies with versions
├── configs/
│   └── config.yaml              # Training hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Data loading, augmentation, and DataLoader creation
│   ├── models.py                # SEAttentionCNN and EfficientNet-B0 model definitions
│   ├── train.py                 # Training pipeline with AMP, label smoothing, early stopping
│   └── evaluate.py              # Metrics computation, confusion matrix, and ROC curve plotting
├── notebooks/
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   └── demo.ipynb               # Model inference and performance comparison
├── report/
│   └── Project2_TeamCRC_Report.md
├── results/                     # Training outputs (metrics, plots, model checkpoints)
│   ├── models/                  # Saved model weights (.pth files)
│   ├── *_metrics.json           # Per-model evaluation metrics
│   └── *.png                    # Visualisation figures
└── data/                        # Dataset directory (auto-downloaded, not committed)
```

## Authors and Contributions

| Name | Role |
|---|---|
| Jialu Liang | Data preprocessing, EDA, documentation, report writing |
| Bryan Quiala Llera | Report: Abstract, introduction, literature review, discussion and limitations |


## References

- Yang et al., "MedMNIST v2," Scientific Data, 2023.
- Kather et al., "Predicting survival from colorectal cancer histology slides using deep learning," PLOS Medicine, 2019.
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML, 2019.
- Hu et al., "Squeeze-and-Excitation Networks," CVPR, 2018.
- He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
