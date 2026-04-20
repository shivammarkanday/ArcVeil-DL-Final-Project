# ArcVeil 🎭
### ArcFace-Based Masked Face Recognition Using a Manual ResNet50 Backbone

> A deep learning system that identifies individuals even when their face is partially occluded by a mask — built with a manually implemented ResNet50 backbone, ArcFace angular-margin loss, and an auxiliary mask detection head, all trained jointly in a single end-to-end pipeline.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Team](#team)
- [References](#references)

---

## Overview

Standard face recognition systems fail when the lower half of the face is covered by a mask. **ArcVeil** addresses this by building a mask-aware recognition pipeline that:

- Accepts both **masked and unmasked** probe images at inference time
- Enrolls known individuals via a small **gallery of reference images**
- Returns a **predicted identity**, a **calibrated cosine similarity score**, and a **mask-status flag**
- Maintains strong accuracy on standard (same-condition) recognition while degrading gracefully on harder cross-condition (unmasked gallery, masked query) scenarios

The core insight is that the **periocular region** — eyes, eyebrows, and nose bridge — retains sufficient discriminative information for reliable identity matching even when the lower face is completely hidden.

---

## Key Results

### Same-Condition Validation (RMFRD, threshold = 0.45)

| Metric | Value |
|---|---|
| Identity Accuracy | **90.26%** |
| Known-only Accuracy | 93.67% |
| Unknown Rate | 3.63% |
| Mask F1 Score | **95.37%** |
| Mean Cosine Similarity | 0.7414 |

### Cross-Condition: Unmasked Gallery / Masked Query (threshold = 0.38)

| Metric | Value |
|---|---|
| Identity Accuracy | **52.79%** |
| Known-only Accuracy | 67.58% |
| Unknown Rate | 21.89% |
| Mask F1 Score | **99.89%** |

### Custom Real-World Dataset (6 identities, 52 images, threshold = 0.38)

| Metric | Value |
|---|---|
| Identity Accuracy | **86.54%** |
| Unknown Rate | 0.00% |
| Mean Cosine Similarity | 0.6568 |

---

## Architecture

ArcVeil is a 6-stage end-to-end pipeline:

```
Input Image
    │
    ▼
[Stage 2] InsightFace Detector (buffalo_sc / SCRFD)
          Face Crop → Resize 224×224 → ImageNet Normalisation
    │
    ▼
[Stage 3] Manual ResNet50 Backbone
          Stem: Conv 7×7, stride 2 → BN → ReLU → MaxPool
          Stage 1: 3× Bottleneck [64→256ch]
          Stage 2: 4× Bottleneck [128→512ch], stride 2
          Stage 3: 6× Bottleneck [256→1024ch], stride 2
          Stage 4: 3× Bottleneck [512→2048ch], stride 2
          Global Average Pooling → 2048-d feature vector f
    │
    ├─────────────────────────────────┐
    ▼                                 ▼
[Stage 4A] Embedding Head         [Stage 4B] Mask Head
BN → Dropout → Linear(2048→1024)  Linear(2048→512) → BN
→ BN → PReLU → Dropout            → ReLU → Dropout
→ Linear(1024→512) → BN           → Linear(512→1) → Sigmoid
→ L2-Normalise                    → p_mask (mask status flag)
→ 512-d embedding ê
→ ArcFace Head
  cos(θ_yi + m), m=0.3, s=32
    │
    ▼
[Stage 5] Combined Loss (training only)
          L_total = 1.0 × L_arc + 0.1 × L_mask

[Stage 6] Gallery Matching
          sim = (q · g_i) / (‖q‖ · ‖g_i‖)
          sim ≥ τ → Predicted Identity
          sim <  τ → UNKNOWN
```

### ArcFace Loss

The ArcFace loss adds an additive angular margin *m* to the ground-truth class angle:

```
cos(θ_yi + m) = cos(θ_yi)·cos(m) − sin(θ_yi)·sin(m)

L_arc = −log[ e^(s·cos(θ_yi + m)) / (e^(s·cos(θ_yi + m)) + Σ_{j≠yi} e^(s·cos(θ_j))) ]
```

This forces the model to place embeddings within a narrower cone around the class centre, producing more compact and discriminative identity representations on the unit hypersphere.

---

## Dataset

The project uses the **AFDB (Asian Face Database)** hosted on Kaggle:

```
/kaggle/input/.../AFDB_face_dataset          ← unmasked images
/kaggle/input/.../AFDB_masked_face_dataset   ← masked images
```

| Statistic | Value |
|---|---|
| Total identities | 460 |
| Total images | 91,867 |
| Training split (80%) | 73,492 |
| Validation split (20%) | 18,374 |

**Kaggle dataset:** [muhammeddalkran/masked-facerecognition](https://www.kaggle.com/datasets/muhammeddalkran/masked-facerecognition)

### Augmentation Policy

A 12-stage Albumentations pipeline is applied during training. Critically, synthetic occlusion is **restricted to the eye-level band (20–45% of image height)** to simulate glasses — preserving lower-face identity cues for same-condition recognition.

| Transform | Parameters |
|---|---|
| HorizontalFlip | p = 0.5 |
| ShiftScaleRotate | shift 5%, scale 5%, rotate 10° |
| RandomBrightnessContrast | ±0.2 |
| HueSaturationValue | ±20, ±30, ±20 |
| GaussianBlur / MotionBlur | kernel 3–5 |
| GaussNoise | var 10–50 |
| ImageCompression | quality 70–100 |
| Eye-level dropout | rows 20%–45% of height |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/arcveil.git
cd arcveil

# Install dependencies
pip install torch torchvision
pip install insightface onnxruntime
pip install albumentations opencv-python
pip install timm shap
```

> **Note:** Training requires a CUDA-capable GPU. The project was developed and tested on a Tesla T4 (16 GB VRAM) via the Kaggle platform.

### Software Environment

| Library | Version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| InsightFace | 0.7.3 |
| timm | 1.0.25 |
| Albumentations | latest |
| NumPy | 2.0.2 |
| OpenCV | 4.13.0 |

---

## Usage

### Building the Gallery

```python
from arcveil import ArcVeilModel, build_gallery

model = ArcVeilModel()
model.load_checkpoint("best_model.pth")

# gallery_dir should contain one subfolder per identity
gallery = build_gallery(model, gallery_dir="path/to/gallery")
```

### Running Inference

```python
from arcveil import predict

result = predict(
    model=model,
    gallery=gallery,
    image_path="path/to/probe_image.jpg",
    threshold=0.38       # use 0.45 for same-condition, 0.38 for cross-condition
)

print(result["identity"])     # predicted name or "UNKNOWN"
print(result["similarity"])   # cosine similarity score
print(result["mask_worn"])    # True / False
```

### Real-Time Webcam Inference

```python
from arcveil import webcam_demo

webcam_demo(model=model, gallery=gallery, threshold=0.38)
```

---

## Training

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Backbone | Manual ResNet50 (ImageNet init) |
| Embedding dimension | 512 |
| ArcFace margin *m* | 0.3 rad (~17.2°) |
| ArcFace scale *s* | 32 |
| Batch size | 16 |
| Epochs | 50 (best at epoch 27) |
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻³ |
| Weight decay | 5 × 10⁻⁴ |
| LR Schedule | LinearWarmup (5 ep) → ReduceLROnPlateau |
| Mixed Precision | FP16 |
| Gradient Clipping | 1.0 |
| Label Smoothing | 0.1 |
| λ_arc / λ_mask | 1.0 / 0.1 |

```bash
python train.py \
  --data_dir /path/to/AFDB \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-3 \
  --arc_margin 0.3 \
  --arc_scale 32
```

> ⚠️ **Important:** Avoid aggressive ArcFace parameters (e.g. m=0.5, s=64) on smaller datasets — this causes embedding collapse where all identity vectors converge to nearly identical representations.

---

## Evaluation

Two evaluation settings are used:

**1. Same-condition (RMFRD validation split)**
```bash
python evaluate.py --mode same_condition --threshold 0.45
```

**2. Cross-condition (unmasked gallery / masked query)**
```bash
python evaluate.py --mode cross_condition --threshold 0.38
```

**3. Threshold sweep (to find optimal operating point)**
```bash
python threshold_sweep.py --start 0.38 --end 0.45 --step 0.01
```

### Reported Metrics

- **Identity Accuracy** — fraction of all queries correctly matched (including UNKNOWN rejections)
- **Known-only Accuracy** — fraction of accepted queries correctly matched
- **Unknown Rate** — fraction of queries rejected as UNKNOWN
- **Mask F1** — F1 score of the binary mask classifier
- **TAR@FAR=1e-4** — True Acceptance Rate at False Acceptance Rate of 10⁻⁴ (used for checkpoint selection)

---

## Explainability

SHAP (SHapley Additive exPlanations) pixel-importance heatmaps were computed using a `DeepExplainer` on the embedding head. Results consistently show:

- **High importance:** periocular region (eyes, eyebrows, nose bridge)
- **Lower importance:** forehead, chin, masked lower face

This confirms the model has learned to rely on features that remain available under mask occlusion — exactly the expected behaviour for a mask-robust system.

```bash
python shap_analysis.py --num_samples 50 --output_dir shap_outputs/
```

---

## Project Structure

```
arcveil/
├── backbone/
│   ├── resnet50.py          # Manual ResNet50 implementation
│   └── bottleneck.py        # Bottleneck block primitive
├── heads/
│   ├── embedding_head.py    # 2048 → 512-d projection head
│   ├── arcface_head.py      # ArcFace angular-margin head
│   └── mask_head.py         # Auxiliary binary mask classifier
├── data/
│   ├── dataset.py           # AFDB dataset loader
│   ├── augmentations.py     # 12-stage Albumentations pipeline
│   └── preprocessing.py     # InsightFace crop + normalisation
├── gallery/
│   └── gallery_builder.py   # Gallery construction & cosine matching
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── threshold_sweep.py       # Threshold calibration utility
├── shap_analysis.py         # SHAP explainability
├── webcam_demo.py           # Real-time inference demo
├── best_model.pth           # Trained checkpoint (epoch 27)
└── README.md
```

---

## Limitations & Future Work

### Current Limitations

- RMFRD validation overlaps with training identities — a fully held-out test set on **unseen identities** would give a more rigorous evaluation
- A ~37% cross-condition performance gap remains, indicating the masked/unmasked embedding distributions have not been fully aligned
- Pipeline depends on CUDA-specific components and has not been migrated to TPU execution

### Planned Improvements

| Priority | Item |
|---|---|
| High | Proper train/val/test split with unseen identities |
| High | Cross-condition training using contrastive or triplet loss to minimise the masked/unmasked embedding gap |
| Medium | Systematic precision–recall curve reporting across the full threshold range |
| Medium | Larger backbones — IResNet100 or ViT-based architectures |
| Low | ONNX quantisation and pruning for mobile / edge deployment |

---

## Team

| Name | Roll Number |
|---|---|
| Manuj Dave | 1AUA24BCS909 |
| Shivam Markanday | 1AUA23BCS171 |
| Varan Antal | 1AUA23BCS199 |

**Guide / Supervisor:** Dr. Spoorthy Venkatesh

**Institution:** Adani University, Department of CSE (AI-ML), Gandhinagar, India

*Submitted in partial fulfilment of the B.Tech degree in Computer Science & Engineering (AI-ML), Academic Year 2025–2026.*

---

## References

1. Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," CVPR 2019
2. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
3. Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition," CVPR 2018
4. Liu et al., "SphereFace: Deep Hypersphere Embedding for Face Recognition," CVPR 2017
5. Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering," CVPR 2015
6. Anwar & Raychowdhury, "Masked Face Recognition for Secure Authentication," arXiv 2020
7. [InsightFace](https://github.com/deepinsight/insightface) — Open source 2D & 3D deep face analysis toolbox
8. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017

---

<p align="center">
  Made with ❤️ at Adani University · April 2026
</p>