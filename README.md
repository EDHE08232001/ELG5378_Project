# MCUCoder — Progressive Learned Image Compression

A convolutional autoencoder that supports multiple bitrate–quality operating
points from a single trained model.  During training, random *tail-dropout* of
latent channels forces early channels to carry the most important image
information.  At inference time, any prefix of 1–12 channels can be decoded
independently, trading bitrate for reconstruction quality without re-encoding.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Download](#4-dataset-download)
   - 4.1 [Kodak Validation Set](#41-kodak-validation-set)
   - 4.2 [ImageNet Training Subset](#42-imagenet-training-subset)
   - 4.3 [Pre-trained Model Weights](#43-pre-trained-model-weights)
5. [How to Run](#5-how-to-run)
   - 5.1 [Training](#51-training)
   - 5.2 [Evaluation](#52-evaluation)
   - 5.3 [ImageNet Preparation](#53-imagenet-preparation-optional)
6. [Configuration Reference](#6-configuration-reference)
7. [Output Files](#7-output-files)
8. [Architecture Summary](#8-architecture-summary)

---

## 1. Project Overview

| Component | Details |
|-----------|---------|
| Task | Single-model progressive image compression |
| Framework | PyTorch (no Lightning required) |
| Encoder | 3-layer lightweight CNN → 12-channel latent at 28×28 |
| Decoder | Deep residual network (compressai blocks, N=196) |
| Training strategy | Stochastic tail-dropout of latent channels |
| Loss | λ·(1−MS-SSIM) + (1−λ)·MSE |
| Evaluation metrics | PSNR (dB), MS-SSIM (dB), bits-per-pixel (bpp) |
| Baseline | JPEG at 6 quality levels |

---

## 2. Repository Layout

```
ELG5378_Project/
│
├── main.py                    ← entry point (always run from here)
├── requirements.txt
├── README.md
│
├── src/                   ← source package
│   ├── __init__.py
│   ├── config.py              ← all paths and hyperparameters
│   ├── data.py                ← RecursiveImageDataset + DataLoaders
│   ├── model.py               ← Encoder, Decoder, MCUCoder
│   ├── losses.py              ← ProgressiveLoss, PSNR, MS-SSIM helpers
│   ├── train.py               ← training loop + LR schedule + checkpointing
│   ├── evaluate.py            ← RD evaluation + quantization + plots
│   ├── prepare_data.py        ← ImageNet high-res selection & preprocessing
│   └── utils.py               ← set_seed, get_device, format_metrics
│
├── datasets/                  ← place raw data here (created by you)
│   ├── kodak/                 ← 24 Kodak PNG images
│   ├── imagenet/
│   │   └── train/             ← raw ImageNet class subdirectories
│   └── imagenet_prepared/     ← flat PNG output from prepare step (auto)
│
└── outputs/                   ← auto-created at runtime
    ├── checkpoints/
    │   └── mcucoder.pth       ← best model checkpoint
    └── results/
        ├── eval_summary.json  ← numeric RD results
        ├── rd_curves.pdf      ← rate-distortion plot
        └── *.png              ← sample reconstructions
```

---

## 3. Environment Setup

### 3.1 Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU note:** For CUDA support make sure you install the matching `torch`
> wheel from <https://pytorch.org/get-started/locally/>.  The code
> auto-detects CUDA → MPS → CPU at runtime.

---

## 4. Dataset Download

Create the expected directories first:

```bash
mkdir -p datasets/imagenet/train
mkdir -p datasets/kodak
```

### 4.1 Kodak Validation Set

24 lossless PNG images (the standard compression benchmark):

```bash
for i in $(seq -w 1 24); do
  curl -L "https://r0k.us/graphics/kodak/kodak/kodim${i}.png" \
       -o "datasets/kodak/kodim${i}.png"
done
```

This downloads ~35 MB and takes under a minute.

### 4.2 ImageNet Training Subset

The project uses a 300 k-image subset of ILSVRC-2012.  You need a valid
ImageNet account to download the raw data.

**Option A — you already have ImageNet locally:**

Place (or symlink) the `train/` directory at:

```
datasets/imagenet/train/
```

The loader scans recursively, so class subdirectories are handled
automatically.

**Option B — use a public subset (e.g. ImageNet-1k via Hugging Face):**

```bash
pip install datasets
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("imagenet-1k", split="train", streaming=True,
                  trust_remote_code=True)
import os, io
from PIL import Image
out = "datasets/imagenet/train"
os.makedirs(out, exist_ok=True)
for i, example in enumerate(ds):
    if i >= 300_000:
        break
    example["image"].convert("RGB").save(f"{out}/{i:07d}.jpg")
EOF
```

**Option C — run the built-in preparation step (after placing raw data):**

```bash
python main.py
# → choose 3 (Prepare)
```

This selects the 300 k highest-resolution images, optionally halves very
large ones, adds a tiny noise perturbation, and saves flat PNGs to
`datasets/imagenet_prepared/`.  The training step uses this prepared
directory automatically when it exists.

### 4.3 Pre-trained Model Weights

If you want to skip training and evaluate the reference weights provided by
the MCUCoder authors:

```bash
pip install gdown
mkdir -p outputs/checkpoints
gdown "1aWLukhsRV5Fi_DFJUbL5nBdwNhGDpNe0" -O outputs/checkpoints/mcucoder.pth
```

---

## 5. How to Run

**All commands must be run from the repository root** (`ELG5378_Project/`):

```bash
python main.py
```

You will see:

```
============================================================
  MCUCoder — Progressive Learned Image Compression
============================================================

Select an action:
  1) Train    — train the model on ImageNet / validate on Kodak
  2) Evaluate — run rate-distortion evaluation and plot RD curves
  3) Prepare  — pre-process raw ImageNet images for training

Enter 1, 2, or 3:
```

### 5.1 Training

```
Enter 1, 2, or 3: 1
```

What happens:
- Loads ImageNet training images recursively from `datasets/imagenet/train/`
  (or `datasets/imagenet_prepared/` if it exists).
- Loads Kodak validation images from `datasets/kodak/`.
- Trains MCUCoder with stochastic tail-dropout.
- Validates at 3 bitrate levels (2 / 6 / 12 channels) after every epoch.
- Saves the **best checkpoint** (by validation loss) to
  `outputs/checkpoints/mcucoder.pth`.

**Key hyperparameters** (edit `project/config.py`):

| Key | Default | Description |
|-----|---------|-------------|
| `num_epochs` | 10 | Epochs (use 50–150 for production) |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 1e-4 | Adam learning rate |
| `lr_decay_epoch` | 9 | Epoch for ×0.1 LR decay |
| `loss` | `"msssim"` | `"msssim"` or `"mse"` |
| `lambda_msssim` | 0.9 | Weight of MS-SSIM term |

### 5.2 Evaluation

```
Enter 1, 2, or 3: 2
```

What happens:
- Loads the checkpoint from `outputs/checkpoints/mcucoder.pth`.
- Calibrates per-channel quantization statistics from the Kodak set.
- For each active channel count k ∈ {1 … 12}:
  - Zeros out channels k+1 … 12.
  - Applies uniform quantization (step=4, 6 bpp) to k active channels.
  - Decodes and measures PSNR, MS-SSIM, and bpp.
- Evaluates JPEG at 6 quality levels for comparison.
- Saves outputs in `outputs/results/`.

### 5.3 ImageNet Preparation (optional)

```
Enter 1, 2, or 3: 3
```

Pre-processes raw ImageNet: selects the 300 k highest-resolution images,
halves very large ones (shorter side > 512 px), adds small noise, and saves
to `datasets/imagenet_prepared/`.

---

## 6. Configuration Reference

Edit `src/config.py`.  All paths are absolute, resolved relative to the
repository root at import time.

```python
CONFIG = {
    # Data
    "train_data_dir": ...,   # auto-selects imagenet_prepared if it exists
    "val_data_dir":   ...,   # datasets/kodak/
    "image_size":     224,   # crop size (px)

    # DataLoader
    "batch_size":   16,
    "num_workers":  4,

    # Model
    "latent_channels":  12,    # latent channels (1–12 are used progressively)
    "decoder_channels": 196,   # decoder internal width

    # Training
    "num_epochs":    10,
    "learning_rate": 1e-4,
    "lr_decay_epoch": 9,
    "lr_gamma":       0.1,
    "loss":          "msssim",
    "lambda_msssim": 0.9,

    # Evaluation
    "eval_filter_counts": [1..12],
    "jpeg_qualities":     [10, 20, 35, 50, 65, 80],
    "quant_step":         4,     # quantization step (64 levels)
    "quant_bits":         6,     # bits per symbol for bpp formula
    "num_visualizations": 4,

    # Checkpoint
    "model_save_path": "outputs/checkpoints/mcucoder.pth",
}
```

---

## 7. Output Files

| File | Description |
|------|-------------|
| `outputs/checkpoints/mcucoder.pth` | Best model weights (saved during training) |
| `outputs/results/eval_summary.json` | PSNR / MS-SSIM / bpp for all bitrate points |
| `outputs/results/rd_curves.pdf` | Rate-distortion plot (PSNR and MS-SSIM panels) |
| `outputs/results/model_recon_k01_img*.png` | Sample reconstructions at lowest bitrate |
| `outputs/results/jpeg_q10_img*.png` | Sample JPEG reconstructions at lowest quality |

---

## 8. Architecture Summary

```
Input (B, 3, 224, 224)
        │
        ▼
 ┌──────────────┐
 │   Encoder    │  7×7 conv (s=2) → 5×5 conv (s=4) → 3×3 conv (s=1)
 │  3 layers    │  Output: (B, 12, 28, 28)
 └──────┬───────┘
        │  Tail-dropout during training: zero channels k+1 … 12
        │  Quantization during eval:  6-bit uniform per channel
        ▼
 ┌──────────────┐
 │   Decoder    │  AttentionBlock → 3×deconv (stride 2) + residual blocks
 │  deep ResNet │  Output: (B, 3, 224, 224) via sigmoid
 └──────────────┘
        │
        ▼
 Reconstruction x̂

Loss:  L = 0.9 × (1 − MS-SSIM) + 0.1 × MSE

Bitrate (bpp) = Hz × Wz × k × b / (Hx × Wx)
              = 28 × 28 × k × 6 / (224 × 224)  ≈  0.094 × k
```
