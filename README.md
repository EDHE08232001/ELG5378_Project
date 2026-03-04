# MCUCoder — Progressive Learned Image Compression

A convolutional autoencoder that supports multiple bitrate–quality operating points from a single trained model. During training, random *tail-dropout* of latent channels forces early channels to carry the most important image information. At inference time, any prefix of 1–12 channels can be decoded independently, trading bitrate for reconstruction quality without re-encoding.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [File Descriptions](#3-file-descriptions)
4. [Environment Setup](#4-environment-setup)
5. [Dataset Download](#5-dataset-download)
6. [How to Run](#6-how-to-run)
7. [Configuration Reference](#7-configuration-reference)
8. [Output Files](#8-output-files)
9. [Architecture Summary](#9-architecture-summary)

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
├── main.py                      ← entry point (always run from here)
├── env_check.py                 ← quick device check utility
├── requirements.txt             ← pip dependencies
├── README.md
├── .gitignore
│
├── src/                         ← source package
│   ├── __init__.py
│   ├── main.py                  ← stub (do not run directly)
│   ├── config.py                ← all paths and hyperparameters
│   ├── data.py                  ← RecursiveImageDataset + DataLoaders
│   ├── model.py                 ← Encoder, Decoder, MCUCoder
│   ├── losses.py                ← ProgressiveLoss, PSNR, MS-SSIM helpers
│   ├── train.py                 ← training loop + LR schedule + checkpointing
│   ├── evaluate.py              ← RD evaluation + quantization + plots
│   ├── prepare_data.py          ← ImageNet high-res selection & preprocessing
│   └── utils.py                 ← set_seed, get_device, format_metrics
│
├── download_scripts/
│   ├── download_kodak.sh        ← Kodak dataset download (macOS/Linux)
│   ├── download_kodak.bat       ← Kodak dataset download (Windows)
│   └── download_imagenet.py     ← ImageNet subset download via Hugging Face
|   |__ download_pretrained_weights.py 
│
├── datasets/                    ← place raw data here (created by you)
│   ├── kodak/                   ← 24 Kodak PNG images
│   ├── imagenet/
│   │   └── train/               ← raw ImageNet class subdirectories
│   └── imagenet_prepared/       ← flat PNG output from prepare step (auto)
│
└── outputs/                     ← auto-created at runtime
    ├── checkpoints/
    │   └── mcucoder.pth         ← best model checkpoint
    └── results/
        ├── eval_summary.json    ← numeric RD results
        ├── rd_curves.pdf        ← rate-distortion plot
        └── *.png                ← sample reconstructions
```

---

## 3. File Descriptions

### Root-level files

**`main.py`** — The sole entry point for the project. Presents an interactive menu to select between training (option 1), evaluation (option 2), and ImageNet preparation (option 3). Always run this from the repository root.

**`env_check.py`** — Standalone utility that imports `get_device()` from `src/utils.py` and prints the detected compute device (CUDA / MPS / CPU). Useful for verifying your environment before a long training run.

**`requirements.txt`** — All pip dependencies. Core packages are `torch`, `torchvision`, `torchmetrics`, `compressai`, `Pillow`, `opencv-python`, `numpy`, `tqdm`, `matplotlib`, and `datasets` (Hugging Face).

**`.gitignore`** — Excludes virtual environments (`venv/`, `venvMAC/`, `venvPC/`), model weights (`*.pth`, `*.pt`), outputs, datasets, IDE files, and OS junk.

---

### `src/` package

**`src/config.py`** — Single source of truth for all paths and hyperparameters. Resolves all directory paths as absolute paths relative to the repository root so the project runs correctly from any working directory. Automatically selects `imagenet_prepared/` as the training directory if it exists, falling back to raw `imagenet/train/`. All training, evaluation, and architecture hyperparameters are stored in the `CONFIG` dict.

**`src/model.py`** — Defines the full MCUCoder architecture:
- `ResidualBottleneckBlock` — 1×1 → 3×3 → 1×1 bottleneck block with learned skip connection.
- `Encoder` — Lightweight 3-layer CNN that maps `(B, 3, 224, 224)` → `(B, 12, 28, 28)`.
- `Decoder` — Deep residual decoder with `compressai` `AttentionBlock`s and three deconvolution stages that maps `(B, 12, 28, 28)` → `(B, 3, 224, 224)`.
- `MCUCoder` — Full model combining encoder and decoder with stochastic tail-dropout. During training, a random number of channels `k ∈ [1, 12]` is kept active; during evaluation, a fixed `keep_fraction` can be specified.

**`src/data.py`** — Dataset and DataLoader utilities:
- `RecursiveImageDataset` — Recursively scans any directory for images (supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`). Resizes the shortest edge to `image_size` then center-crops to a square. Works for both flat directories (Kodak) and class-subdirectory trees (ImageNet).
- `build_dataloaders` — Returns a `(train_loader, val_loader)` pair. The training loader shuffles and drops incomplete batches; the validation loader uses `batch_size=1`.

**`src/losses.py`** — Loss functions and quality metrics:
- `ProgressiveLoss` — Combined MS-SSIM + MSE loss: `L = λ·(1−MS-SSIM) + (1−λ)·MSE`. Uses `torchmetrics` for MS-SSIM computation.
- `MSELoss` — Plain mean-squared-error alternative.
- `compute_psnr` — Batch-averaged PSNR in dB; returns 99.0 dB for numerically perfect reconstructions.
- `compute_msssim_db` — Batch-averaged MS-SSIM converted to dB scale: `−10·log₁₀(1−MS-SSIM)`.

**`src/train.py`** — Full training loop:
- Builds DataLoaders from config, instantiates the model, chooses loss based on `CONFIG["loss"]`, and sets up an Adam optimizer with a StepLR scheduler.
- Each epoch: trains with stochastic tail-dropout, then validates at three representative bitrate levels (2 / 6 / 12 active channels).
- Saves a checkpoint to `outputs/checkpoints/mcucoder.pth` whenever validation loss improves.

**`src/evaluate.py`** — Rate-distortion evaluation:
- Loads a saved checkpoint and evaluates MCUCoder on the Kodak set for each channel count k ∈ {1, …, 12}, applying per-channel uniform quantization to simulate realistic bitrates.
- Evaluates a JPEG baseline at 6 quality levels for comparison.
- Computes bpp using the formula `bpp = (Hz·Wz·k·b) / (Hx·Wx)`.
- Saves: `eval_summary.json` (numeric results), `rd_curves.pdf` (two-panel RD plot), and sample reconstruction PNGs.

**`src/prepare_data.py`** — ImageNet preprocessing:
- Scans the raw ImageNet directory for all images and selects the N highest-resolution ones.
- For each selected image: optionally halves resolution if the shorter side exceeds 512 px (bicubic, via OpenCV), adds small uniform noise, and saves as a flat PNG in `datasets/imagenet_prepared/`.

**`src/utils.py`** — General utilities:
- `set_seed(seed)` — Seeds Python, NumPy, and PyTorch RNGs for reproducibility.
- `get_device(preferred)` — Returns the best available device: CUDA → MPS → CPU.
- `ensure_dir(path)` — Creates a directory and all parents if they don't exist.
- `format_metrics(metrics)` — Formats a `dict[str, float]` as a compact single-line string for logging.

**`src/__init__.py`** — Empty package marker.

**`src/main.py`** — Stub file. The actual entry point is the root-level `main.py`.

---

### `download_scripts/`

**`download_scripts/download_kodak.sh`** — zsh script to download all 24 Kodak PNG images into `datasets/kodak/`. Uses `curl` in a loop from `r0k.us`. Intended for macOS / Linux.

**`download_scripts/download_kodak.bat`** — Windows equivalent of the above, using `curl` in a `for /L` batch loop.

**`download_scripts/download_imagenet.py`** — Python script that streams the `imagenet-1k` dataset from Hugging Face and saves up to 300,000 images as JPEGs into `datasets/imagenet/train/`. Requires `datasets` and `tqdm`.

---

## 4. Environment Setup

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux (zsh)
# .venv\Scripts\activate         # Windows
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU note:** For CUDA support, install the matching `torch` wheel from <https://pytorch.org/get-started/locally/>. The code auto-detects CUDA → MPS → CPU at runtime.

### Verify your device

```bash
python env_check.py
# Expected output: Using device: mps  (or cuda / cpu)
```

---

## 5. Dataset Download

Create the expected directories first:

```bash
mkdir -p datasets/imagenet/train
mkdir -p datasets/kodak
```

### Kodak Validation Set (required for training + evaluation)

**macOS / Linux:**
```bash
zsh download_scripts/download_kodak.sh
```

**Windows:**
```bat
download_scripts\download_kodak.bat
```

Downloads 24 lossless PNG images (~35 MB) to `datasets/kodak/`.

### ImageNet Training Subset (required for training only)

**Option A — you already have ImageNet locally:**

Place or symlink the `train/` directory at `datasets/imagenet/train/`. The loader scans recursively, so class subdirectories are handled automatically.

**Option B — download via Hugging Face (~300k images):**

```bash
python download_scripts/download_imagenet.py
```

Requires a Hugging Face account and `datasets` package. Downloads images as JPEGs to `datasets/imagenet/train/`.

**Option C — run the built-in preparation step after placing raw data:**

```bash
python main.py
# → choose 3 (Prepare)
```

### Pre-trained Weights (skip training)

```bash
pip install gdown
mkdir -p outputs/checkpoints
gdown "1aWLukhsRV5Fi_DFJUbL5nBdwNhGDpNe0" -O outputs/checkpoints/mcucoder.pth
```

---

## 6. How to Run

**All commands must be run from the repository root** (`ELG5378_Project/`).

### Recommended order for a full run from scratch

```bash
# Step 1 — Download Kodak validation set
zsh download_scripts/download_kodak.sh

# Step 2 — Download or place ImageNet training images
python download_scripts/download_imagenet.py   # or place manually

# Step 3 — (Optional) Preprocess ImageNet
python main.py   # → choose 3

# Step 4 — Train
python main.py   # → choose 1

# Step 5 — Evaluate
python main.py   # → choose 2
```

### Interactive menu

```bash
python main.py
```

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

### Option 1 — Train

```
Enter 1, 2, or 3: 1
```

- Loads ImageNet images from `datasets/imagenet_prepared/` (if it exists) or `datasets/imagenet/train/`.
- Validates on Kodak after each epoch at 3 bitrate levels (2 / 6 / 12 active channels).
- Saves the best checkpoint (by validation loss) to `outputs/checkpoints/mcucoder.pth`.

**Key hyperparameters to tune in `src/config.py`:**

| Key | Default | Notes |
|-----|---------|-------|
| `num_epochs` | 10 | Use 50–150 for production quality |
| `batch_size` | 16 | Reduce if GPU memory is limited |
| `learning_rate` | 1e-4 | Adam LR |
| `lr_decay_epoch` | 9 | Epoch at which LR decays by ×0.1 |
| `loss` | `"msssim"` | `"msssim"` or `"mse"` |
| `lambda_msssim` | 0.9 | MS-SSIM weight in the combined loss |
| `decoder_channels` | 196 | Internal decoder width (reduce for faster training) |

### Option 2 — Evaluate

```
Enter 1, 2, or 3: 2
```

- Requires a trained checkpoint at `outputs/checkpoints/mcucoder.pth`.
- Evaluates MCUCoder for k ∈ {1, …, 12} active channels on all 24 Kodak images.
- Applies uniform quantization (`step=4`, `bits=6`) to simulate realistic bitrates.
- Runs JPEG baseline at 6 quality levels.
- Saves outputs to `outputs/results/`.

### Option 3 — Prepare ImageNet

```
Enter 1, 2, or 3: 3
```

- Requires raw ImageNet images in `datasets/imagenet/train/`.
- Selects the 300k highest-resolution images, optionally halves very large ones, adds small noise, and saves flat PNGs to `datasets/imagenet_prepared/`.
- Run once before training; subsequent training runs will automatically use the prepared directory.

---

## 7. Configuration Reference

All settings are in `src/config.py` inside the `CONFIG` dict.

```python
CONFIG = {
    # Data paths (auto-resolved at import time)
    "train_data_dir": ...,        # imagenet_prepared/ if it exists, else imagenet/train/
    "val_data_dir":   ...,        # datasets/kodak/
    "image_size":     224,        # center-crop size (px)

    # DataLoader
    "batch_size":   16,
    "num_workers":  4,

    # Model architecture
    "latent_channels":  12,       # progressive bitrate levels: 1/12 … 12/12
    "decoder_channels": 196,      # internal decoder width

    # Training
    "num_epochs":     10,
    "learning_rate":  1e-4,
    "lr_decay_epoch": 9,
    "lr_gamma":       0.1,
    "loss":           "msssim",   # "msssim" or "mse"
    "lambda_msssim":  0.9,

    # Evaluation
    "eval_filter_counts": [1..12],
    "jpeg_qualities":     [10, 20, 35, 50, 65, 80],
    "quant_step":         4,      # uniform quantization step (64 levels)
    "quant_bits":         6,      # bits per quantized symbol
    "num_visualizations": 4,      # sample reconstructions saved per run

    # Checkpointing
    "model_save_path": "outputs/checkpoints/mcucoder.pth",

    # ImageNet preparation
    "imagenet_raw_dir":     "datasets/imagenet/train/",
    "imagenet_out_dir":     "datasets/imagenet_prepared/",
    "num_images_to_select": 300_000,
}
```

---

## 8. Output Files

| File | Description |
|------|-------------|
| `outputs/checkpoints/mcucoder.pth` | Best model weights saved during training |
| `outputs/results/eval_summary.json` | PSNR / MS-SSIM / bpp for all bitrate points |
| `outputs/results/rd_curves.pdf` | Two-panel RD plot (PSNR and MS-SSIM vs bpp) |
| `outputs/results/model_recon_k01_img*.png` | Sample reconstructions at lowest bitrate (k=1) |
| `outputs/results/jpeg_q10_img*.png` | Sample JPEG reconstructions at lowest quality |

---

## 9. Architecture Summary

```
Input (B, 3, 224, 224)
        │
        ▼
 ┌──────────────┐
 │   Encoder    │  7×7 conv (s=2) → 5×5 conv (s=4) → 3×3 conv (s=1)
 │  3 layers    │  Output: (B, 12, 28, 28)
 └──────┬───────┘
        │  Tail-dropout during training: zero channels k+1 … 12
        │  Quantization during eval:     6-bit uniform per channel
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