# MCUCoder — Progressive Learned Image Compression

## Overview
A research implementation of a progressive learned image compression system that supports multiple bitrate-quality operating points from a single trained model using "stochastic tail-dropout" during training.

## Architecture
- **Encoder**: 3-layer CNN reducing 224x224 images to 28x28x12 latent representation
- **Progressive Latent**: During training, only the first k channels (random k in [1,12]) are kept
- **Decoder**: Deep residual network (~196 channels wide) reconstructing images from k active channels
- **Loss Function**: λ·(1-MS-SSIM) + (1-λ)·MSE (λ=0.9 default)

## Project Layout
```
main.py              # CLI entry point (Train / Evaluate / Prepare Data)
src/
  config.py          # Centralized hyperparameters and path resolution
  model.py           # Encoder, Decoder, MCUCoder model definitions
  data.py            # Dataset loaders and augmentation
  losses.py          # Progressive loss function
  train.py           # Training loop with LR scheduling and validation
  evaluate.py        # Rate-Distortion evaluation and plotting
  prepare_data.py    # ImageNet preprocessing utility
  utils.py           # Shared utilities
datasets/            # User-provided datasets (Kodak, ImageNet subset)
outputs/             # Auto-created: checkpoints/ and results/
```

## Technology Stack
- **Language**: Python 3.12
- **ML Framework**: PyTorch 2.10 (with CUDA support)
- **Compression Layers**: CompressAI (AttentionBlock, ResidualBottleneckBlock)
- **Metrics**: TorchMetrics (MS-SSIM)
- **Image Processing**: Pillow, OpenCV
- **Visualization**: Matplotlib

## Workflow
- **Start application**: `python3 main.py` — interactive CLI console
  - Option 1: Train model on ImageNet subset, validate on Kodak
  - Option 2: Run rate-distortion evaluation and plot RD curves
  - Option 3: Pre-process raw ImageNet images for training

## Dependencies
All installed via pip (see requirements.txt):
- torch, torchvision, torchmetrics, compressai
- Pillow, opencv-python, numpy, tqdm, matplotlib

## Dataset Setup
- Place Kodak validation images in `datasets/kodak/`
- Place ImageNet images in `datasets/imagenet_raw/` then run option 3 to prepare
