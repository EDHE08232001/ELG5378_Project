# MCUCoder — Progressive Learned Image Compression

## Overview
A research implementation of a progressive learned image compression system. It supports multiple bitrate-quality operating points from a single trained model using "stochastic tail-dropout" during training. A decoder can reconstruct an image using any prefix of the 12 latent channels, trading off file size (bitrate) for image quality without needing separate models.

## Architecture
- **Encoder**: 3-layer lightweight CNN reducing 224x224 images to 28x28x12 latent representation
- **Progressive Latent**: During training, a random k in [1,12] is chosen; only the first k channels are kept
- **Decoder**: Deep residual network (~196 channels wide) with attention blocks, reconstructing images from k active channels
- **Loss Function**: λ·(1-MS-SSIM) + (1-λ)·MSE where λ defaults to 0.9

## Project Layout
```
main.py                  # CLI entry point (Train / Evaluate / Prepare Data)
requirements.txt         # Python dependencies
src/
  __init__.py            # Package init
  config.py              # Centralized hyperparameters and path resolution
  model.py               # Encoder, Decoder, MCUCoder model definitions
  data.py                # Dataset loaders (RecursiveImageDataset) and augmentation
  losses.py              # Progressive loss function (MS-SSIM + MSE)
  train.py               # Training loop with LR scheduling and periodic validation
  evaluate.py            # Rate-Distortion evaluation, JPEG comparison, and plotting
  prepare_data.py        # ImageNet preprocessing (select, resize, add noise)
  utils.py               # Shared utilities
  main.py                # Alternate entry point within src/
datasets/                # User-provided datasets (created by user)
  imagenet/train/        # Raw ImageNet training images
  imagenet_prepared/     # Processed flat PNG output (created by prepare step)
  kodak/                 # Kodak validation images (24 images)
outputs/                 # Auto-created at runtime
  checkpoints/           # Saved model weights (.pth files)
  results/               # RD plots, metrics JSON, reconstructed image samples
```

## Technology Stack
- **Language**: Python 3.12
- **ML Framework**: PyTorch (with torchvision)
- **Compression Layers**: CompressAI (AttentionBlock, ResidualBottleneckBlock)
- **Metrics**: TorchMetrics (MS-SSIM)
- **Image Processing**: Pillow, OpenCV (cv2)
- **Visualization**: Matplotlib
- **Utilities**: NumPy, tqdm

## Dependencies
All managed via pip (listed in requirements.txt):
- torch, torchvision, torchmetrics
- compressai
- Pillow, opencv-python
- numpy, tqdm, matplotlib

## Workflow
- **Start application**: `python3 main.py` — interactive CLI (console output)
  - Option 1: Train model on ImageNet subset, validate on Kodak
  - Option 2: Run rate-distortion evaluation and plot RD curves
  - Option 3: Pre-process raw ImageNet images for training

## Key Configuration (src/config.py)
- `image_size`: 224 (center-crop size)
- `batch_size`: 16
- `latent_channels`: 12 (progressive channels)
- `decoder_channels`: 196 (decoder width)
- `num_epochs`: 10 (increase for production runs)
- `learning_rate`: 1e-4
- `num_images_to_select`: 300,000 (for ImageNet prep, can be reduced)

## Dataset Setup
- **Training**: Place images in `datasets/imagenet/train/`, then run option 3 to prepare. Any collection of .jpg/.jpeg/.png images works — does not have to be ImageNet specifically.
- **Validation**: Place Kodak images in `datasets/kodak/`
- The `num_images_to_select` config can be lowered for smaller training runs.

## Notes
- Replit does not provide GPU resources; training runs on CPU only and will be slow for large datasets.
- For experimentation, use a small image set and fewer epochs.
- The model checkpoint is saved to `outputs/checkpoints/mcucoder.pth`.
