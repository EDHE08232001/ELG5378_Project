# Progressive Learned Image Compression Project

This folder contains a full modular Python implementation of the ELG5378 project proposal:

- Progressive convolutional autoencoder
- Stochastic latent tail-dropout during training
- Rate-distortion style evaluation against JPEG
- Interactive entry script that asks whether to **train** or **evaluate**

## 1) Environment setup

From the repository root (`/workspace/ELG5378_Project`):

```bash
python -m venv /workspace/ELG5378_Project/.venv
source /workspace/ELG5378_Project/.venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchmetrics pillow tqdm numpy
```

## 2) Dataset download and folder layout

The code uses **absolute paths built with `os.path.abspath(...)`** in `project/config.py`.

Expected absolute data directories:

- ImageNet training subset root: `/workspace/ELG5378_Project/datasets/imagenet/train`
- Kodak validation root: `/workspace/ELG5378_Project/datasets/kodak`

Create them first:

```bash
mkdir -p /workspace/ELG5378_Project/datasets/imagenet/train
mkdir -p /workspace/ELG5378_Project/datasets/kodak
```

### 2.1 ImageNet training subset

You can use any ImageNet subset you have permission to access.

Place JPEG/PNG images recursively under:

```text
/workspace/ELG5378_Project/datasets/imagenet/train
```

The loader scans recursively, so class subfolders are optional.

### 2.2 Kodak validation set

Download Kodak images into:

```text
/workspace/ELG5378_Project/datasets/kodak
```

Example (24 images):

```bash
for i in $(seq -w 1 24); do
  curl -L "https://r0k.us/graphics/kodak/kodak/kodim${i}.png" \
    -o "/workspace/ELG5378_Project/datasets/kodak/kodim${i}.png"
done
```

## 3) Project structure

```text
/workspace/ELG5378_Project/project
├── config.py        # absolute paths + hyperparameters
├── data.py          # recursive dataset and dataloaders
├── model.py         # progressive autoencoder + latent truncation
├── losses.py        # proposal loss (MS-SSIM + MSE) and PSNR
├── train.py         # training loop + checkpointing
├── evaluate.py      # model/JPEG evaluation + JSON output
├── utils.py         # seed/device/format helpers
└── main.py          # interactive entry point
```

## 4) How to run

Go to project folder:

```bash
cd /workspace/ELG5378_Project/project
```

Run:

```bash
python /workspace/ELG5378_Project/project/main.py
```

You will be prompted with:

1. `training`
2. `evaluating`

### 4.1 Training flow

Choose `1`.

- Trains model on ImageNet subset path from config.
- Validates on Kodak each epoch.
- Saves best checkpoint to:
  - `/workspace/ELG5378_Project/project/artifacts/checkpoints/progressive_autoencoder.pt`

### 4.2 Evaluation flow

Choose `2`.

- Loads trained checkpoint.
- Evaluates multiple latent fractions for progressive decoding.
- Evaluates JPEG qualities as baseline.
- Saves summary JSON to:
  - `/workspace/ELG5378_Project/project/artifacts/results/evaluation_summary.json`
- Saves sample reconstructions in:
  - `/workspace/ELG5378_Project/project/artifacts/results`

## 5) Configuration

Edit hyperparameters in:

- `/workspace/ELG5378_Project/project/config.py`

Notable options:

- `num_epochs`, `batch_size`, `learning_rate`
- `latent_channels`
- `eval_latent_fractions`
- `jpeg_qualities`

## 6) Notes

- Input images are center-cropped to `224x224` to match the project proposal.
- If no checkpoint exists, evaluation raises an explicit error asking you to train first.
- All runtime paths are normalized to absolute paths with `os.path.abspath`.
