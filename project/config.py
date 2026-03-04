"""Project-wide configuration for progressive learned image compression."""

import os


# ------------------------
# Absolute Path Resolution
# ------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))

DATA_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "datasets"))
IMAGENET_TRAIN_DIR = os.path.abspath(os.path.join(DATA_ROOT, "imagenet", "train"))
KODAK_DIR = os.path.abspath(os.path.join(DATA_ROOT, "kodak"))

ARTIFACT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "artifacts"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(ARTIFACT_ROOT, "checkpoints"))
RESULTS_DIR = os.path.abspath(os.path.join(ARTIFACT_ROOT, "results"))

for required_dir in [ARTIFACT_ROOT, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(required_dir, exist_ok=True)


CONFIG = {
    "train_data_dir": IMAGENET_TRAIN_DIR,
    "val_data_dir": KODAK_DIR,
    "batch_size": 16,
    "num_workers": 4,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "lambda_msssim": 0.7,
    "image_size": 224,
    "latent_channels": 96,
    "quant_bits": 8,
    "model_save_path": os.path.abspath(os.path.join(CHECKPOINT_DIR, "progressive_autoencoder.pt")),
    "device": "cuda",
    "eval_latent_fractions": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    "jpeg_qualities": [15, 25, 40, 55, 70, 85],
    "num_visualizations": 4,
}
