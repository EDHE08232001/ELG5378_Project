"""Training loop for progressive autoencoder."""

import os

import torch
from torch.optim import Adam
from tqdm import tqdm

from config import CONFIG
from data import build_dataloaders
from losses import DistortionLoss, psnr
from model import ProgressiveAutoencoder
from utils import format_metrics, get_device, set_seed


# ------------------------
# Training Pipeline
# ------------------------
def train_model() -> str:
    """Train the model and save the best checkpoint path."""
    set_seed(42)
    device = get_device(CONFIG["device"])

    train_loader, val_loader = build_dataloaders(
        train_dir=CONFIG["train_data_dir"],
        val_dir=CONFIG["val_data_dir"],
        image_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    model = ProgressiveAutoencoder(latent_channels=CONFIG["latent_channels"]).to(device)
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = DistortionLoss(lambda_msssim=CONFIG["lambda_msssim"], device=device)

    best_val_loss = float("inf")
    save_path = os.path.abspath(CONFIG["model_save_path"])

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}")
        for images in progress:
            images = images.to(device)

            reconstructed, _, _ = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += psnr(reconstructed.detach(), images)

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= max(1, len(train_loader))
        train_psnr /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                reconstructed, _, _ = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
                val_psnr += psnr(reconstructed, images)

        val_loss /= max(1, len(val_loader))
        val_psnr /= max(1, len(val_loader))

        metrics = {
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
        }
        print(format_metrics(metrics))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved improved checkpoint to: {save_path}")

    return save_path


if __name__ == "__main__":
    train_model()
