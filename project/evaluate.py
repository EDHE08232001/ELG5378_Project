"""Evaluation utilities for rate-distortion benchmarking."""

import io
import json
import os
from typing import Dict, List

from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from config import CONFIG, RESULTS_DIR
from data import RecursiveImageDataset
from losses import psnr
from model import ProgressiveAutoencoder
from utils import get_device


# ------------------------
# Evaluation Helpers
# ------------------------
def estimate_bpp(input_tensor: torch.Tensor, latent_tensor: torch.Tensor, keep_fraction: float, quant_bits: int) -> float:
    """Estimate raw latent bits-per-pixel using retained channels and quantization bits."""
    _, _, h_x, w_x = input_tensor.shape
    _, channels, h_z, w_z = latent_tensor.shape
    keep_channels = max(1, int(round(keep_fraction * channels)))
    total_bits = keep_channels * h_z * w_z * quant_bits
    return total_bits / float(h_x * w_x)


def jpeg_reconstruct_and_bpp(image_tensor: torch.Tensor, quality: int) -> Dict[str, float]:
    """Compress single image with JPEG in-memory and return reconstructed tensor + bpp stats."""
    pil_image = Image.fromarray((image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)

    bit_count = len(buffer.getvalue()) * 8
    width, height = pil_image.size
    bpp = bit_count / float(width * height)

    buffer.seek(0)
    reconstructed = Image.open(buffer).convert("RGB")
    reconstructed_np = np.asarray(reconstructed).astype("float32") / 255.0
    reconstructed_tensor = torch.from_numpy(reconstructed_np).permute(2, 0, 1)

    return {"bpp": bpp, "reconstructed": reconstructed_tensor}


def evaluate_model() -> str:
    """Evaluate trained model vs JPEG and save JSON summary."""
    device = get_device(CONFIG["device"])
    model_path = os.path.abspath(CONFIG["model_save_path"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train first.")

    dataset = RecursiveImageDataset(CONFIG["val_data_dir"], image_size=CONFIG["image_size"])
    if len(dataset) == 0:
        raise RuntimeError(f"No validation images found at: {CONFIG['val_data_dir']}")

    model = ProgressiveAutoencoder(latent_channels=CONFIG["latent_channels"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    summary: Dict[str, List[Dict[str, float]]] = {"model": [], "jpeg": []}

    for fraction in CONFIG["eval_latent_fractions"]:
        avg_psnr = 0.0
        avg_bpp = 0.0

        for idx in tqdm(range(len(dataset)), desc=f"Model fraction={fraction:.2f}"):
            image = dataset[idx].unsqueeze(0).to(device)
            with torch.no_grad():
                latent = model.encoder(image)
                truncated = model.truncate_latent(latent, fraction=fraction)
                reconstruction = model.decoder(truncated)

            avg_psnr += psnr(reconstruction, image)
            avg_bpp += estimate_bpp(image, latent, keep_fraction=fraction, quant_bits=CONFIG["quant_bits"])

            if idx < CONFIG["num_visualizations"] and fraction == CONFIG["eval_latent_fractions"][0]:
                save_image(
                    reconstruction.cpu(),
                    os.path.abspath(os.path.join(RESULTS_DIR, f"model_recon_{idx}.png")),
                )

        avg_psnr /= len(dataset)
        avg_bpp /= len(dataset)
        summary["model"].append({"fraction": fraction, "bpp": avg_bpp, "psnr": avg_psnr})

    for quality in CONFIG["jpeg_qualities"]:
        avg_psnr = 0.0
        avg_bpp = 0.0

        for idx in tqdm(range(len(dataset)), desc=f"JPEG quality={quality}"):
            image = dataset[idx]
            jpeg_stats = jpeg_reconstruct_and_bpp(image, quality=quality)
            reconstructed = jpeg_stats["reconstructed"].unsqueeze(0).to(device)
            target = image.unsqueeze(0).to(device)
            avg_psnr += psnr(reconstructed, target)
            avg_bpp += float(jpeg_stats["bpp"])

            if idx < CONFIG["num_visualizations"] and quality == CONFIG["jpeg_qualities"][0]:
                save_image(
                    reconstructed.cpu(),
                    os.path.abspath(os.path.join(RESULTS_DIR, f"jpeg_recon_{idx}.png")),
                )

        avg_psnr /= len(dataset)
        avg_bpp /= len(dataset)
        summary["jpeg"].append({"quality": quality, "bpp": avg_bpp, "psnr": avg_psnr})

    output_path = os.path.abspath(os.path.join(RESULTS_DIR, "evaluation_summary.json"))
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved evaluation summary to: {output_path}")
    return output_path


if __name__ == "__main__":
    evaluate_model()
