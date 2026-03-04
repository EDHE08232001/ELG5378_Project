"""Loss and quality metric helpers."""

import math

import torch
import torch.nn.functional as F
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


class DistortionLoss:
    """Combined MS-SSIM and MSE loss used in the proposal."""

    def __init__(self, lambda_msssim: float, device: torch.device):
        self.lambda_msssim = lambda_msssim
        self.msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        msssim_score = self.msssim_metric(prediction, target)
        mse = F.mse_loss(prediction, target)
        return self.lambda_msssim * (1.0 - msssim_score) + (1.0 - self.lambda_msssim) * mse


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute batch-average PSNR in dB."""
    mse = F.mse_loss(prediction, target).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))
