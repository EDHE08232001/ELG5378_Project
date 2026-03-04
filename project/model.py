"""Progressive convolutional autoencoder with latent tail dropout."""

from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolution + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """Basic transposed-convolution + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProgressiveAutoencoder(nn.Module):
    """Autoencoder where latent channels can be truncated at inference time."""

    def __init__(self, latent_channels: int = 96):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder progressively downsamples input 224 -> 112 -> 56 -> 28.
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 96, stride=2),
            nn.Conv2d(96, latent_channels, kernel_size=3, stride=1, padding=1),
        )

        # Decoder reconstructs 28 -> 56 -> 112 -> 224.
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, 96),
            DeconvBlock(96, 64),
            DeconvBlock(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def apply_tail_dropout(self, latent: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Randomly zero-out a tail of latent channels during training."""
        if not self.training:
            return latent, 1.0

        keep_fraction = torch.rand(1).item()
        keep_channels = max(1, int(round(keep_fraction * self.latent_channels)))

        # Keep early channels; drop the rest to enforce channel importance ordering.
        dropped = latent.clone()
        dropped[:, keep_channels:, :, :] = 0.0
        return dropped, keep_channels / self.latent_channels

    def truncate_latent(self, latent: torch.Tensor, fraction: float) -> torch.Tensor:
        """Deterministically truncate latent channels to a target fraction."""
        keep_channels = max(1, int(round(fraction * self.latent_channels)))
        truncated = latent.clone()
        truncated[:, keep_channels:, :, :] = 0.0
        return truncated

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        latent = self.encoder(x)
        latent_for_decode, used_fraction = self.apply_tail_dropout(latent)
        reconstruction = self.decoder(latent_for_decode)
        return reconstruction, latent, used_fraction
