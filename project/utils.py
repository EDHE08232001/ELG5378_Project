"""Utility helpers for pathing, logging, and metrics."""

import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed common random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred_device: str = "cuda") -> torch.device:
    """Get best available compute device with graceful CPU fallback."""
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    """Create a directory path if missing and return absolute path."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def format_metrics(metrics: Dict[str, float]) -> str:
    """Create a compact printable metrics string."""
    return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
