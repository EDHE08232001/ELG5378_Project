"""Dataset and dataloader creation for training and evaluation."""

import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RecursiveImageDataset(Dataset):
    """Loads all image files recursively under a root directory."""

    def __init__(self, root_dir: str, image_size: int = 224):
        self.root_dir = os.path.abspath(root_dir)
        self.image_paths = self._scan_images(self.root_dir)
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def _scan_images(root_dir: str) -> List[str]:
        image_paths: List[str] = []
        for current_root, _, files in os.walk(root_dir):
            for file_name in files:
                extension = os.path.splitext(file_name)[1].lower()
                if extension in VALID_EXTENSIONS:
                    image_paths.append(os.path.abspath(os.path.join(current_root, file_name)))
        image_paths.sort()
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)


def build_dataloaders(
    train_dir: str,
    val_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation dataloaders using recursive image scanning."""
    train_dataset = RecursiveImageDataset(train_dir, image_size=image_size)
    val_dataset = RecursiveImageDataset(val_dir, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
