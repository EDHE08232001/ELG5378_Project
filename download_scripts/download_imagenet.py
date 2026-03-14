import os
import sys

# Ensure the repo root is on sys.path so we can resolve directories correctly
_HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
OUT_DIR = os.path.join(REPO_ROOT, "datasets", "imagenet", "train")

num_images = 300000
dataset_name = "benjamin-paine/imagenet-1k-256x256"

def download_data(num_images: int, dataset_name: str) -> str:
    """Downloads a subset of the ImageNet-1k dataset."""
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("Missing required packages. Please run: pip install datasets tqdm")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {num_images:,} images to: {OUT_DIR}")

    ds = load_dataset(
        dataset_name, 
        split="train", 
        streaming=True, 
        trust_remote_code=True
    )

    for i, example in enumerate(tqdm(ds, total=num_images, desc="Downloading ImageNet")):
        if i >= num_images:
            break
        
        out_path = os.path.join(OUT_DIR, f"{i:07d}.jpg")
        example["image"].convert("RGB").save(out_path)
        
    return OUT_DIR

if __name__ == "__main__":
    download_data(num_images, dataset_name)
