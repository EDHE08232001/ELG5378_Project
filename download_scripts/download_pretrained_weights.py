"""
Downloda pre-trained weights from Google Drive
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) # path to the current file
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir)) # add the repo root to the system path

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT) # add the repo root to the system path

FILE_ID = "1aWLukhsRV5Fi_DFJUbL5nBdwNhGDpNe0"

def download_weights() -> str:
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Please install it using 'pip install gdown'")
        sys.exit(1)
    
    from src.config import CONFIG

    out_path = os.path.abspath(CONFIG["model_save_path"])
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        print(f"File already exists at {out_path}. Skipping download.")
        return out_path
    
    print(f"Downloading pre-trained weights to {out_path}...")
    gdown.download(id=FILE_ID, output=out_path, quiet=False, fuzzy=True)

    if not os.path.exists(out_path):
        print(f"Failed to download the file. Please check the FILE_ID and your internet connection.")
        sys.exit(1)
    
    print(f"Pre-trained weights downloaded successfully to {out_path}.")

    return out_path

if __name__ == "__main__":
    download_weights()