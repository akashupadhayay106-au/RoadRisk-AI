"""
RoadRisk AI — CCTV Dataset Preprocessing & Manifest Pipeline
============================================================
Handles image indexing, corrupt file validation, torchvision transformation
pipelines (augmentation for training, normalization for eval), and PyTorch
Dataset/DataLoader instantiation for CCTV accident detection.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from PIL import Image

from src.config import CCTV_TRAIN_DIR, CCTV_TEST_DIR, CCTV_VAL_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Class Constants & Image Standards ─────────────────────────────────────────
CLASS_NAMES = ["Accident", "Non Accident"]
CLASS_TO_IDX = {"Accident": 0, "Non Accident": 1}
IDX_TO_CLASS = {0: "Accident", 1: "Non Accident"}

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def scan_cctv_manifest(
    train_dir: Path = CCTV_TRAIN_DIR,
    val_dir: Path = CCTV_VAL_DIR,
    test_dir: Path = CCTV_TEST_DIR,
    verify_images: bool = False,
) -> pd.DataFrame:
    """
    Scan all split directories and produce an index manifest of all CCTV frames.
    Optionally opens each image to verify header integrity and resolution.
    """
    splits = [
        ("train", train_dir),
        ("val", val_dir),
        ("test", test_dir),
    ]

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    records = []

    for split_name, split_path in splits:
        if not split_path.exists():
            continue

        for cls_name in CLASS_NAMES:
            cls_dir = split_path / cls_name
            if not cls_dir.exists():
                continue

            for img_file in cls_dir.glob("*.*"):
                if img_file.suffix.lower() not in valid_extensions:
                    continue

                width, height, is_corrupt = None, None, False
                file_size_kb = round(img_file.stat().st_size / 1024, 2)

                if verify_images:
                    try:
                        with Image.open(img_file) as im:
                            im.verify()
                        # Re-open to read size (verify closes or alters state)
                        with Image.open(img_file) as im:
                            width, height = im.size
                    except Exception as e:
                        logger.warning(f"Corrupt image detected: {img_file} ({e})")
                        is_corrupt = True

                records.append({
                    "split": split_name,
                    "class_name": cls_name,
                    "class_idx": CLASS_TO_IDX[cls_name],
                    "filename": img_file.name,
                    "filepath": str(img_file),
                    "file_size_kb": file_size_kb,
                    "width": width,
                    "height": height,
                    "is_corrupt": is_corrupt,
                })

    df_manifest = pd.DataFrame(records)
    return df_manifest


def get_image_transforms(is_train: bool = False):
    """
    Get standard torchvision transforms for training (with data augmentation)
    or evaluation (resize + center crop + normalization).
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required for CCTV image transformations.")

    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def load_cctv_dataset(
    split: str = "train",
    transform=None,
    manifest_df: Optional[pd.DataFrame] = None,
):
    """
    Instantiate a PyTorch Dataset for the requested split ('train', 'val', or 'test').
    """
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        raise ImportError("torch is required for CCTV Dataset operations.")

    if transform is None:
        transform = get_image_transforms(is_train=(split == "train"))

    if manifest_df is None:
        manifest_df = scan_cctv_manifest(verify_images=False)

    df_split = manifest_df[manifest_df["split"] == split].reset_index(drop=True)

    class CCTVDataset(Dataset):
        def __init__(self, df: pd.DataFrame, transform=None):
            self.df = df
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row["filepath"]
            label = int(row["class_idx"])

            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

            return image, label, row["filename"]

    return CCTVDataset(df_split, transform=transform)


def get_cctv_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    manifest_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Construct train, val, and test PyTorch DataLoaders.
    """
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError("torch is required for DataLoader generation.")

    if manifest_df is None:
        manifest_df = scan_cctv_manifest()

    train_ds = load_cctv_dataset("train", transform=get_image_transforms(is_train=True), manifest_df=manifest_df)
    val_ds = load_cctv_dataset("val", transform=get_image_transforms(is_train=False), manifest_df=manifest_df)
    test_ds = load_cctv_dataset("test", transform=get_image_transforms(is_train=False), manifest_df=manifest_df)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders


if __name__ == "__main__":
    print("Scanning CCTV Image Dataset...")
    df_m = scan_cctv_manifest()
    print(f"Total Images indexed: {len(df_m)}")
    print(df_m.groupby(["split", "class_name"]).size())
