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

import logging
from typing import Dict, Any, Optional
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


def extract_source_video_id(filename: str) -> str:
    """
    Extract the source video / sequence identifier from a CCTV frame filename.
    
    Heuristic:
    - Many CCTV datasets extract frames with naming conventions like:
      'acc1 (13).jpg' -> sequence 'acc1'
      'test10_12.jpg' -> sequence 'test10'
      '5_14.jpg'      -> sequence '5'
      'seq_01_frame_04.jpg' -> sequence 'seq_01'
    - Matches the prefix before parentheses, spaces, or the second underscore.
    
    Limitations:
    - If video files were renamed arbitrarily without prefix retention,
      individual frames from the same camera could still share latent backgrounds.
    - Frames sharing similar road geometry from different cameras might not be grouped.
    """
    import re
    # Remove file extension
    base = Path(filename).stem
    # Match prefix before space/parenthesis or trailing underscore+number
    # e.g., 'acc1 (12)' -> 'acc1', 'test10_12' -> 'test10', '5_10' -> '5'
    m = re.match(r"^([a-zA-Z0-9_\-]+?)(?:[\s\(]|_\d+$)", base)
    if m:
        return m.group(1).strip()
    # Fallback: if single underscore, take prefix
    parts = base.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return base


def scan_cctv_manifest(
    train_dir: Path = CCTV_TRAIN_DIR,
    val_dir: Path = CCTV_VAL_DIR,
    test_dir: Path = CCTV_TEST_DIR,
    verify_images: bool = False,
) -> pd.DataFrame:
    """
    Scan all split directories and produce an index manifest of all CCTV frames.
    Optionally opens each image to verify header integrity and resolution.
    Includes extracted source video sequence IDs for split-leakage auditing.
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
                video_group = extract_source_video_id(img_file.name)

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
                    "video_group": video_group,
                    "width": width,
                    "height": height,
                    "is_corrupt": is_corrupt,
                })

    df_manifest = pd.DataFrame(records)
    return df_manifest


def audit_split_leakage(manifest_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Audit potential video sequence leakage across naive train/val/test splits.
    
    Returns structured metrics on cross-split contamination:
    - Number of video sequences leaking across multiple splits
    - Count and fraction of frames belonging to contaminated sequences
    - Detailed breakdown per class
    - Heuristic limitations disclosure
    """
    if manifest_df is None:
        manifest_df = scan_cctv_manifest(verify_images=False)

    if manifest_df.empty:
        return {
            "status": "empty",
            "total_frames": 0,
            "total_groups": 0,
            "leaking_groups_count": 0,
            "leakage_percentage": 0.0,
        }

    # Group by (class_name, video_group) and collect set of splits
    group_splits = (
        manifest_df.groupby(["class_name", "video_group"])["split"]
        .unique()
        .reset_index()
    )
    group_splits["split_count"] = group_splits["split"].apply(len)
    group_splits["is_leaking"] = group_splits["split_count"] > 1

    leaking_groups = group_splits[group_splits["is_leaking"]]
    leaking_group_set = set(zip(leaking_groups["class_name"], leaking_groups["video_group"]))

    # Flag leaking frames in main manifest
    manifest_df = manifest_df.copy()
    manifest_df["is_in_leaking_group"] = manifest_df.apply(
        lambda r: (r["class_name"], r["video_group"]) in leaking_group_set, axis=1
    )

    total_frames = len(manifest_df)
    leaking_frames = int(manifest_df["is_in_leaking_group"].sum())
    leak_pct = round((leaking_frames / total_frames) * 100, 2) if total_frames > 0 else 0.0

    class_leak_breakdown = {}
    for cls in CLASS_NAMES:
        cls_df = manifest_df[manifest_df["class_name"] == cls]
        cls_groups = group_splits[group_splits["class_name"] == cls]
        cls_leaking = cls_groups[cls_groups["is_leaking"]]
        class_leak_breakdown[cls] = {
            "total_groups": len(cls_groups),
            "leaking_groups": len(cls_leaking),
            "total_frames": len(cls_df),
            "leaking_frames": int(cls_df["is_in_leaking_group"].sum()),
            "leakage_percentage": round((cls_df["is_in_leaking_group"].sum() / max(1, len(cls_df))) * 100, 2),
            "sample_leaking_groups": cls_leaking["video_group"].head(8).tolist(),
        }

    return {
        "status": "audited",
        "total_frames": total_frames,
        "total_groups": len(group_splits),
        "leaking_groups_count": len(leaking_groups),
        "leaking_frames_count": leaking_frames,
        "leakage_percentage": leak_pct,
        "class_breakdown": class_leak_breakdown,
        "heuristic_description": (
            "Prefix-based sequence clustering (extracting common prefix before frame delimiter). "
            "Frames from the same sequence exhibit high temporal autocorrelation and background similarity."
        ),
        "mitigation": (
            "Enforce GroupKFold / GroupShuffleSplit so that all frames of any sequence ID "
            "are strictly partitioned into only one split (train, val, or test)."
        ),
    }


def create_leakage_free_splits(
    manifest_df: Optional[pd.DataFrame] = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate clean, leakage-free grouped splits where all frames from any given
    video group reside exclusively in ONE split (train, val, or test).
    
    Maintains class stratification across video groups.
    """
    if manifest_df is None:
        manifest_df = scan_cctv_manifest(verify_images=False)

    df = manifest_df.copy()
    if df.empty:
        return df

    import numpy as np
    rng = np.random.RandomState(random_state)

    df["grouped_split"] = "train"

    for cls in df["class_name"].unique():
        cls_mask = df["class_name"] == cls
        cls_groups = np.array(df[cls_mask]["video_group"].unique())
        rng.shuffle(cls_groups)

        n_groups = len(cls_groups)
        n_test = max(1, int(round(n_groups * test_ratio)))
        n_val = max(1, int(round(n_groups * val_ratio)))

        test_groups = set(cls_groups[:n_test])
        val_groups = set(cls_groups[n_test : n_test + n_val])
        train_groups = set(cls_groups[n_test + n_val :])

        df.loc[cls_mask & df["video_group"].isin(test_groups), "grouped_split"] = "test"
        df.loc[cls_mask & df["video_group"].isin(val_groups), "grouped_split"] = "val"
        df.loc[cls_mask & df["video_group"].isin(train_groups), "grouped_split"] = "train"

    # Save grouped manifest artifact for reproducibility
    try:
        grouped_manifest_path = PROCESSED_DIR / "cctv_grouped_manifest.csv"
        df.to_csv(grouped_manifest_path, index=False)
        logger.info(f"Saved leakage-free grouped CCTV manifest to {grouped_manifest_path}")
    except Exception as e:
        logger.warning(f"Could not persist grouped manifest: {e}")

    return df


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
    split_col: str = "split",
):
    """
    Instantiate a PyTorch Dataset for the requested split ('train', 'val', or 'test').
    Accepts split_col ('split' for naive directory split, 'grouped_split' for leakage-free split).
    """
    try:
        import torch  # noqa: F401  (intentional availability probe)
        from torch.utils.data import Dataset
    except ImportError:
        raise ImportError("torch is required for CCTV Dataset operations.")

    if transform is None:
        transform = get_image_transforms(is_train=(split == "train"))

    if manifest_df is None:
        manifest_df = scan_cctv_manifest(verify_images=False)

    target_col = split_col if split_col in manifest_df.columns else "split"
    df_split = manifest_df[manifest_df[target_col] == split].reset_index(drop=True)

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
    split_col: str = "split",
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

    train_ds = load_cctv_dataset("train", transform=get_image_transforms(is_train=True), manifest_df=manifest_df, split_col=split_col)
    val_ds = load_cctv_dataset("val", transform=get_image_transforms(is_train=False), manifest_df=manifest_df, split_col=split_col)
    test_ds = load_cctv_dataset("test", transform=get_image_transforms(is_train=False), manifest_df=manifest_df, split_col=split_col)

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
