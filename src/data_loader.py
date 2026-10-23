"""
RoadRisk AI — Data Loader Module
=================================
Robust loading functions for all verified project datasets.
Handles delimiters, types, and path resolution across raw/ and processed/ dirs.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    CCTV_TRAIN_DIR,
    CCTV_TEST_DIR,
    CCTV_VAL_DIR,
)


def find_dataset_file(filename: str) -> Path | None:
    """Search for a dataset file across raw/, processed/, and root directories."""
    candidates = [
        RAW_DATA_DIR / filename,
        PROCESSED_DIR / filename,
        PROJECT_ROOT / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# Backwards compatibility alias
find_dataset = find_dataset_file


@st.cache_data(show_spinner="Loading Indian Road Accidents dataset…")
def load_indian_roads_dataset(nrows: int | None = None) -> pd.DataFrame:
    """
    Load the Indian Road Accident Dataset (2022–2025).
    20,000 records across 8 major Indian cities.
    """
    path = find_dataset_file("indian_roads_dataset.csv")
    if not path:
        raise FileNotFoundError("Could not find 'indian_roads_dataset.csv' in data/raw/ or root.")
    
    df = pd.read_csv(path, nrows=nrows)
    # Parse dates if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading Severity training set…")
def load_severity_train_dataset(nrows: int | None = None) -> pd.DataFrame:
    """
    Load the Road Traffic Accident Severity training dataset.
    8,210 rows × 33 columns with target 'Accident_severity'.
    """
    path = find_dataset_file("train.csv")
    if not path:
        raise FileNotFoundError("Could not find 'train.csv' in data/raw/ or root.")
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner="Loading Severity test set…")
def load_severity_test_dataset(nrows: int | None = None) -> pd.DataFrame:
    """Load the Road Traffic Accident Severity test dataset (4,106 rows)."""
    path = find_dataset_file("test.csv")
    if not path:
        raise FileNotFoundError("Could not find 'test.csv' in data/raw/ or root.")
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner="Loading UrbanGB Great Britain coordinates…")
def load_urban_gb_dataset(nrows: int | None = None, include_labels: bool = True) -> pd.DataFrame:
    """
    Load the UrbanGB (Great Britain) accident coordinates and cluster labels.
    Uses comma delimiter (verified during Phase 2 inspection).
    Warning: This dataset belongs strictly to Great Britain, not India.
    """
    coords_path = find_dataset_file("urbanGB.txt")
    if not coords_path:
        raise FileNotFoundError("Could not find 'urbanGB.txt' in data/raw/ or root.")
    
    df = pd.read_csv(coords_path, sep=",", header=None, names=["longitude", "latitude"], nrows=nrows)
    
    if include_labels:
        labels_path = find_dataset_file("urbanGB.labels.txt")
        if labels_path and labels_path.exists():
            labels = pd.read_csv(labels_path, header=None, names=["cluster_id"], nrows=nrows)
            df["cluster_id"] = labels["cluster_id"]
    return df


@st.cache_data(show_spinner="Loading OpenCity India 2023 dataset…")
def load_opencity_dataset(nrows: int | None = None) -> pd.DataFrame:
    """Load the OpenCity India 2023 highway accident records."""
    path = find_dataset_file("Accdataset_hk_PS_BAEL_Combined.csv")
    if not path:
        raise FileNotFoundError("Could not find 'Accdataset_hk_PS_BAEL_Combined.csv' in data/raw/ or root.")
    return pd.read_csv(path, nrows=nrows)


def get_cctv_image_manifest() -> pd.DataFrame:
    """
    Scan data/train, data/test, and data/val to construct an image manifest.
    Returns a DataFrame with columns: ['filepath', 'split', 'class_label', 'filename'].
    """
    records = []
    splits = {
        "train": CCTV_TRAIN_DIR,
        "test": CCTV_TEST_DIR,
        "val": CCTV_VAL_DIR,
    }
    for split_name, split_path in splits.items():
        if not split_path.exists():
            continue
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                for img_file in class_dir.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        records.append({
                            "filepath": str(img_file),
                            "split": split_name,
                            "class_label": label,
                            "filename": img_file.name,
                        })
    return pd.DataFrame(records)
