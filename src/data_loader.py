"""
RoadRisk AI — Data Loader
==========================
Functions for loading and caching datasets from raw/ and processed/ directories.
Full implementation in Phase 2.
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DIR, PROJECT_ROOT


def find_dataset(filename: str) -> Path | None:
    """Search for a file in raw/, processed/, then project root."""
    for search_dir in [RAW_DATA_DIR, PROCESSED_DIR, PROJECT_ROOT]:
        candidate = search_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_csv(filename: str, **kwargs) -> pd.DataFrame | None:
    """Load a CSV file by name, searching standard directories."""
    path = find_dataset(filename)
    if path is None:
        return None
    return pd.read_csv(path, **kwargs)


def load_indian_accidents() -> pd.DataFrame | None:
    """Load the Indian Road Accident dataset."""
    return load_csv("indian_roads_dataset.csv")


def load_severity_train() -> pd.DataFrame | None:
    """Load the severity classification training set."""
    return load_csv("train.csv")


def load_severity_test() -> pd.DataFrame | None:
    """Load the severity classification test set."""
    return load_csv("test.csv")


def load_opencity() -> pd.DataFrame | None:
    """Load the OpenCity India 2023 dataset."""
    return load_csv("Accdataset_hk_PS_BAEL_Combined.csv")


def load_urban_gb_coords() -> pd.DataFrame | None:
    """Load UrbanGB coordinates (lon, lat) from text file."""
    path = find_dataset("urbanGB.txt")
    if path is None:
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=["longitude", "latitude"])


def load_urban_gb_labels() -> pd.DataFrame | None:
    """Load UrbanGB cluster labels."""
    path = find_dataset("urbanGB.labels.txt")
    if path is None:
        return None
    return pd.read_csv(path, header=None, names=["cluster_label"])
