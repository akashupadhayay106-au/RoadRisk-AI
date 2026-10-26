"""
RoadRisk AI — Central Configuration
====================================
All paths, constants, and settings for the project.
Uses pathlib for cross-platform compatibility.
"""

from pathlib import Path


# ── Project Root ──────────────────────────────────────────────────────────────
# This file lives at src/config.py → project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Local .env loading ────────────────────────────────────────────────────────
# Loads a developer's local .env if present. On Streamlit Community Cloud there
# is no .env; credentials come from st.secrets instead. Values already set in the
# real environment always win (override=False).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:  # python-dotenv optional / .env absent
    pass

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR         = PROJECT_ROOT / "data"
RAW_DATA_DIR     = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"
SAMPLE_DIR       = DATA_DIR / "sample"

# CCTV image directories (may be directly under data/ based on download)
CCTV_TRAIN_DIR   = DATA_DIR / "train"
CCTV_TEST_DIR    = DATA_DIR / "test"
CCTV_VAL_DIR     = DATA_DIR / "val"

# ── Model Paths ──────────────────────────────────────────────────────────────
MODELS_DIR       = PROJECT_ROOT / "models"

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_PATH    = PROCESSED_DIR / "roadrisk.db"

# ── Raw Dataset Filenames (will be validated in Phase 2) ─────────────────────
INDIAN_ACCIDENTS_FILE       = RAW_DATA_DIR / "indian_roads_dataset.csv"
SEVERITY_TRAIN_FILE         = RAW_DATA_DIR / "train.csv"
SEVERITY_TEST_FILE          = RAW_DATA_DIR / "test.csv"
OPENCITY_FILE               = RAW_DATA_DIR / "Accdataset_hk_PS_BAEL_Combined.csv"
URBAN_GB_COORDS_FILE        = RAW_DATA_DIR / "urbanGB.txt"
URBAN_GB_LABELS_FILE        = RAW_DATA_DIR / "urbanGB.labels.txt"
URBAN_GB_CENTROIDS_FILE     = RAW_DATA_DIR / "urbanGB.centr.txt"

# ── App Metadata ─────────────────────────────────────────────────────────────
APP_TITLE       = "🚦 RoadRisk AI"
APP_SUBTITLE    = "AI-Powered Road Accident Analysis & Risk Intelligence"
APP_VERSION     = "0.1.0"
APP_AUTHOR      = "RoadRisk AI Team"

# ── Page Configuration ────────────────────────────────────────────────────────
PAGE_CONFIG = {
    "page_title": "RoadRisk AI",
    "page_icon": "🚦",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# ── Color Palette (road-safety themed) ───────────────────────────────────────
COLORS = {
    "primary":       "#FF4B4B",   # Alert red
    "secondary":     "#FF8C42",   # Warning orange
    "accent":        "#00D4AA",   # Safe green-teal
    "info":          "#4DA8FF",   # Info blue
    "dark_bg":       "#0E1117",   # Background
    "card_bg":       "#1A1F2C",   # Card background
    "text":          "#FAFAFA",   # Primary text
    "text_muted":    "#8899A6",   # Muted text
    "success":       "#00C853",   # Success green
    "warning":       "#FFD600",   # Warning yellow
    "danger":        "#FF1744",   # Danger red
    "chart_palette": [
        "#FF4B4B", "#FF8C42", "#FFD600", "#00D4AA",
        "#4DA8FF", "#A855F7", "#EC4899", "#14B8A6",
    ],
}

# ── Dataset Metadata ─────────────────────────────────────────────────────────
DATASETS = {
    "indian_accidents": {
        "name": "Indian Road Accident Dataset (2022–2025)",
        "source": "Kaggle (sehaj1104)",
        "url": "https://www.kaggle.com/datasets/sehaj1104/indian-road-accident-dataset-20222025",
        "license": "Check Kaggle page",
        "description": "Indian road accident records with state, date, vehicle, and casualty info.",
    },
    "severity": {
        "name": "Road Traffic Accident Severity",
        "source": "Kaggle Competition",
        "url": "https://www.kaggle.com/competitions/road-traffic-accidents/data",
        "license": "Competition rules apply",
        "description": "Labeled dataset for accident severity classification (train/test split).",
    },
    "urban_gb": {
        "name": "UrbanGB Urban Road Accidents Coordinates",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/550",
        "license": "Open Government Licence (UK) + ODbL",
        "description": "360K+ Great Britain urban accident coordinates with urban-center labels.",
    },
    "opencity_india": {
        "name": "Road Accidents in India 2023",
        "source": "OpenCity.in",
        "url": "https://data.opencity.in/dataset/road-accidents-in-india-2023",
        "license": "Check OpenCity page",
        "description": "Indian road accident statistics for 2023.",
    },
    "cctv": {
        "name": "Accident Detection from CCTV Footage",
        "source": "Kaggle (ckay16)",
        "url": "https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage",
        "license": "Check Kaggle page",
        "description": "Image frames labeled Accident / Non-Accident for binary classification.",
    },
}


def ensure_directories() -> None:
    """Create required project directories if they don't exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Retrieve a secret from Streamlit secrets (deployment) or env vars (local).
    Returns *default* if neither source has the key.
    """
    import os

    # Try Streamlit secrets first (works on Streamlit Cloud)
    try:
        import streamlit as st
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:
        pass

    # Fall back to environment variable
    return os.environ.get(key, default)
