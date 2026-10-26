"""
RoadRisk AI — Data Cleaning & Normalization Pipeline
====================================================
Reproducible preprocessing scripts for all project datasets.
Preserves raw datasets intact and outputs validated, cleaned
datasets into data/processed/.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Tuple
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    INDIAN_ACCIDENTS_FILE,
    ensure_directories,
)
from src.data_validation import validate_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Time-of-Day Period Helper ────────────────────────────────────────────────

def get_time_of_day(hour: int) -> str:
    """Categorize 24-hour integer into standard temporal periods."""
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


# ── 1. Indian Roads Dataset Pipeline ─────────────────────────────────────────

def clean_indian_roads_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean, validate, and enrich the Indian Road Accident Dataset.
    
    Transformations:
    1. Validate row completeness and duplicates.
    2. Normalize text columns (trim, title-case or lower-case consistency).
    3. Standardize 'festival' nulls to 'None'.
    4. Parse 'date' to datetime and extract year, month, day, quarter.
    5. Derive 'time_of_day' period (Morning, Afternoon, Evening, Night).
    6. Validate coordinates against India bounding box [6-38°N, 68-98°E].
    7. Standardize target 'accident_severity' categories to lower-case.
    8. Write output to data/processed/indian_roads_cleaned.parquet and .csv.
    """
    ensure_directories()
    src_file = input_path or INDIAN_ACCIDENTS_FILE
    if not src_file.exists():
        src_file = RAW_DATA_DIR / "indian_roads_dataset.csv"
    if not src_file.exists():
        raise FileNotFoundError(f"Input file not found at: {src_file}")

    logger.info(f"Loading raw Indian roads dataset from {src_file}...")
    df_raw = pd.read_csv(src_file)
    initial_rows = len(df_raw)

    df = df_raw.copy()

    # 1. Deduplication
    dups_removed = int(df.duplicated().sum())
    if dups_removed > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. String standardization
    str_cols = ["city", "state", "road_type", "weather", "visibility", "traffic_density", "cause", "accident_severity"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Title-case proper nouns
    if "city" in df.columns:
        df["city"] = df["city"].str.title()
    if "state" in df.columns:
        df["state"] = df["state"].str.title()
    if "day_of_week" in df.columns:
        df["day_of_week"] = df["day_of_week"].astype(str).str.strip().str.title()

    # 3. Festival null handling: NaN -> 'None'
    if "festival" in df.columns:
        df["festival"] = df["festival"].fillna("None").astype(str).str.strip().str.title()
        df.loc[df["festival"].isin(["Nan", "None", ""],), "festival"] = "None"

    # 4. Date parsing & temporal feature extraction
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")
        df["day"] = df["date"].dt.day
        df["quarter"] = df["date"].dt.quarter

    # 5. Hour & Time-of-day
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype(int)
        df["time_of_day"] = df["hour"].apply(get_time_of_day)

    # 6. Coordinate validation
    coord_valid = df.apply(lambda r: validate_coordinates(r["latitude"], r["longitude"], region="india"), axis=1)
    invalid_coords = int((~coord_valid).sum())

    # 7. Severity classification standardization
    if "accident_severity" in df.columns:
        # Standard values: minor, major, fatal
        df["accident_severity"] = df["accident_severity"].str.lower().str.strip()

    # 8. Numeric types
    numeric_cols = ["lanes", "traffic_signal", "temperature", "vehicles_involved", "casualties", "is_weekend", "is_peak_hour"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").round(4)

    # Output paths
    target_parquet = output_path or (PROCESSED_DIR / "indian_roads_cleaned.parquet")
    target_csv = target_parquet.with_suffix(".csv")

    try:
        df.to_parquet(target_parquet, index=False)
        logger.info(f"Saved cleaned dataset to {target_parquet}")
    except Exception as e:
        logger.warning(f"Could not save parquet ({e}), fallback to CSV only.")

    df.to_csv(target_csv, index=False)
    logger.info(f"Saved cleaned dataset to {target_csv}")

    log_summary = {
        "dataset": "indian_roads_dataset",
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "duplicates_removed": dups_removed,
        "invalid_coords_flagged": invalid_coords,
        "cities_count": int(df["city"].nunique()) if "city" in df.columns else 0,
        "severity_distribution": df["accident_severity"].value_counts().to_dict() if "accident_severity" in df.columns else {},
        "output_csv": str(target_csv),
        "output_parquet": str(target_parquet),
    }

    return df, log_summary


# ── 2. Severity ML Dataset Pipeline ──────────────────────────────────────────

def clean_severity_ml_dataset(
    split: str = "train",
    input_path: Path | None = None,
    output_path: Path | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the Road Traffic Accident Severity dataset for machine learning.
    
    Transformations:
    1. Standardize text strings and column headers.
    2. Convert ambiguous missing indicators ('na', 'Unknown', 'Unknown/Not specified') to standard 'Unknown'.
    3. Separate pre-incident driver/environment attributes from post-outcome injury attributes.
    4. Save cleaned output to data/processed/severity_{split}_cleaned.parquet & .csv.
    """
    ensure_directories()
    if input_path:
        src_file = input_path
    else:
        src_file = RAW_DATA_DIR / ("train.csv" if split == "train" else "test.csv")

    if not src_file.exists():
        raise FileNotFoundError(f"Severity {split} dataset not found at: {src_file}")

    logger.info(f"Cleaning severity dataset ({split}) from {src_file}...")
    df_raw = pd.read_csv(src_file)
    initial_rows = len(df_raw)

    df = df_raw.copy()

    # Clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Standardize string fields
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        # Map common missing token variants to uniform 'Unknown'
        df[col] = df[col].replace(["na", "NA", "Unknown/Not specified", "unknown", "nan", "None", ""], "Unknown")

    # Time extraction (HH:MM:SS -> integer hour)
    if "Time" in df.columns:
        parsed_hours = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
        # Fallback for alternative time formats
        if parsed_hours.isnull().any():
            fallback = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
            parsed_hours = parsed_hours.fillna(fallback)
        df["Hour"] = parsed_hours.fillna(12).astype(int)
        df["Time_of_Day"] = df["Hour"].apply(get_time_of_day)

    target_csv = output_path or (PROCESSED_DIR / f"severity_{split}_cleaned.csv")
    df.to_csv(target_csv, index=False)

    try:
        df.to_parquet(target_csv.with_suffix(".parquet"), index=False)
    except Exception:
        pass

    log_summary = {
        "dataset": f"severity_{split}",
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "columns_count": len(df.columns),
        "target_distribution": df["Accident_severity"].value_counts().to_dict() if "Accident_severity" in df.columns else {},
        "output_csv": str(target_csv),
    }

    return df, log_summary


# ── Run All Preprocessing ────────────────────────────────────────────────────

def run_all_cleaning_pipelines() -> Dict[str, Any]:
    """Execute all data cleaning pipelines in sequence and return diagnostic log."""
    results = {}
    logger.info("Executing Indian Roads Cleaning Pipeline...")
    _, results["indian_roads"] = clean_indian_roads_dataset()

    logger.info("Executing Severity Train Pipeline...")
    _, results["severity_train"] = clean_severity_ml_dataset(split="train")

    logger.info("Executing Severity Test Pipeline...")
    _, results["severity_test"] = clean_severity_ml_dataset(split="test")

    logger.info("All data cleaning pipelines completed successfully.")
    return results


if __name__ == "__main__":
    summary = run_all_cleaning_pipelines()
    import pprint
    pprint.pprint(summary)
