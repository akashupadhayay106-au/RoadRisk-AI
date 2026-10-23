"""
RoadRisk AI — Data Validation Module
=====================================
Comprehensive data validation rules, bounds checks, schema verifications,
and target leakage inspection for road safety datasets.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


# ── Geographic Coordinate Bounding Boxes ─────────────────────────────────────
# India approximate bounding box
INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 38.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 98.0

# Great Britain (UK) approximate bounding box
UK_LAT_MIN, UK_LAT_MAX = 49.0, 61.0
UK_LON_MIN, UK_LON_MAX = -9.0, 3.0


def validate_coordinates(lat: float, lon: float, region: str = "global") -> bool:
    """
    Validate that latitude and longitude fall within acceptable ranges.
    region can be 'global', 'india', or 'uk'.
    """
    if pd.isna(lat) or pd.isna(lon):
        return False
    
    # Global sanity check
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    
    if region.lower() == "india":
        return INDIA_LAT_MIN <= lat <= INDIA_LAT_MAX and INDIA_LON_MIN <= lon <= INDIA_LON_MAX
    elif region.lower() == "uk" or region.lower() == "gb":
        return UK_LAT_MIN <= lat <= UK_LAT_MAX and UK_LON_MIN <= lon <= UK_LON_MAX
    
    return True


def check_dataframe_coordinates(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    region: str = "global"
) -> Dict[str, Any]:
    """
    Audit coordinates in a dataframe and return percentage of valid entries.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        return {"has_coordinates": False, "valid_count": 0, "invalid_count": 0, "valid_pct": 0.0}
    
    valid_mask = df.apply(lambda row: validate_coordinates(row[lat_col], row[lon_col], region), axis=1)
    valid_count = int(valid_mask.sum())
    invalid_count = len(df) - valid_count
    
    return {
        "has_coordinates": True,
        "total_records": len(df),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_pct": round((valid_count / len(df)) * 100, 2) if len(df) > 0 else 0.0,
        "lat_min": float(df[lat_col].min()),
        "lat_max": float(df[lat_col].max()),
        "lon_min": float(df[lon_col].min()),
        "lon_max": float(df[lon_col].max()),
    }


def generate_data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Generate a complete quality audit for any dataset:
    dimensions, duplicate rows, missing values per column, and memory usage.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    dup_count = int(df.duplicated().sum())
    
    missing_summary = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        missing_summary.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
            "unique_values": int(df[col].nunique(dropna=True)),
        })
    
    return {
        "name": name,
        "rows": total_rows,
        "columns": total_cols,
        "duplicate_rows": dup_count,
        "duplicate_pct": round((dup_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
        "column_report": missing_summary,
    }


def detect_target_leakage_features(
    columns: List[str],
    target_column: str = "Accident_severity",
    leakage_keywords: List[str] | None = None
) -> List[str]:
    """
    Flag features that could introduce post-outcome data leakage.
    For example: Casualty_severity, death_count, post-accident hospital records.
    """
    if leakage_keywords is None:
        leakage_keywords = [
            "casualty_severity", "injury_severity", "fatal_count",
            "hospital", "death", "post_crash", "treatment"
        ]
    
    suspicious = []
    for col in columns:
        if col.lower() == target_column.lower():
            continue
        col_clean = col.lower().replace(" ", "_")
        if any(term in col_clean for term in leakage_keywords):
            suspicious.append(col)
    
    return suspicious


def validate_date_string(date_str: str, fmt: str = "%Y-%m-%d") -> bool:
    """Validate whether a string parses as a valid date."""
    from datetime import datetime
    try:
        datetime.strptime(str(date_str), fmt)
        return True
    except (ValueError, TypeError):
        return False
