"""
RoadRisk AI — Feature Engineering & Preprocessing Pipeline
==========================================================
Constructs Scikit-learn preprocessing pipelines, handles categorical
encoding, numerical scaling, and implements data leakage prevention
by isolating post-crash outcome attributes.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ── Feature Sets for Road Traffic Accident Severity ──────────────────────────

# Post-outcome / target-leakage features that MUST be excluded from pre-crash models
LEAKAGE_FEATURES = [
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Casualty_severity",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
]

# Primary predictors available at or before the moment of impact
PRE_CRASH_CATEGORICAL_FEATURES = [
    "Day_of_week",
    "Age_band_of_driver",
    "Sex_of_driver",
    "Educational_level",
    "Vehicle_driver_relation",
    "Driving_experience",
    "Type_of_vehicle",
    "Owner_of_vehicle",
    "Service_year_of_vehicle",
    "Defect_of_vehicle",
    "Area_accident_occured",
    "Lanes_or_Medians",
    "Road_allignment",
    "Types_of_Junction",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Vehicle_movement",
    "Cause_of_accident",
    "Time_of_Day",
]

PRE_CRASH_NUMERIC_FEATURES = [
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "Hour",
]

TARGET_COLUMN = "Accident_severity"


def get_feature_lists(
    include_all: bool = False
) -> Tuple[List[str], List[str], str]:
    """
    Return lists of categorical features, numeric features, and the target column.
    If include_all=False, strictly omits post-outcome casualty leakage columns.
    """
    cat_features = PRE_CRASH_CATEGORICAL_FEATURES.copy()
    num_features = PRE_CRASH_NUMERIC_FEATURES.copy()
    return cat_features, num_features, TARGET_COLUMN


def build_preprocessor(
    categorical_features: List[str],
    numeric_features: List[str]
) -> ColumnTransformer:
    """
    Build a Scikit-learn ColumnTransformer for categorical and numerical features.
    Categorical: SimpleImputer (most_frequent) -> OneHotEncoder (handle_unknown='ignore')
    Numerical: SimpleImputer (median) -> StandardScaler()
    """
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_training_data(
    df: pd.DataFrame,
    exclude_leakage: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separate feature matrix X and target vector y from the dataset.
    Excludes identifiers (like 'Num') and outcome leakage features if specified.
    """
    df_clean = df.copy()

    # Drop identifiers if present
    if "Num" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Num"])

    # Separate target
    if TARGET_COLUMN not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    y = df_clean[TARGET_COLUMN].astype(str)

    # Exclude leakage columns
    cols_to_drop = [TARGET_COLUMN]
    if exclude_leakage:
        for leak_col in LEAKAGE_FEATURES:
            if leak_col in df_clean.columns:
                cols_to_drop.append(leak_col)

    X = df_clean.drop(columns=cols_to_drop, errors="ignore")

    # Filter to available features
    cat_features = [c for c in PRE_CRASH_CATEGORICAL_FEATURES if c in X.columns]
    num_features = [c for c in PRE_CRASH_NUMERIC_FEATURES if c in X.columns]

    return X[cat_features + num_features], y, cat_features, num_features
