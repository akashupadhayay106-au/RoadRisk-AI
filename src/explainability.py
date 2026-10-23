"""
RoadRisk AI — Explainable AI (XAI) & Model Interpretability Module
===================================================================
Module J: Computes global feature importance, group-aggregated importance
scores, local prediction factor attributions, and enforces rigorous causal
governance ("Feature Importance ≠ Causality").
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_MODEL_PATH = MODELS_DIR / "severity_pipeline.joblib"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

_CACHED_IMPORTANCE: Optional[pd.DataFrame] = None


def load_severity_artifact(model_path: Path = SEVERITY_MODEL_PATH) -> Dict[str, Any]:
    """Load the serialized severity classification pipeline dictionary."""
    if not model_path.exists():
        raise FileNotFoundError(f"Severity model artifact not found at {model_path}")
    return joblib.load(model_path)


def get_global_feature_importance(
    artifact: Optional[Dict[str, Any]] = None,
    save_summary: bool = True,
) -> pd.DataFrame:
    """
    Extract feature importances from the trained RandomForestClassifier,
    aggregate one-hot encoded categories back to their parent features,
    and return a ranked DataFrame.
    """
    global _CACHED_IMPORTANCE
    if _CACHED_IMPORTANCE is not None and artifact is None:
        return _CACHED_IMPORTANCE

    if artifact is None:
        artifact = load_severity_artifact()

    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Raw encoded feature names and importances
    raw_feature_names = preprocessor.get_feature_names_out()
    raw_importances = classifier.feature_importances_

    # Map one-hot encoded features back to their original root column
    grouped_importance = {}
    for feat_name, imp in zip(raw_feature_names, raw_importances):
        # Format usually: 'cat__ColumnName_CategoryValue' or 'num__ColumnName'
        parts = feat_name.split("__")
        clean_name = parts[1] if len(parts) > 1 else feat_name
        # If categorical, extract root column before first underscore matching original features
        root_col = clean_name
        for orig_col in artifact.get("features", []):
            if clean_name.startswith(orig_col):
                root_col = orig_col
                break

        grouped_importance[root_col] = grouped_importance.get(root_col, 0.0) + imp

    # Convert to DataFrame
    df_imp = pd.DataFrame([
        {"feature": k, "importance": v} for k, v in grouped_importance.items()
    ])
    total_imp = df_imp["importance"].sum()
    df_imp["importance_pct"] = (df_imp["importance"] / total_imp) * 100
    df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
    df_imp["cumulative_pct"] = df_imp["importance_pct"].cumsum()
    df_imp["rank"] = df_imp.index + 1

    if save_summary:
        try:
            records = df_imp.to_dict(orient="records")
            with open(FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            logger.info(f"Saved feature importance artifact to {FEATURE_IMPORTANCE_PATH}")
        except Exception as e:
            logger.warning(f"Could not persist feature importance: {e}")

    _CACHED_IMPORTANCE = df_imp
    return df_imp


def explain_single_prediction(
    input_features: Dict[str, Any],
    artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate local risk explanation for an individual incident input.
    Computes class probabilities and identifies high-influence risk indicators.
    """
    if artifact is None:
        artifact = load_severity_artifact()

    pipeline = artifact["pipeline"]
    classes = artifact["classes"]

    # Convert input dict to 1-row DataFrame
    df_input = pd.DataFrame([input_features])

    # Predict probabilities
    probs = pipeline.predict_proba(df_input)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]

    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

    # High-impact attribute heuristic
    risk_factors = []
    protective_factors = []

    # Check key features
    if input_features.get("Light_conditions") in ["Darkness - no lighting", "Darkness - lights unlit"]:
        risk_factors.append("Unlit night driving significantly impairs reaction distance and hazard perception.")
    elif input_features.get("Light_conditions") == "Daylight":
        protective_factors.append("Full daylight provides optimal driver visibility.")

    if input_features.get("Weather_conditions") in ["Raining", "Fog or mist"]:
        risk_factors.append("Adverse weather creates slick asphalt and reduces braking friction coefficient.")

    if input_features.get("Driving_experience") in ["Under 1yr", "1-2yr"]:
        risk_factors.append("Novice driving experience is correlated with reduced evasive maneuvering capability.")
    elif input_features.get("Driving_experience") == "Above 10yr":
        protective_factors.append("Experienced driver tenure provides defensive reaction habits.")

    if input_features.get("Lanes_or_Medians") in ["Undivided Two way", "One way"]:
        risk_factors.append("Undivided lanes increase exposure to opposing vehicle head-on trajectory conflicts.")

    return {
        "predicted_class": pred_class,
        "probabilities": prob_dict,
        "primary_risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "causal_disclaimer": (
            "Feature importance reflects statistical correlation in police incident logs, "
            "not a definitive mechanical cause of crash severity."
        ),
    }


def get_causal_disclaimer() -> str:
    """Return explicit ethical and causal governance statement."""
    return (
        "⚠️ **Feature Importance ≠ Causality:**\n"
        "Machine learning feature importance metrics (e.g., Gini impurity decrease or Permutation Importance) "
        "measure how predictive a variable is inside the mathematical model. They do NOT establish that changing "
        "that variable will cause an accident outcome to change in the physical world. Confounders, reporting "
        "biases, and latent infrastructural variables heavily influence these statistical correlations."
    )


if __name__ == "__main__":
    print("Testing Explainability Module...")
    df_imp = get_global_feature_importance()
    print("Top 5 Influential Features:")
    print(df_imp.head(5)[["rank", "feature", "importance_pct", "cumulative_pct"]])
