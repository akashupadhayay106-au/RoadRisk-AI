"""
RoadRisk AI — Accident Severity Model Training Pipeline
=======================================================
Trains baseline (Logistic Regression) and advanced (Random Forest, XGBoost)
classifiers with class imbalance handling and leakage prevention.
Saves the optimal pipeline and metrics artifact into models/.
"""

import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR, MODELS_DIR, ensure_directories
from src.features import build_preprocessor, prepare_training_data
from src.evaluate_model import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SeverityPredictor:
    """Portable wrapper for predicting crash severity and probabilities."""
    def __init__(self, bundle: Dict[str, Any]):
        self.pipeline = bundle["pipeline"]
        self.label_encoder = bundle["label_encoder"]
        self.classes_ = np.array(bundle["classes"])
        self.feature_names = bundle["features"]
        self.best_model_name = bundle.get("best_model_name", "Random Forest")

    def predict(self, X_input: pd.DataFrame) -> np.ndarray:
        preds_encoded = self.pipeline.predict(X_input)
        return self.label_encoder.inverse_transform(preds_encoded)

    def predict_proba(self, X_input: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X_input)


# Backwards compatibility alias
SeverityInferencePipeline = SeverityPredictor


def load_severity_model(model_path: Path | str | None = None) -> SeverityPredictor:
    """Load the trained severity pipeline bundle from disk."""
    path = Path(model_path or (MODELS_DIR / "severity_pipeline.joblib"))
    if not path.exists():
        logger.info(f"Model artifact not found at {path}. Triggering training...")
        train_and_compare_models()
    bundle = joblib.load(path)
    # Support both dict bundle and raw object
    if isinstance(bundle, dict):
        return SeverityPredictor(bundle)
    elif hasattr(bundle, "pipeline"):
        return bundle
    else:
        raise ValueError("Unrecognized model artifact format.")


def train_and_compare_models(
    csv_path: Path | None = None,
    save_artifacts: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train Logistic Regression, Random Forest, and XGBoost models.
    Compares macro-F1 on holdout validation data and saves the best pipeline.
    """
    ensure_directories()
    src_csv = csv_path or (PROCESSED_DIR / "severity_train_cleaned.csv")
    if not src_csv.exists():
        from src.data_cleaning import clean_severity_ml_dataset
        df_train, _ = clean_severity_ml_dataset(split="train")
    else:
        df_train = pd.read_csv(src_csv)

    logger.info(f"Loaded training data: {df_train.shape[0]:,} rows × {df_train.shape[1]} columns")

    # 1. Feature & Target Extraction with Leakage Prevention
    X, y, cat_features, num_features = prepare_training_data(df_train, exclude_leakage=True)
    logger.info(f"Features: {len(cat_features)} categorical, {len(num_features)} numerical. (Outcome features excluded)")

    # 2. Encode string targets for compatibility with all models
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    logger.info(f"Target classes: {class_names}")

    # 3. Stratified Train / Validation Split (80 / 20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )
    logger.info(f"Train set: {len(X_train):,} samples | Validation set: {len(X_val):,} samples")

    # 4. Build Preprocessor
    preprocessor = build_preprocessor(cat_features, num_features)

    # 5. Define Candidate Models
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        logger.info(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        # Decode for metrics
        y_val_orig = label_encoder.inverse_transform(y_val)
        y_pred_orig = label_encoder.inverse_transform(y_pred)

        metrics = compute_classification_metrics(y_val_orig, y_pred_orig, labels=class_names)
        results[name] = metrics
        fitted_pipelines[name] = pipeline

        logger.info(
            f"{name} -> Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | Weighted-F1: {metrics['weighted_f1']:.4f}"
        )

    # 6. Model Selection based on Macro-F1 (critical for class-imbalance)
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_pipeline = fitted_pipelines[best_name]
    logger.info(f"Selected Optimal Model: '{best_name}' (Macro-F1: {results[best_name]['macro_f1']:.4f})")

    # 7. Save Portable Dictionary Bundle
    if save_artifacts:
        pipeline_file = MODELS_DIR / "severity_pipeline.joblib"
        metrics_file = MODELS_DIR / "severity_model_comparison.json"

        bundle = {
            "pipeline": best_pipeline,
            "label_encoder": label_encoder,
            "classes": class_names,
            "features": cat_features + num_features,
            "best_model_name": best_name,
        }
        joblib.dump(bundle, pipeline_file)
        logger.info(f"Saved inference pipeline bundle to {pipeline_file}")

        comparison_payload = {
            "best_model": best_name,
            "classes": class_names,
            "feature_count": len(cat_features) + len(num_features),
            "categorical_features": cat_features,
            "numerical_features": num_features,
            "comparison": results,
        }
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2)
        logger.info(f"Saved model comparison metrics to {metrics_file}")

    return best_pipeline, results


if __name__ == "__main__":
    train_and_compare_models()
