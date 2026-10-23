"""
RoadRisk AI — Phase 5 Tests: Severity ML Pipeline
=================================================
Tests verifying feature isolation, leakage exclusion, preprocessor transformations,
metrics computation, and serialized model inference.
"""

import json
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DIR
from src.features import (
    prepare_training_data,
    build_preprocessor,
    LEAKAGE_FEATURES,
    get_feature_lists,
)
from src.evaluate_model import compute_classification_metrics
from src.train_severity_model import SeverityInferencePipeline


class TestFeatureEngineering:
    """Validate feature extraction and leakage prevention."""

    @pytest.fixture(scope="class")
    @staticmethod
    def sample_train_df():
        csv_path = PROCESSED_DIR / "severity_train_cleaned.csv"
        return pd.read_csv(csv_path, nrows=200)

    def test_leakage_features_excluded(self, sample_train_df):
        X, y, cat_cols, num_cols = prepare_training_data(sample_train_df, exclude_leakage=True)
        for leak_col in LEAKAGE_FEATURES:
            assert leak_col not in X.columns, f"Leakage column {leak_col} was not excluded!"

    def test_target_separation(self, sample_train_df):
        X, y, _, _ = prepare_training_data(sample_train_df)
        assert "Accident_severity" not in X.columns
        assert len(y) == len(sample_train_df)

    def test_preprocessor_transformation(self, sample_train_df):
        X, y, cat_cols, num_cols = prepare_training_data(sample_train_df)
        preprocessor = build_preprocessor(cat_cols, num_cols)
        X_trans = preprocessor.fit_transform(X)
        assert X_trans.shape[0] == len(sample_train_df)
        assert X_trans.shape[1] > len(cat_cols)  # One-hot expansion


class TestModelEvaluationMetrics:
    """Validate metrics computation logic."""

    def test_metrics_calculation(self):
        y_true = np.array(["Slight Injury", "Serious Injury", "Fatal injury", "Slight Injury"])
        y_pred = np.array(["Slight Injury", "Serious Injury", "Slight Injury", "Slight Injury"])

        metrics = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["macro_f1"] <= 1.0


class TestModelInference:
    """Validate saved model pipeline inference."""

    def test_saved_model_artifact_exists(self):
        pipeline_file = MODELS_DIR / "severity_pipeline.joblib"
        assert pipeline_file.exists()

    def test_model_prediction_and_proba(self):
        from src.train_severity_model import load_severity_model
        pipeline_file = MODELS_DIR / "severity_pipeline.joblib"
        model = load_severity_model(pipeline_file)

        # Create dummy input with valid features
        sample_input = {
            "Day_of_week": "Friday",
            "Age_band_of_driver": "18-30",
            "Sex_of_driver": "Male",
            "Educational_level": "High school",
            "Vehicle_driver_relation": "Employee",
            "Driving_experience": "2-5yr",
            "Type_of_vehicle": "Automobile",
            "Owner_of_vehicle": "Owner",
            "Service_year_of_vehicle": "Above 10yr",
            "Defect_of_vehicle": "No defect",
            "Area_accident_occured": "Office areas",
            "Lanes_or_Medians": "Two-way (divided with broken lines road marking)",
            "Road_allignment": "Tangent road with flat terrain",
            "Types_of_Junction": "Y Shape",
            "Road_surface_type": "Asphalt roads",
            "Road_surface_conditions": "Dry",
            "Light_conditions": "Daylight",
            "Weather_conditions": "Normal",
            "Type_of_collision": "Vehicle with vehicle collision",
            "Vehicle_movement": "Going straight",
            "Cause_of_accident": "Driving carelessly",
            "Time_of_Day": "Evening",
            "Number_of_vehicles_involved": 2,
            "Number_of_casualties": 1,
            "Hour": 18,
        }
        df_input = pd.DataFrame([sample_input])

        pred = model.predict(df_input)
        assert len(pred) == 1
        assert pred[0] in model.classes_

        probs = model.predict_proba(df_input)
        assert probs.shape == (1, len(model.classes_))
        assert np.isclose(probs.sum(), 1.0, atol=1e-3)
