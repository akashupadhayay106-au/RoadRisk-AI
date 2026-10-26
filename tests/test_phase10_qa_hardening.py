"""
Phase 10 Unit Tests — Comprehensive QA, Explainability & Security Hardening
===========================================================================
Audits end-to-end data integrity, edge cases, malformed data, coordinate bounds,
leakage screening, explainability attribution, SQL security, and secret handling.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd

from src.data_validation import validate_coordinates, validate_date_string
from src.config import get_secret
from src.features import (
    LEAKAGE_FEATURES,
    PRE_CRASH_CATEGORICAL_FEATURES,
    prepare_training_data,
)
from src.database import execute_query, check_database_integrity
from src.explainability import (
    get_global_feature_importance,
    explain_single_prediction,
    get_causal_disclaimer,
)


class TestDataEdgeCasesAndSanitization:
    """Validate boundary conditions and malformed data handling."""

    def test_coordinate_validation_boundaries(self):
        # Valid Indian coordinates
        assert validate_coordinates(19.0760, 72.8777, region="india") is True   # Mumbai
        assert validate_coordinates(28.6139, 77.2090, region="india") is True   # Delhi

        # UK coordinate should be invalid for India, but valid for UK
        assert validate_coordinates(51.5074, -0.1278, region="india") is False  # London
        assert validate_coordinates(51.5074, -0.1278, region="uk") is True

        # Invalid global coordinates
        assert validate_coordinates(91.0, 77.0) is False        # Latitude > 90
        assert validate_coordinates(20.0, 190.0) is False       # Longitude > 180
        assert validate_coordinates(None, 77.0) is False
        assert validate_coordinates(20.0, float("nan")) is False

    def test_date_validation_edge_cases(self):
        assert validate_date_string("2023-05-15") is True
        assert validate_date_string("2023-02-30") is False  # Non-existent date
        assert validate_date_string("not-a-date") is False
        assert validate_date_string(None) is False
        assert validate_date_string("") is False

    def test_empty_dataframe_feature_preparation(self):
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            prepare_training_data(empty_df)

    def test_missing_target_column_feature_preparation(self):
        df_no_target = pd.DataFrame({"Some_Feature": [1, 2, 3]})
        with pytest.raises(ValueError):
            prepare_training_data(df_no_target)


class TestLeakageAndFeatureSafety:
    """Validate strict exclusion of post-crash leakage indicators."""

    def test_hardcoded_leakage_exclusion_list(self):
        # Critical casualty severity and outcome flags must be in LEAKAGE_FEATURES
        for feat in ["Casualty_severity", "Casualty_class", "Sex_of_casualty"]:
            assert feat in LEAKAGE_FEATURES

    def test_feature_list_clean_of_outcomes(self):
        for feat in PRE_CRASH_CATEGORICAL_FEATURES:
            assert feat not in [
                "Accident_severity",
                "injury_severity",
                "Casualty_severity",
                "Casualty_class",
            ]


class TestDatabaseSecurityAndIntegrity:
    """Validate SQL safety and SQLite index integrity."""

    def test_destructive_sql_blocked_in_execute_query(self):
        with pytest.raises(ValueError, match=r"(Dangerous|SELECT|Read-only)"):
            execute_query("DROP TABLE accidents_india;")

        with pytest.raises(ValueError, match=r"(Dangerous|SELECT|Read-only)"):
            execute_query("DELETE FROM accidents_india WHERE accident_id = 1;")

        with pytest.raises(ValueError, match=r"(Dangerous|SELECT|Read-only)"):
            execute_query("INSERT INTO accidents_india (accident_id) VALUES (9999);")

    def test_database_integrity_check(self):
        integrity = check_database_integrity()
        assert isinstance(integrity, dict)
        assert "is_ok" in integrity
        assert "tables" in integrity


class TestExplainabilitySuite:
    """Validate explainable AI feature importance and single prediction attribution."""

    def test_global_feature_importance_structure(self):
        df_imp = get_global_feature_importance()
        assert not df_imp.empty
        assert "feature" in df_imp.columns
        assert "importance_pct" in df_imp.columns
        assert "cumulative_pct" in df_imp.columns
        assert pytest.approx(df_imp["importance_pct"].sum(), abs=1e-1) == 100.0

    def test_explain_single_prediction(self):
        sample_input = {
            "Number_of_vehicles_involved": 2,
            "Number_of_casualties": 1,
            "Hour": 22,
            "Day_of_week": "Friday",
            "Age_band_of_driver": "18-30",
            "Sex_of_driver": "Male",
            "Educational_level": "Junior high school",
            "Vehicle_driver_relation": "Employee",
            "Driving_experience": "Under 1yr",
            "Type_of_vehicle": "Automobile",
            "Owner_of_vehicle": "Owner",
            "Service_year_of_vehicle": "1-2yr",
            "Defect_of_vehicle": "No defect",
            "Area_accident_occured": "Residential areas",
            "Lanes_or_Medians": "Undivided Two way",
            "Road_allignment": "Tangent road with flat terrain",
            "Types_of_Junction": "No junction",
            "Road_surface_type": "Asphalt roads",
            "Road_surface_conditions": "Dry",
            "Light_conditions": "Darkness - no lighting",
            "Weather_conditions": "Raining",
            "Type_of_collision": "Vehicle with vehicle collision",
            "Vehicle_movement": "Going straight",
            "Cause_of_accident": "Moving Backward",
            "Time_of_Day": "Night",
        }
        res = explain_single_prediction(sample_input)
        assert "predicted_class" in res
        assert "probabilities" in res
        assert "primary_risk_factors" in res
        assert len(res["primary_risk_factors"]) > 0

    def test_causal_disclaimer_content(self):
        disclaimer = get_causal_disclaimer()
        assert "Causality" in disclaimer
        assert "correlation" in disclaimer.lower()


class TestSecretHandlingAndEnvironment:
    """Validate safe configuration and secret masking."""

    def test_missing_secret_returns_default(self):
        res = get_secret("NON_EXISTENT_SECRET_12345", default="safe_default")
        assert res == "safe_default"

    def test_secrets_file_example_has_no_real_keys(self):
        example_path = PROJECT_ROOT / ".streamlit" / "secrets.toml.example"
        if example_path.exists():
            with open(example_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "AIza" not in content  # Standard Google API key prefix
            assert "sk-" not in content   # Standard OpenAI prefix
