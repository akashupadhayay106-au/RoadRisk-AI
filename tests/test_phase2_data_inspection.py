"""
RoadRisk AI — Phase 2 Tests: Data Inspection & Quality Validation
==================================================================
Tests verifying that all 5 datasets are properly discovered, validly formatted,
and conform to schema, geographic bounds, and data quality requirements.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.config import PROJECT_ROOT
from src.data_loader import (
    find_dataset_file,
    load_indian_roads_dataset,
    load_severity_train_dataset,
    load_severity_test_dataset,
    load_urban_gb_dataset,
    load_opencity_dataset,
    get_cctv_image_manifest,
)
from src.data_validation import (
    validate_coordinates,
    check_dataframe_coordinates,
    generate_data_quality_report,
    detect_target_leakage_features,
)


class TestDatasetAvailability:
    """Verify that all raw files exist in the project."""

    def test_indian_roads_file_exists(self):
        assert find_dataset_file("indian_roads_dataset.csv") is not None

    def test_severity_files_exist(self):
        assert find_dataset_file("train.csv") is not None
        assert find_dataset_file("test.csv") is not None

    def test_urbangb_files_exist(self):
        assert find_dataset_file("urbanGB.txt") is not None
        assert find_dataset_file("urbanGB.labels.txt") is not None

    def test_opencity_file_exists(self):
        assert find_dataset_file("Accdataset_hk_PS_BAEL_Combined.csv") is not None


class TestIndianRoadsDataset:
    """Validate schema and contents of the Indian Road Accident Dataset."""

    @pytest.fixture(scope="class")
    @staticmethod
    def df_indian():
        return load_indian_roads_dataset(nrows=1000)

    def test_required_columns_present(self, df_indian):
        required = [
            "accident_id", "city", "state", "latitude", "longitude",
            "date", "accident_severity", "weather", "cause"
        ]
        for col in required:
            assert col in df_indian.columns, f"Missing required column: {col}"

    def test_coordinates_within_india_bounds(self, df_indian):
        geo = check_dataframe_coordinates(df_indian, "latitude", "longitude", region="india")
        assert geo["has_coordinates"] is True
        assert geo["valid_pct"] >= 99.0, f"Expected >=99% valid coordinates, got {geo['valid_pct']}%"

    def test_severity_classes(self, df_indian):
        classes = set(df_indian["accident_severity"].dropna().str.lower().unique())
        expected = {"minor", "major", "fatal"}
        assert expected.issubset(classes) or classes.issubset(expected)


class TestSeverityClassificationDataset:
    """Validate the ML severity dataset and leakage rules."""

    @pytest.fixture(scope="class")
    @staticmethod
    def df_train():
        return load_severity_train_dataset(nrows=500)

    def test_target_column_present(self, df_train):
        assert "Accident_severity" in df_train.columns

    def test_leakage_detection(self, df_train):
        suspicious = detect_target_leakage_features(df_train.columns.tolist())
        # Casualty_severity is a known post-outcome feature
        assert any("casualty" in s.lower() for s in suspicious)


class TestUrbanGBDataset:
    """Validate UrbanGB coordinates and separation from Indian data."""

    @pytest.fixture(scope="class")
    @staticmethod
    def df_ugb():
        return load_urban_gb_dataset(nrows=500, include_labels=True)

    def test_coordinates_within_uk_bounds(self, df_ugb):
        geo = check_dataframe_coordinates(df_ugb, "latitude", "longitude", region="uk")
        assert geo["has_coordinates"] is True
        assert geo["valid_pct"] >= 95.0, f"Expected >=95% valid UK coordinates, got {geo['valid_pct']}%"

    def test_not_in_india_bounds(self, df_ugb):
        geo_india = check_dataframe_coordinates(df_ugb, "latitude", "longitude", region="india")
        assert geo_india["valid_pct"] == 0.0, "UrbanGB coordinates should not be inside India"


class TestCCTVDataset:
    """Validate CCTV image dataset structure."""

    def test_image_manifest_indexing(self):
        manifest = get_cctv_image_manifest()
        assert len(manifest) > 0, "No CCTV images found in data/train, data/test, data/val"
        splits = set(manifest["split"].unique())
        assert {"train", "test", "val"}.issubset(splits)
        classes = set(manifest["class_label"].unique())
        assert {"Accident", "Non Accident"}.issubset(classes)
