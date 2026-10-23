"""
RoadRisk AI — Phase 1 Tests
=============================
Basic tests to verify project structure and imports work correctly.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectStructure:
    """Verify the project directory structure exists."""

    def test_app_py_exists(self):
        assert (PROJECT_ROOT / "app.py").exists()

    def test_src_package_exists(self):
        assert (PROJECT_ROOT / "src" / "__init__.py").exists()

    def test_config_module_exists(self):
        assert (PROJECT_ROOT / "src" / "config.py").exists()

    def test_utils_module_exists(self):
        assert (PROJECT_ROOT / "src" / "utils.py").exists()

    def test_pages_directory_exists(self):
        pages_dir = PROJECT_ROOT / "pages"
        assert pages_dir.exists()
        assert pages_dir.is_dir()

    def test_all_pages_exist(self):
        expected_pages = [
            "1_Dashboard.py",
            "2_Data_Explorer.py",
            "3_SQL_Analytics.py",
            "4_Severity_Prediction.py",
            "5_Hotspot_Map.py",
            "6_Weather_Analysis.py",
            "7_CCTV_Detection.py",
            "8_AI_Assistant.py",
            "9_Model_Explainability.py",
            "10_About.py",
        ]
        pages_dir = PROJECT_ROOT / "pages"
        for page in expected_pages:
            assert (pages_dir / page).exists(), f"Missing page: {page}"

    def test_streamlit_config_exists(self):
        assert (PROJECT_ROOT / ".streamlit" / "config.toml").exists()


class TestConfigImports:
    """Test that the config module loads correctly."""

    def test_import_config(self):
        from src.config import APP_TITLE, APP_VERSION, PROJECT_ROOT as PR
        assert APP_TITLE == "🚦 RoadRisk AI"
        assert APP_VERSION == "0.1.0"
        assert PR.exists()

    def test_color_palette(self):
        from src.config import COLORS
        assert "primary" in COLORS
        assert "chart_palette" in COLORS
        assert len(COLORS["chart_palette"]) >= 4

    def test_datasets_metadata(self):
        from src.config import DATASETS
        assert "indian_accidents" in DATASETS
        assert "severity" in DATASETS
        assert "urban_gb" in DATASETS
        assert "cctv" in DATASETS
        for key, meta in DATASETS.items():
            assert "name" in meta
            assert "url" in meta

    def test_ensure_directories(self):
        from src.config import ensure_directories, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR
        ensure_directories()
        assert RAW_DATA_DIR.exists()
        assert PROCESSED_DIR.exists()
        assert MODELS_DIR.exists()


class TestDataValidation:
    """Test data validation functions."""

    def test_valid_coordinates(self):
        from src.data_validation import validate_coordinates
        assert validate_coordinates(28.6139, 77.2090)  # Delhi
        assert validate_coordinates(-33.8688, 151.2093)  # Sydney
        assert validate_coordinates(0.0, 0.0)  # Null Island

    def test_invalid_coordinates(self):
        from src.data_validation import validate_coordinates
        assert not validate_coordinates(91, 0)
        assert not validate_coordinates(0, 181)
        assert not validate_coordinates(-91, -181)

    def test_valid_date(self):
        from src.data_validation import validate_date_string
        assert validate_date_string("2024-01-15")
        assert validate_date_string("2023-12-31")

    def test_invalid_date(self):
        from src.data_validation import validate_date_string
        assert not validate_date_string("not-a-date")
        assert not validate_date_string("2024-13-01")
        assert not validate_date_string("")


class TestDataLoader:
    """Test data loader functions (file discovery, not data content)."""

    def test_find_dataset_missing(self):
        from src.data_loader import find_dataset
        result = find_dataset("nonexistent_file_xyz.csv")
        assert result is None

    def test_find_dataset_existing(self):
        from src.data_loader import find_dataset
        # This should find the Indian accidents file (in root or raw/)
        result = find_dataset("indian_roads_dataset.csv")
        # May or may not exist depending on data setup
        if result is not None:
            assert result.exists()
