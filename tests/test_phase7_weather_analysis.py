"""
Phase 7 Unit Tests — Historical Weather Client & Risk Intelligence
===================================================================
Tests WMO mapping, SQLite caching, API fetching with offline fallback,
spatiotemporal incident matching, and macro weather risk metrics.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import pandas as pd
import numpy as np

from src.weather_client import (
    get_wmo_metadata,
    make_cache_key,
    init_weather_cache,
    save_weather_to_cache,
    get_cached_weather,
    fetch_historical_weather,
    _generate_fallback_weather,
    get_incident_weather_match,
    compute_weather_risk_metrics,
)


class TestWMOInterpretation:
    """Validate WMO weather interpretation code mapping."""

    def test_clear_sky_code(self):
        meta = get_wmo_metadata(0)
        assert meta["category"] == "Clear"
        assert meta["is_adverse"] is False
        assert "clear" in meta["description"].lower()

    def test_heavy_rain_code(self):
        meta = get_wmo_metadata(65)
        assert meta["category"] == "Rain"
        assert meta["is_adverse"] is True

    def test_dense_fog_code(self):
        meta = get_wmo_metadata(45)
        assert meta["category"] == "Fog"
        assert meta["is_adverse"] is True

    def test_thunderstorm_code(self):
        meta = get_wmo_metadata(95)
        assert meta["category"] == "Thunderstorm"
        assert meta["is_adverse"] is True

    def test_unknown_code_fallback(self):
        meta = get_wmo_metadata(9999)
        assert "description" in meta
        assert meta["category"] == "Other"
        assert meta["is_adverse"] is False

    def test_none_nan_code_handling(self):
        assert get_wmo_metadata(None)["category"] == "Other"
        assert get_wmo_metadata(np.nan)["category"] == "Other"


class TestWeatherCache:
    """Validate SQLite weather caching operations."""

    def test_cache_key_generation(self):
        key = make_cache_key(19.07604, 72.87771, "2024-01-15", 14)
        assert key == "19.08_72.88_2024-01-15_14"

    def test_cache_initialization_and_crud(self, tmp_path):
        test_db = tmp_path / "test_cache.db"
        init_weather_cache(test_db)

        # Verify table exists
        with sqlite3.connect(test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weather_cache'")
            assert cursor.fetchone() is not None

        # Create dummy weather row
        df_dummy = pd.DataFrame([{
            "cache_key": "19.08_72.88_2024-01-15_14",
            "latitude": 19.08,
            "longitude": 72.88,
            "date": "2024-01-15",
            "hour": 14,
            "datetime": "2024-01-15 14:00:00",
            "temperature_2m": 31.5,
            "relative_humidity_2m": 55.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "weather_code": 0,
            "wind_speed_10m": 12.0,
            "wmo_description": "Clear sky",
            "wmo_category": "Clear",
            "wmo_emoji": "☀️",
            "is_adverse": 0,
        }])

        saved_count = save_weather_to_cache(df_dummy, db_path=test_db)
        assert saved_count == 1

        # Retrieve and verify
        cached = get_cached_weather(19.08, 72.88, "2024-01-15", hour=14, db_path=test_db)
        assert len(cached) == 1
        assert cached.iloc[0]["temperature_2m"] == 31.5
        assert cached.iloc[0]["wmo_description"] == "Clear sky"


class TestWeatherClientAPI:
    """Validate Open-Meteo fetching and offline fallback generation."""

    def test_offline_fallback_generator(self):
        df_fallback = _generate_fallback_weather(
            lat=19.0760, lon=72.8777, start_date="2024-07-15", end_date="2024-07-15"
        )
        assert len(df_fallback) == 24
        assert "temperature_2m" in df_fallback.columns
        assert "precipitation" in df_fallback.columns
        assert "wmo_description" in df_fallback.columns
        assert df_fallback["temperature_2m"].min() > 10.0
        assert df_fallback["temperature_2m"].max() < 50.0

    def test_fetch_historical_weather_structure(self, tmp_path):
        test_db = tmp_path / "fetch_test.db"
        df = fetch_historical_weather(
            lat=19.0760,
            lon=72.8777,
            start_date="2024-01-15",
            end_date="2024-01-15",
            use_cache=True,
            db_path=test_db,
        )
        assert not df.empty
        assert len(df) >= 24
        required_cols = ["date", "hour", "temperature_2m", "wmo_description", "is_adverse"]
        for col in required_cols:
            assert col in df.columns

    def test_incident_weather_match(self, tmp_path):
        test_db = tmp_path / "match_test.db"
        match = get_incident_weather_match(
            lat=19.0760,
            lon=72.8777,
            date_str="2024-01-15",
            hour=14,
            use_cache=True,
            db_path=test_db,
        )
        assert isinstance(match, dict)
        assert match.get("hour") == 14
        assert "temperature_2m" in match


class TestWeatherRiskMetrics:
    """Validate macro statistical weather risk metric calculation."""

    def test_empty_dataframe(self):
        res = compute_weather_risk_metrics(pd.DataFrame())
        assert res == {}

    def test_adverse_vs_normal_metrics_calculation(self):
        data = {
            "weather": ["Clear", "Rain", "Fog", "Clear", "Clear", "Rain"],
            "accident_severity": ["Minor", "Fatal", "Serious", "Minor", "Minor", "Serious"],
            "temperature": [28.0, 24.0, 18.0, 30.0, 29.0, 23.0],
            "visibility": ["High", "Low", "Low", "High", "High", "Low"],
        }
        df = pd.DataFrame(data)
        metrics = compute_weather_risk_metrics(df)

        assert metrics["total_accidents"] == 6
        assert "Clear" in metrics["weather_counts"]
        assert "Rain" in metrics["weather_counts"]

        # Adverse group (Rain + Fog) has Fatal and Serious accidents
        adv_comp = metrics["adverse_comparison"]
        assert "Adverse Weather (Rain/Fog/Storm)" in adv_comp
        assert "Normal Weather (Clear/Cloudy)" in adv_comp

        adv_stats = adv_comp["Adverse Weather (Rain/Fog/Storm)"]
        assert adv_stats["total_accidents"] == 3
        # In adverse: 1 Fatal, 2 Serious -> 100% severe combined rate
        assert adv_stats["severe_combined_rate"] == 100.0

        # Relative risk ratio should be > 1.0 (since normal is 0% severe in this sample)
        assert "relative_severe_risk_ratio" in metrics
