"""
RoadRisk AI — Phase 3 Tests: Data Cleaning & SQL Database Pipelines
===================================================================
Tests verifying data cleaning pipelines, SQLite database initialization,
indexing, safety checks against injection/destructive SQL, and query functions.
"""

import pytest
from pathlib import Path
import tempfile

from src.data_cleaning import (
    clean_indian_roads_dataset,
    clean_severity_ml_dataset,
    get_time_of_day,
)
from src.database import (
    init_database,
    load_data_to_database,
    execute_query,
    get_database_summary,
)
from src.sql_queries import (
    get_overall_kpis,
    get_monthly_trends,
    get_city_comparisons,
    PREDEFINED_QUERIES,
)


class TestDataCleaningPipeline:
    """Test data cleaning and normalization routines."""

    def test_time_of_day_categorization(self):
        assert get_time_of_day(6) == "Morning"
        assert get_time_of_day(13) == "Afternoon"
        assert get_time_of_day(19) == "Evening"
        assert get_time_of_day(2) == "Night"
        assert get_time_of_day(23) == "Night"

    def test_indian_roads_cleaning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_parquet = Path(tmpdir) / "test_cleaned.parquet"
            df, summary = clean_indian_roads_dataset(output_path=out_parquet)

            assert len(df) == 20000
            assert "year" in df.columns
            assert "month" in df.columns
            assert "time_of_day" in df.columns
            # Festival nulls standardized to 'None'
            assert (df["festival"] == "None").sum() > 19000
            # Target categories are lowercase
            assert set(df["accident_severity"].unique()) == {"minor", "major", "fatal"}

    def test_severity_dataset_cleaning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "test_sev_cleaned.csv"
            df, summary = clean_severity_ml_dataset(split="train", output_path=out_csv)

            assert len(df) == 8210
            assert "Hour" in df.columns
            assert "Time_of_Day" in df.columns
            # Missing tokens standardized away from 'na'
            assert "na" not in df["Educational_level"].values


class TestDatabaseManagement:
    """Test SQLite initialization, loading, and read-only query execution."""

    @pytest.fixture(scope="class")
    @staticmethod
    def test_db_path():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        init_database(db_path, force_recreate=True)
        load_data_to_database(db_path)
        yield db_path
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass

    def test_database_tables_exist(self, test_db_path):
        summary = get_database_summary(test_db_path)
        assert "accidents_india" in summary["tables"]
        assert "highway_accidents_opencity" in summary["tables"]
        assert summary["tables"]["accidents_india"]["row_count"] == 20000

    def test_database_indexes_exist(self, test_db_path):
        summary = get_database_summary(test_db_path)
        index_names = [idx["name"] for idx in summary["indexes"]]
        assert "idx_acc_date" in index_names
        assert "idx_acc_city" in index_names
        assert "idx_acc_severity" in index_names

    def test_safe_query_execution(self, test_db_path):
        df = execute_query("SELECT COUNT(*) AS total FROM accidents_india;", db_path=test_db_path)
        assert len(df) == 1
        assert df.iloc[0]["total"] == 20000

    def test_destructive_query_blocked(self, test_db_path):
        with pytest.raises(ValueError, match="Dangerous operation detected"):
            execute_query("DROP TABLE accidents_india;", db_path=test_db_path)

        with pytest.raises(ValueError, match="Dangerous operation detected"):
            execute_query("DELETE FROM accidents_india WHERE 1=1;", db_path=test_db_path)


class TestSQLQueries:
    """Test SQL query bank analytical functions."""

    def test_kpi_query(self):
        kpis = get_overall_kpis()
        assert "total_accidents" in kpis
        assert kpis["total_accidents"] == 20000
        assert kpis["fatal_accidents"] > 0
        assert kpis["fatality_rate_pct"] > 0

    def test_kpi_filtered_query(self):
        kpis_delhi = get_overall_kpis(city="Delhi")
        assert kpis_delhi["total_accidents"] > 0
        assert kpis_delhi["total_accidents"] < 20000

    def test_monthly_trends(self):
        df_trends = get_monthly_trends()
        assert not df_trends.empty
        assert "year_month" in df_trends.columns
        assert "accident_count" in df_trends.columns

    def test_city_comparisons(self):
        df_cities = get_city_comparisons()
        assert len(df_cities) == 8
        assert "fatal_rate_pct" in df_cities.columns

    def test_predefined_queries_catalog(self):
        assert len(PREDEFINED_QUERIES) >= 4
        for name, q in PREDEFINED_QUERIES.items():
            assert "sql" in q
            # Test that all predefined queries execute without error
            df = execute_query(q["sql"])
            assert not df.empty, f"Predefined query '{name}' returned empty result"
