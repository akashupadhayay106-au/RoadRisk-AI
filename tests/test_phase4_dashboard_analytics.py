"""
RoadRisk AI — Phase 4 Tests: Dashboard & Exploratory Analytics
==============================================================
Tests verifying multi-faceted filtering queries, day-hour heatmap matrix,
state benchmarks, vehicle collision dynamics, and single-record dossier retrieval.
"""

from src.sql_queries import (
    get_overall_kpis,
    get_day_hour_heatmap_matrix,
    get_state_comparisons,
    get_vehicle_collision_breakdown,
    get_detailed_accident_record,
)


class TestDashboardAnalyticalQueries:
    """Validate query outputs feeding executive dashboard visualizations."""

    def test_multi_filter_kpi_query(self):
        # Filter by city and severity
        kpis = get_overall_kpis(city="Mumbai", severity="fatal")
        assert kpis["total_accidents"] > 0
        assert kpis["fatal_accidents"] == kpis["total_accidents"]
        assert kpis["fatality_rate_pct"] == 100.0

    def test_day_hour_heatmap_matrix(self):
        df_heat = get_day_hour_heatmap_matrix()
        assert not df_heat.empty
        assert "day_of_week" in df_heat.columns
        assert "hour" in df_heat.columns
        assert "incident_count" in df_heat.columns
        # Check hour range is 0 to 23
        assert df_heat["hour"].min() == 0
        assert df_heat["hour"].max() == 23

    def test_state_comparisons(self):
        df_states = get_state_comparisons()
        assert not df_states.empty
        assert "state" in df_states.columns
        assert "fatal_rate_pct" in df_states.columns
        assert len(df_states) == 7  # 7 distinct Indian states in dataset

    def test_vehicle_collision_breakdown(self):
        df_veh = get_vehicle_collision_breakdown()
        assert not df_veh.empty
        assert "vehicles_involved" in df_veh.columns
        assert "avg_casualties_per_accident" in df_veh.columns
        assert (df_veh["avg_casualties_per_accident"] > 0).all()

    def test_single_record_dossier_inspector(self):
        # Fetch accident_id 0
        rec = get_detailed_accident_record(0)
        assert rec is not None
        assert rec["accident_id"] == 0
        assert "city" in rec
        assert "state" in rec
        assert "latitude" in rec
        assert "longitude" in rec
        assert "accident_severity" in rec
        assert "casualties" in rec

    def test_nonexistent_record_returns_none(self):
        rec = get_detailed_accident_record(99999999)
        assert rec is None
