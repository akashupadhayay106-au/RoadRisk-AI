"""
RoadRisk AI — Phase 6 Tests: Geospatial Hotspot Analysis & DBSCAN
==================================================================
Tests verifying Haversine geodesic DBSCAN clustering, centroid calculations,
spatial boundary handling for India and Great Britain, and map figure builders.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.hotspot_analysis import (
    compute_dbscan_hotspots,
    get_city_accident_clusters,
    get_urbangb_hotspot_clusters,
    build_cluster_scatter_mapbox,
    build_density_heatmap_mapbox,
)
from src.data_validation import validate_coordinates


class TestDBSCANClustering:
    """Validate spatial clustering algorithms and metric computations."""

    @pytest.fixture(scope="class")
    @staticmethod
    def synthetic_spatial_data():
        # Cluster 1 around Mumbai (Lat ~19.0, Lon ~72.8)
        np.random.seed(42)
        c1_lat = np.random.normal(19.05, 0.005, 30)
        c1_lon = np.random.normal(72.85, 0.005, 30)

        # Cluster 2 around Mumbai north
        c2_lat = np.random.normal(19.20, 0.005, 25)
        c2_lon = np.random.normal(72.90, 0.005, 25)

        # Noise points
        noise_lat = np.random.uniform(18.9, 19.3, 10)
        noise_lon = np.random.uniform(72.7, 73.0, 10)

        lats = np.concatenate([c1_lat, c2_lat, noise_lat])
        lons = np.concatenate([c1_lon, c2_lon, noise_lon])

        return pd.DataFrame({
            "accident_id": range(len(lats)),
            "latitude": lats,
            "longitude": lons,
            "casualties": np.random.randint(0, 4, len(lats)),
            "risk_score": np.random.uniform(0.1, 0.9, len(lats)),
        })

    def test_dbscan_cluster_detection(self, synthetic_spatial_data):
        df_clustered, df_summary = compute_dbscan_hotspots(
            synthetic_spatial_data,
            eps_km=1.5,
            min_samples=10,
        )
        assert "cluster_label" in df_clustered.columns
        assert len(df_summary) >= 2  # Discovered at least the two generated clusters
        assert "centroid_latitude" in df_summary.columns
        assert "cluster_radius_km" in df_summary.columns
        # Check noise points exist
        assert (df_clustered["cluster_label"] == -1).any()

    def test_empty_dataframe_handling(self):
        df_empty = pd.DataFrame(columns=["latitude", "longitude"])
        df_clustered, df_summary = compute_dbscan_hotspots(df_empty)
        assert df_clustered.empty
        assert df_summary.empty


class TestIndianCityHotspots:
    """Validate DBSCAN clustering against live Indian city database records."""

    def test_mumbai_hotspot_clustering(self):
        df_clustered, df_summary = get_city_accident_clusters("Mumbai", eps_km=1.2, min_samples=15)
        assert len(df_clustered) > 0
        assert "cluster_label" in df_clustered.columns
        # Verify coordinates of centroids fall in India
        if not df_summary.empty:
            c = df_summary.iloc[0]
            assert validate_coordinates(c["centroid_latitude"], c["centroid_longitude"], region="india")


class TestUrbanGBHotspots:
    """Validate Great Britain benchmark clustering isolation."""

    def test_urbangb_clustering_bounds(self):
        df_uk, df_uk_summary = get_urbangb_hotspot_clusters(sample_size=3000, eps_km=3.0, min_samples=15)
        assert len(df_uk) > 0
        assert "cluster_label" in df_uk.columns
        # Check centroids fall within UK boundaries
        if not df_uk_summary.empty:
            c = df_uk_summary.iloc[0]
            assert validate_coordinates(c["centroid_latitude"], c["centroid_longitude"], region="uk")


class TestMapBuilders:
    """Validate Mapbox figure generators."""

    def test_cluster_scatter_mapbox_builder(self):
        sample = pd.DataFrame({
            "latitude": [19.0, 19.01, 19.02],
            "longitude": [72.8, 72.81, 72.82],
            "cluster_label": [0, 0, -1],
            "accident_id": [1, 2, 3],
        })
        fig = build_cluster_scatter_mapbox(sample)
        assert isinstance(fig, go.Figure)

    def test_density_heatmap_mapbox_builder(self):
        sample = pd.DataFrame({
            "latitude": [19.0, 19.01, 19.02],
            "longitude": [72.8, 72.81, 72.82],
            "casualties": [1, 3, 0],
        })
        fig = build_density_heatmap_mapbox(sample)
        assert isinstance(fig, go.Figure)
