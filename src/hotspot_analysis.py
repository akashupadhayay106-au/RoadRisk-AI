"""
RoadRisk AI — Geospatial Hotspot Analysis & Clustering Module
=============================================================
Implements DBSCAN spatial clustering using Haversine geodesic distance,
computes cluster centroids and density metrics, and prepares interactive
Plotly and Folium spatial visualizations.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap

from src.database import execute_query
from src.data_loader import load_urban_gb_dataset

# Earth radius in kilometers for haversine conversion
EARTH_RADIUS_KM = 6371.0088


def compute_dbscan_hotspots(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    eps_km: float = 1.5,
    min_samples: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform DBSCAN density-based spatial clustering using haversine geodesic metric.
    
    Parameters:
    - df: DataFrame containing latitude and longitude.
    - eps_km: Maximum distance in kilometers between two samples for one to be considered in the neighborhood of the other.
    - min_samples: Number of samples in a neighborhood for a point to be considered a core point.
    
    Returns:
    - enriched_df: Original df with 'cluster_label' column (-1 indicates noise).
    - cluster_summary: Aggregated metrics per cluster (centroid lat/lon, size, casualties, mean risk).
    """
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return df, pd.DataFrame()

    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    if len(df_clean) < min_samples:
        df_clean["cluster_label"] = -1
        return df_clean, pd.DataFrame()

    # Convert coordinates from degrees to radians for haversine metric
    coords_rad = np.radians(df_clean[[lat_col, lon_col]].values)
    eps_rad = eps_km / EARTH_RADIUS_KM

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    labels = db.fit_predict(coords_rad)
    df_clean["cluster_label"] = labels

    # Cluster summary aggregation (excluding noise points with label -1)
    clustered = df_clean[df_clean["cluster_label"] != -1]
    
    if clustered.empty:
        return df_clean, pd.DataFrame()

    agg_dict = {
        lat_col: ["mean", "min", "max"],
        lon_col: ["mean", "min", "max"],
    }
    if "casualties" in clustered.columns:
        agg_dict["casualties"] = ["sum", "mean"]
    if "risk_score" in clustered.columns:
        agg_dict["risk_score"] = ["mean"]
    if "accident_id" in clustered.columns:
        agg_dict["accident_id"] = ["count"]
    else:
        agg_dict[lat_col].append("count")

    grouped = clustered.groupby("cluster_label")

    summaries = []
    for cid, group in grouped:
        centroid_lat = float(group[lat_col].mean())
        centroid_lon = float(group[lon_col].mean())
        count = len(group)
        total_cas = int(group["casualties"].sum()) if "casualties" in group.columns else 0
        mean_risk = round(float(group["risk_score"].mean()), 3) if "risk_score" in group.columns else 0.0
        
        # Approximate cluster radius in km
        rad_coords = np.radians(group[[lat_col, lon_col]].values)
        centroid_rad = np.radians([centroid_lat, centroid_lon])
        # Haversine distance from centroid
        dlat = rad_coords[:, 0] - centroid_rad[0]
        dlon = rad_coords[:, 1] - centroid_rad[1]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(centroid_rad[0]) * np.cos(rad_coords[:, 0]) * np.sin(dlon / 2.0) ** 2
        dists_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        radius_km = round(float(np.max(dists_km)), 2) if len(dists_km) > 0 else 0.0

        summaries.append({
            "cluster_id": int(cid),
            "incident_count": count,
            "centroid_latitude": round(centroid_lat, 5),
            "centroid_longitude": round(centroid_lon, 5),
            "cluster_radius_km": radius_km,
            "total_casualties": total_cas,
            "mean_risk_score": mean_risk,
        })

    cluster_summary_df = pd.DataFrame(summaries).sort_values("incident_count", ascending=False).reset_index(drop=True)
    return df_clean, cluster_summary_df


def get_city_accident_clusters(
    city_name: str,
    eps_km: float = 1.0,
    min_samples: int = 15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch incident records for a specific Indian city and calculate spatial clusters.
    """
    sql = """
    SELECT accident_id, city, state, latitude, longitude, accident_severity,
           road_type, weather, casualties, vehicles_involved, risk_score
    FROM accidents_india
    WHERE city = ?;
    """
    df = execute_query(sql, [city_name])
    return compute_dbscan_hotspots(df, eps_km=eps_km, min_samples=min_samples)


def get_urbangb_hotspot_clusters(
    sample_size: int = 25000,
    eps_km: float = 2.0,
    min_samples: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and cluster UrbanGB (Great Britain) accident points.
    Strictly segregated and labeled as Great Britain data.
    """
    df_ugb = load_urban_gb_dataset(nrows=sample_size, include_labels=True)
    return compute_dbscan_hotspots(df_ugb, eps_km=eps_km, min_samples=min_samples)


# ── Interactive Plotly Map Builders ──────────────────────────────────────────

def build_cluster_scatter_mapbox(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    cluster_col: str = "cluster_label",
    title: str = "DBSCAN Spatial Hotspots"
) -> go.Figure:
    """
    Create a high-contrast Plotly OpenStreetMap scatter map highlighting clusters
    and displaying noise (-1) in muted dark gray. Supports Plotly v5 and v6/v7.
    """
    df_plot = df.copy()
    # String cluster labels for clean categorical legend
    df_plot["Cluster"] = df_plot[cluster_col].apply(
        lambda c: "Noise (Unclustered)" if c == -1 else f"Cluster {c}"
    )

    center_lat = float(df_plot[lat_col].mean())
    center_lon = float(df_plot[lon_col].mean())
    hover_cols = [c for c in ["accident_id", "accident_severity", "casualties", "road_type"] if c in df_plot.columns]

    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            df_plot,
            lat=lat_col,
            lon=lon_col,
            color="Cluster",
            hover_data=hover_cols,
            zoom=10,
            center=dict(lat=center_lat, lon=center_lon),
            title=title,
        )
    else:
        fig = px.scatter_mapbox(
            df_plot,
            lat=lat_col,
            lon=lon_col,
            color="Cluster",
            hover_data=hover_cols,
            zoom=10,
            center=dict(lat=center_lat, lon=center_lon),
            mapbox_style="carto-darkmatter",
            title=title,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter"),
    )
    return fig


def build_density_heatmap_mapbox(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    z_col: Optional[str] = "casualties",
    title: str = "Spatial Incident Density Heatmap"
) -> go.Figure:
    """
    Generate an interactive density heatmap using Plotly map/mapbox.
    """
    center_lat = float(df[lat_col].mean())
    center_lon = float(df[lon_col].mean())
    z_arg = z_col if z_col in df.columns else None

    if hasattr(px, "density_map"):
        fig = px.density_map(
            df,
            lat=lat_col,
            lon=lon_col,
            z=z_arg,
            radius=14,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10,
            title=title,
        )
    else:
        fig = px.density_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            z=z_arg,
            radius=14,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10,
            mapbox_style="carto-darkmatter",
            title=title,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter"),
    )
    return fig


def build_folium_hotspot_map(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude"
) -> folium.Map:
    """
    Build a rich interactive Folium map with cluster markers and centroid circles.
    """
    center_lat = float(df[lat_col].mean())
    center_lon = float(df[lon_col].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    # Add heatmap layer
    heat_data = [[row[lat_col], row[lon_col]] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=12, blur=15, min_opacity=0.4).add_to(m)

    # Add centroid markers for identified clusters
    if not cluster_summary.empty:
        for _, c in cluster_summary.iterrows():
            cid = int(c["cluster_id"])
            c_lat = float(c["centroid_latitude"])
            c_lon = float(c["centroid_longitude"])
            count = int(c["incident_count"])
            radius_km = float(c["cluster_radius_km"])

            folium.Circle(
                location=[c_lat, c_lon],
                radius=max(radius_km * 1000, 300),  # meters
                color="#FF4B4B",
                fill=True,
                fill_color="#FF4B4B",
                fill_opacity=0.25,
                tooltip=f"Hotspot Cluster #{cid}: {count} accidents (radius ~{radius_km} km)",
            ).add_to(m)

            folium.Marker(
                location=[c_lat, c_lon],
                popup=f"<b>Hotspot #{cid}</b><br>Incidents: {count}<br>Casualties: {c.get('total_casualties', 0)}<br>Risk: {c.get('mean_risk_score', 0.0)}",
                icon=folium.Icon(color="red", icon="warning-sign"),
            ).add_to(m)

    return m
