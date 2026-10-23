"""
🗺️ Hotspot Map — Geospatial Accident Cluster Explorer
=======================================================
Module F: Map-based spatial hotspot exploration with DBSCAN clustering,
geodesic Haversine neighborhood metrics, interactive Plotly/Folium overlays,
and separate international benchmark analysis (UrbanGB Great Britain).
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

from src.config import PAGE_CONFIG, COLORS
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    render_kpi_row,
    info_card,
)
from src.database import execute_query, DATABASE_PATH, load_data_to_database
from src.hotspot_analysis import (
    compute_dbscan_hotspots,
    get_city_accident_clusters,
    get_urbangb_hotspot_clusters,
    build_cluster_scatter_mapbox,
    build_density_heatmap_mapbox,
    build_folium_hotspot_map,
)

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Ensure Database is Loaded ────────────────────────────────────────────────
if not DATABASE_PATH.exists():
    with st.spinner("Preparing analytical database…"):
        load_data_to_database()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #00D4AA, #4DA8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🗺️ Geospatial Hotspot Explorer</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Density-based spatial clustering (DBSCAN) using geodesic Haversine distance to discover empirical accident concentrations.
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Tabs: Indian Cities vs UrbanGB UK Benchmark vs Methodology ───────────────
tab_india, tab_uk, tab_methodology = st.tabs([
    "🇮🇳 Indian City Hotspots (DBSCAN)",
    "🇬🇧 Great Britain UrbanGB Benchmark",
    "⚖️ Count vs Risk Methodology"
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Indian City Hotspots
# ═══════════════════════════════════════════════════════════════════════════════
with tab_india:
    section_header("Urban Hotspot Clustering Configuration", "⚙️")

    cities_df = execute_query("SELECT DISTINCT city FROM accidents_india ORDER BY city ASC;")
    cities_list = cities_df["city"].tolist()

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        sel_city = st.selectbox("Select Target City:", cities_list, index=cities_list.index("Mumbai") if "Mumbai" in cities_list else 0)
    with col_ctrl2:
        eps_km = st.slider("Neighborhood Radius (Eps in km):", 0.5, 3.5, 1.2, step=0.1, help="Max distance between two incidents to be considered part of the same hotspot.")
    with col_ctrl3:
        min_pts = st.slider("Min Incidents for Hotspot:", 5, 40, 15, step=1, help="Minimum accident records required to form a high-density cluster core.")
    with col_ctrl4:
        map_style = st.selectbox("Map Visualization Style:", ["DBSCAN Clusters (Plotly)", "Density Heatmap (Plotly)", "Interactive Centroids (Folium)"], index=0)

    # Execute DBSCAN clustering
    with st.spinner(f"Computing Haversine geodesic clusters for {sel_city}…"):
        df_clustered, df_summary = get_city_accident_clusters(
            city_name=sel_city,
            eps_km=eps_km,
            min_samples=min_pts,
        )

    # Hotspot KPIs
    total_points = len(df_clustered)
    num_clusters = len(df_summary)
    noise_count = int((df_clustered["cluster_label"] == -1).sum())
    clustered_count = total_points - noise_count
    clustered_pct = round((clustered_count / total_points * 100), 1) if total_points > 0 else 0.0

    kpis = [
        ("📍", f"{sel_city} Total Incidents", total_points),
        ("🔥", "Identified Hotspots", num_clusters),
        ("🎯", "Clustered Incidents", f"{clustered_count} ({clustered_pct}%)"),
        ("⚪", "Unclustered Noise", noise_count),
    ]
    render_kpi_row(kpis)

    st.markdown("<br>", unsafe_allow_html=True)

    # Render Map
    if map_style == "DBSCAN Clusters (Plotly)":
        fig_map = build_cluster_scatter_mapbox(
            df_clustered,
            lat_col="latitude",
            lon_col="longitude",
            cluster_col="cluster_label",
            title=f"{sel_city} — DBSCAN Spatial Hotspots (Radius={eps_km} km, MinPts={min_pts})",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    elif map_style == "Density Heatmap (Plotly)":
        fig_heat = build_density_heatmap_mapbox(
            df_clustered,
            lat_col="latitude",
            lon_col="longitude",
            z_col="casualties",
            title=f"{sel_city} — Casualties Density Heatmap",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.markdown(f"##### Interactive Folium Map — {sel_city} Hotspot Centroids & Radii")
        folium_map = build_folium_hotspot_map(
            df_clustered,
            df_summary,
            lat_col="latitude",
            lon_col="longitude",
        )
        st_folium(folium_map, width="100%", height=500)

    # Hotspots Summary Table
    if not df_summary.empty:
        section_header(f"Ranked Hotspot Clusters for {sel_city}", "📋")
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        csv_summary = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {sel_city} Hotspot Summary (CSV)",
            data=csv_summary,
            file_name=f"{sel_city.lower()}_hotspots_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("No dense clusters detected with current parameter thresholds. Try increasing the radius (Eps) or lowering Min Incidents.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Great Britain UrbanGB Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
with tab_uk:
    section_header("Great Britain UrbanGB Benchmark Exploration", "🇬🇧")

    info_card(
        "🛡️ <strong>Provenance & Spatial Segregation:</strong> This dataset contains 360,177 urban road accident "
        "coordinates in Great Britain (UK) from the UCI Machine Learning Repository. It is strictly kept separate "
        "and serves as an international spatial clustering benchmark."
    )

    col_uk1, col_uk2 = st.columns(2)
    with col_uk1:
        uk_sample_size = st.slider("Sample Records for Benchmark:", 5000, 40000, 15000, step=5000)
    with col_uk2:
        uk_eps_km = st.slider("UK DBSCAN Radius (km):", 1.0, 5.0, 2.5, step=0.5)

    with st.spinner("Clustering UrbanGB Great Britain points…"):
        df_uk_clustered, df_uk_summary = get_urbangb_hotspot_clusters(
            sample_size=uk_sample_size,
            eps_km=uk_eps_km,
            min_samples=25,
        )

    kpis_uk = [
        ("📍", "Sample Points Evaluated", len(df_uk_clustered)),
        ("🏙️", "Identified Urban Clusters", len(df_uk_summary)),
        ("🔥", "Largest Cluster Incidents", df_uk_summary["incident_count"].max() if not df_uk_summary.empty else 0),
    ]
    render_kpi_row(kpis_uk)

    st.markdown("<br>", unsafe_allow_html=True)

    # UK Mapbox
    fig_uk = build_cluster_scatter_mapbox(
        df_uk_clustered,
        lat_col="latitude",
        lon_col="longitude",
        cluster_col="cluster_label",
        title=f"Great Britain UrbanGB Hotspot Clusters (Sample={uk_sample_size:,}, Eps={uk_eps_km} km)",
    )
    st.plotly_chart(fig_uk, use_container_width=True)

    if not df_uk_summary.empty:
        st.markdown("##### Top 10 Great Britain Urban Cluster Centroids")
        st.dataframe(df_uk_summary.head(10), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Count vs Risk Rate Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with tab_methodology:
    section_header("Senior Data Science Note: Accident Counts vs True Risk Rates", "⚖️")

    st.markdown("""
    ### 1. The Denominator Problem in Road Safety
    A fundamental error in road accident analytics is labeling an intersection or road as **"most dangerous"**
    purely because it records the highest raw count of crashes.
    
    $$\\text{Empirical Risk Rate} = \\frac{\\text{Reported Crashes}}{\\text{Traffic Exposure (VKT or AADT)}}$$
    
    - **Exposure Variable:** AADT (*Average Annual Daily Traffic*) or Vehicle-Kilometers Traveled (VKT).
    - **Example:**
      - **Location A:** An arterial ring road carrying **100,000 vehicles per day** logs **25 accidents/year**.
      - **Location B:** A residential intersection carrying **1,000 vehicles per day** logs **10 accidents/year**.
      - **Conclusion:** While Location A has a higher *raw accident count*, Location B has an exposure-adjusted crash rate **40 times higher**!

    ---
    ### 2. Spatial Clustering (DBSCAN) Strengths & Limits
    - **Why DBSCAN over K-Means:** K-Means forces spherical clusters and requires pre-specifying $k$. DBSCAN discovers arbitrary-shaped corridors (like arterial highways) and explicitly filters sparse random incidents as noise (-1).
    - **Distance Metric:** Standard Euclidean distance $(\\sqrt{\\Delta x^2 + \\Delta y^2})$ distorts true distances across latitudes. We convert degrees to radians and employ the **Haversine geodesic metric** scaled by Earth's radius ($R = 6,371.0088\\text{ km}$).

    ---
    ### 3. Policy & Operational Guidance
    Observed clusters identify **locations with high emergency response demand**, but must not be used to condemn road geometry without integrating traffic census volume metrics.
    """)
