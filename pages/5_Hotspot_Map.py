"""
🗺️ Hotspot Map — Geospatial Accident Cluster Explorer
=======================================================
Module F: Map-based spatial hotspot exploration with DBSCAN clustering,
geodesic Haversine neighborhood metrics, interactive Plotly/Folium overlays,
and separate international benchmark analysis (UrbanGB Great Britain).
"""

import streamlit as st
import plotly.graph_objects as go
from streamlit_folium import st_folium

from src.config import PAGE_CONFIG, COLORS
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    render_kpi_row,
    info_card,
    empty_state,
    error_state,
    service_status_row,
)
from src.database import execute_query, DATABASE_PATH, load_data_to_database
from src.hotspot_analysis import (
    get_city_accident_clusters,
    get_urbangb_hotspot_clusters,
    build_cluster_scatter_mapbox,
    build_density_heatmap_mapbox,
    build_folium_hotspot_map,
)
from src.routing_client import (
    get_routing_client,
    routing_service_status,
    SUPPORTED_PROFILES,
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
tab_india, tab_uk, tab_route, tab_methodology = st.tabs([
    "🇮🇳 Indian City Hotspots (DBSCAN)",
    "🇬🇧 Great Britain UrbanGB Benchmark",
    "🧭 Route & Reachability",
    "⚖️ Count vs Risk Methodology"
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Indian City Hotspots
# ═══════════════════════════════════════════════════════════════════════════════
with tab_india:
    section_header("Urban Hotspot Clustering Configuration", "⚙️")

    cities_df = execute_query("SELECT DISTINCT city FROM accidents_india ORDER BY city ASC;")
    cities_list = cities_df["city"].tolist()

    if not cities_list:
        empty_state(
            "No Indian incident data available",
            "The accident database contains no city records, so DBSCAN hotspot clustering cannot run.",
            hint="Place the source dataset in data/raw/ and rebuild the database from the Home page to enable spatial clustering.",
            icon="🛰️",
        )
    else:
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
        noise_count = int((df_clustered["cluster_label"] == -1).sum()) if "cluster_label" in df_clustered.columns else 0
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
            empty_state(
                "No dense clusters detected",
                "The current DBSCAN parameters did not find any cluster meeting the minimum incident threshold.",
                hint="Try increasing the neighborhood radius (Eps) or lowering the minimum incidents required for a hotspot.",
                icon="🔥",
            )


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

    uk_clustered = None
    with st.spinner("Clustering UrbanGB Great Britain points…"):
        try:
            uk_clustered, df_uk_summary = get_urbangb_hotspot_clusters(
                sample_size=uk_sample_size,
                eps_km=uk_eps_km,
                min_samples=25,
            )
        except FileNotFoundError:
            uk_clustered = None

    if uk_clustered is None:
        empty_state(
            "UrbanGB benchmark data not available",
            "The Great Britain UrbanGB coordinate file is not present in `data/raw/`.",
            hint="Download `urbanGB.txt` from the UCI Machine Learning Repository (dataset 550) and place it in data/raw/ to enable this international benchmark.",
            icon="🇬🇧",
        )
    else:
        kpis_uk = [
            ("📍", "Sample Points Evaluated", len(uk_clustered)),
            ("🏙️", "Identified Urban Clusters", len(df_uk_summary)),
            ("🔥", "Largest Cluster Incidents", df_uk_summary["incident_count"].max() if not df_uk_summary.empty else 0),
        ]
        render_kpi_row(kpis_uk)

        st.markdown("<br>", unsafe_allow_html=True)

        # UK Mapbox
        fig_uk = build_cluster_scatter_mapbox(
            uk_clustered,
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
#  TAB 3: Route & Reachability (HeiGIT openrouteservice)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_route:
    section_header("Road-Network Routing & Reachability", "🧭")

    st.caption(
        "Live road-network distance and duration are provided by HeiGIT openrouteservice. "
        "This is an optional integration: without a key it stays disabled and every offline "
        "DBSCAN hotspot analysis on the other tabs continues to work normally."
    )

    r_status = routing_service_status(probe=False)
    service_status_row([r_status])
    if not r_status["configured"]:
        st.caption(
            "Set `OPENROUTESERVICE_API_KEY` (locally in `.env`, or in Streamlit secrets when "
            "deployed) to enable live routing. Offline hotspot analytics work regardless."
        )

    city_coords_df = execute_query(
        """
        SELECT city,
               AVG(latitude)  AS lat,
               AVG(longitude) AS lon,
               COUNT(*)       AS incidents
        FROM accidents_india
        GROUP BY city
        HAVING incidents >= 20
        ORDER BY incidents DESC;
        """
    )

    if city_coords_df.empty:
        error_state(
            "No city coordinates available",
            "The database returned no city coordinates, so origin/destination routing cannot be demonstrated.",
            hint="Rebuild the database from the Home page so city geolocations are available.",
            icon="🧭",
        )
    else:
        city_options = city_coords_df["city"].tolist()
        coord_lookup = {
            row["city"]: (float(row["lat"]), float(row["lon"]))
            for _, row in city_coords_df.iterrows()
        }

        col_r1, col_r2, col_r3 = st.columns([2, 2, 1])
        with col_r1:
            origin_city = st.selectbox("Origin (city centroid)", city_options, index=0, key="route_origin")
        with col_r2:
            dest_city = st.selectbox(
                "Destination (city centroid)",
                city_options,
                index=min(1, len(city_options) - 1),
                key="route_dest",
            )
        with col_r3:
            route_profile = st.selectbox("Profile", SUPPORTED_PROFILES, index=0, key="route_profile")

        if st.button("🧭 Compute live route", use_container_width=True, key="compute_route"):
            if origin_city == dest_city:
                st.warning("Choose two different cities to compute a route.")
            else:
                with st.spinner("Requesting route from openrouteservice…"):
                    route = get_routing_client().get_route(
                        coord_lookup[origin_city],
                        coord_lookup[dest_city],
                        profile=route_profile,
                    )

                if route and route.get("geometry"):
                    render_kpi_row([
                        ("🛣️", "Road Distance", f"{route['distance_km']:,.1f} km"),
                        ("⏱️", "Estimated Duration", f"{route['duration_min'] / 60:,.2f} h"),
                        ("🚗", "Travel Profile", route["profile"]),
                    ])

                    lats = [point[1] for point in route["geometry"]]
                    lons = [point[0] for point in route["geometry"]]
                    origin_lat, origin_lon = coord_lookup[origin_city]
                    dest_lat, dest_lon = coord_lookup[dest_city]

                    fig_route = go.Figure()
                    fig_route.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons, mode="lines",
                        line=dict(width=4, color=COLORS["accent"]),
                        name="Road route",
                    ))
                    fig_route.add_trace(go.Scattermapbox(
                        lat=[origin_lat], lon=[origin_lon], mode="markers",
                        marker=dict(size=15, color=COLORS["primary"]),
                        name=f"Origin · {origin_city}",
                    ))
                    fig_route.add_trace(go.Scattermapbox(
                        lat=[dest_lat], lon=[dest_lon], mode="markers",
                        marker=dict(size=15, color=COLORS["info"]),
                        name=f"Destination · {dest_city}",
                    ))
                    fig_route.update_layout(
                        mapbox_style="carto-darkmatter",
                        mapbox=dict(
                            center=dict(lat=(origin_lat + dest_lat) / 2, lon=(origin_lon + dest_lon) / 2),
                            zoom=4.4,
                        ),
                        height=480,
                        margin=dict(l=0, r=0, t=36, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01),
                        title=f"{origin_city} → {dest_city} · road-network route ({route_profile})",
                    )
                    st.plotly_chart(fig_route, use_container_width=True)

                    st.caption(
                        "Distance and duration are real road-network values returned by openrouteservice "
                        "(HeiGIT). No traffic, weather or toll adjustments are applied."
                    )
                else:
                    error_state(
                        "Routing currently unavailable",
                        "The routing service did not return a route — it may be unconfigured, rate-limited, or temporarily unreachable.",
                        hint="Offline DBSCAN hotspot analytics on the other tabs are unaffected and remain fully functional.",
                        icon="🛣️",
                    )

    st.markdown(
        """
        <div style="font-size:0.78rem;color:#8899A6;line-height:1.5;">
            <strong>Why this matters:</strong> straight-line (Haversine) distance understates real
            emergency-response travel time. Road-network routing converts hotspot coordinates into
            realistic response distances, which is the metric that actually drives ambulance
            dispatch planning.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: Count vs Risk Rate Methodology
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
