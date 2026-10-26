"""
📊 Dashboard — Interactive Analytics & KPI Overview
=====================================================
Module A: High-level executive dashboard presenting verified historical
road safety metrics, temporal trajectories, regional comparisons,
and environmental associations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PAGE_CONFIG
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    render_kpi_row,
    info_card,
    empty_state,
)
from src.database import execute_query, DATABASE_PATH, load_data_to_database
from src.sql_queries import (
    get_state_comparisons,
    get_city_comparisons,
    get_vehicle_collision_breakdown,
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
    background: linear-gradient(135deg, #FF4B4B, #FF8C42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">📊 Executive Road Safety Dashboard</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Multi-dimensional intelligence and empirical risk patterns across 20,000 audited Indian road incidents (2022–2025).
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar Multi-Faceted Filters ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filter Panel")

    # City filter
    cities_df = execute_query("SELECT DISTINCT city FROM accidents_india ORDER BY city ASC;")
    city_options = ["All"] + cities_df["city"].tolist()
    selected_city = st.selectbox("Select City:", city_options, index=0)

    # State filter
    states_df = execute_query("SELECT DISTINCT state FROM accidents_india ORDER BY state ASC;")
    state_options = ["All"] + states_df["state"].tolist()
    selected_state = st.selectbox("Select State:", state_options, index=0)

    # Year filter
    years_df = execute_query("SELECT DISTINCT year FROM accidents_india ORDER BY year ASC;")
    year_options = ["All"] + [str(y) for y in years_df["year"].tolist()]
    selected_year = st.selectbox("Select Year:", year_options, index=0)

    # Severity filter
    severity_options = ["All", "Minor", "Major", "Fatal"]
    selected_severity = st.selectbox("Accident Severity:", severity_options, index=0)

    # Weather filter
    weather_df = execute_query("SELECT DISTINCT weather FROM accidents_india ORDER BY weather ASC;")
    weather_options = ["All"] + [w.title() for w in weather_df["weather"].tolist()]
    selected_weather = st.selectbox("Weather Condition:", weather_options, index=0)

    # Road Type filter
    road_df = execute_query("SELECT DISTINCT road_type FROM accidents_india ORDER BY road_type ASC;")
    road_options = ["All"] + [r.title() for r in road_df["road_type"].tolist()]
    selected_road = st.selectbox("Road Classification:", road_options, index=0)

    st.markdown("---")
    st.caption("Filters apply in real time via parameterized SQLite query execution.")


# ── Query Construction ───────────────────────────────────────────────────────
where_clauses = ["1=1"]
params = []

if selected_city != "All":
    where_clauses.append("city = ?")
    params.append(selected_city)
if selected_state != "All":
    where_clauses.append("state = ?")
    params.append(selected_state)
if selected_year != "All":
    where_clauses.append("year = ?")
    params.append(int(selected_year))
if selected_severity != "All":
    where_clauses.append("accident_severity = ?")
    params.append(selected_severity.lower())
if selected_weather != "All":
    where_clauses.append("weather = ?")
    params.append(selected_weather.lower())
if selected_road != "All":
    where_clauses.append("road_type = ?")
    params.append(selected_road.lower())

where_sql = " AND ".join(where_clauses)


# ── 1. KPI Cards Row ─────────────────────────────────────────────────────────
kpi_sql = f"""
SELECT 
    COUNT(*) AS total_accidents,
    COALESCE(SUM(casualties), 0) AS total_casualties,
    COALESCE(SUM(vehicles_involved), 0) AS total_vehicles,
    COALESCE(SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END), 0) AS fatal_accidents,
    COALESCE(SUM(CASE WHEN accident_severity = 'major' THEN 1 ELSE 0 END), 0) AS major_accidents,
    ROUND(AVG(risk_score), 3) AS avg_risk_score
FROM accidents_india
WHERE {where_sql};
"""
df_kpi = execute_query(kpi_sql, params)

if not df_kpi.empty:
    k = df_kpi.iloc[0]
    total_acc = int(k["total_accidents"])
    total_cas = int(k["total_casualties"])
    total_veh = int(k["total_vehicles"])
    fatal_acc = int(k["fatal_accidents"])
    fatal_rate = round((fatal_acc / total_acc * 100), 2) if total_acc > 0 else 0.0
    risk_score = float(k["avg_risk_score"]) if pd.notna(k["avg_risk_score"]) else 0.0

    kpis = [
        ("🚗", "Total Incidents", total_acc),
        ("🩹", "Total Casualties", total_cas),
        ("⚠️", "Fatal Incidents", fatal_acc),
        ("🚨", "Fatal Crash Rate", f"{fatal_rate}%"),
        ("⚡", "Mean Risk Index", f"{risk_score:.3f}"),
    ]
    render_kpi_row(kpis)

st.markdown("<br>", unsafe_allow_html=True)


# ── 2. Analytical Deep-Dive Tabs ─────────────────────────────────────────────
tab_temporal, tab_geo, tab_env, tab_vehicles = st.tabs([
    "📈 Temporal & Hourly Trends",
    "🏙️ Regional & City Benchmarks",
    "🌦️ Environmental & Road Context",
    "🚗 Collision & Vehicle Dynamics"
])


# ── TAB 1: Temporal Trends ───────────────────────────────────────────────────
with tab_temporal:
    col1, col2 = st.columns([3, 2])
    with col1:
        section_header("Monthly Trajectory (2022–2025)", "📅")
        trend_sql = f"""
        SELECT 
            printf('%04d-%02d', year, month) AS timeline,
            COUNT(*) AS incidents,
            SUM(casualties) AS casualties
        FROM accidents_india
        WHERE {where_sql}
        GROUP BY year, month
        ORDER BY year ASC, month ASC;
        """
        df_trend = execute_query(trend_sql, params)
        if not df_trend.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_trend["timeline"],
                y=df_trend["incidents"],
                name="Incidents",
                line=dict(color="#FF4B4B", width=3),
                mode="lines+markers",
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_trend["timeline"],
                y=df_trend["casualties"],
                name="Casualties",
                line=dict(color="#FF8C42", width=2, dash="dash"),
                mode="lines",
            ))
            fig_trend.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            empty_state(
                "No matching incident records",
                "No incidents matched the current filter combination, so no monthly trajectory can be plotted.",
                hint="Try widening the City, State, Year, Severity, Weather or Road filters in the sidebar.",
                icon="📅",
            )

    with col2:
        section_header("Severity Proportion", "🎯")
        sev_sql = f"""
        SELECT accident_severity, COUNT(*) AS count
        FROM accidents_india
        WHERE {where_sql}
        GROUP BY accident_severity;
        """
        df_sev = execute_query(sev_sql, params)
        if not df_sev.empty:
            fig_sev = px.pie(
                df_sev,
                names="accident_severity",
                values="count",
                hole=0.6,
                color="accident_severity",
                color_discrete_map={"minor": "#00D4AA", "major": "#FFD600", "fatal": "#FF4B4B"},
            )
            fig_sev.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_sev, use_container_width=True)

    # Day-of-Week vs Hour Heatmap
    section_header("Crash Hour vs Day-of-Week Density Matrix", "🕒")
    heatmap_sql = f"""
    SELECT 
        day_of_week,
        hour,
        COUNT(*) AS incident_count
    FROM accidents_india
    WHERE {where_sql}
    GROUP BY day_of_week, hour;
    """
    df_heat = execute_query(heatmap_sql, params)
    if not df_heat.empty:
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot_heat = df_heat.pivot(index="day_of_week", columns="hour", values="incident_count").reindex(days_order).fillna(0)

        fig_heat = px.imshow(
            pivot_heat,
            labels=dict(x="Hour of Day (0–23)", y="Day of Week", color="Incidents"),
            x=[f"{h:02d}:00" for h in range(24)],
            y=days_order,
            color_continuous_scale="Reds",
            aspect="auto",
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ── TAB 2: Regional & City Benchmarks ────────────────────────────────────────
with tab_geo:
    col_c, col_s = st.columns(2)
    with col_c:
        section_header("City Incident & Fatality Comparison", "🏙️")
        df_cities = get_city_comparisons()
        if not df_cities.empty:
            fig_c = px.bar(
                df_cities,
                x="city",
                y="total_accidents",
                color="fatal_rate_pct",
                color_continuous_scale=["#00D4AA", "#FF8C42", "#FF4B4B"],
                labels={"total_accidents": "Accident Count", "city": "City", "fatal_rate_pct": "Fatal %"},
            )
            fig_c.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_c, use_container_width=True)

    with col_s:
        section_header("State Risk Index Summary", "🏛️")
        df_states = get_state_comparisons()
        if not df_states.empty:
            fig_s = px.bar(
                df_states,
                x="state",
                y="avg_risk_score",
                color="total_casualties",
                color_continuous_scale="Purples",
                labels={"state": "State", "avg_risk_score": "Mean Risk Score", "total_casualties": "Casualties"},
            )
            fig_s.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("#### Detailed Regional Benchmarks Table")
    st.dataframe(df_cities, use_container_width=True, hide_index=True)


# ── TAB 3: Environmental Context ─────────────────────────────────────────────
with tab_env:
    col_w, col_r = st.columns(2)
    with col_w:
        section_header("Weather Condition vs Severity", "🌦️")
        weather_sev_sql = f"""
        SELECT 
            weather,
            accident_severity,
            COUNT(*) AS count
        FROM accidents_india
        WHERE {where_sql}
        GROUP BY weather, accident_severity;
        """
        df_ws = execute_query(weather_sev_sql, params)
        if not df_ws.empty:
            fig_ws = px.bar(
                df_ws,
                x="weather",
                y="count",
                color="accident_severity",
                barmode="group",
                color_discrete_map={"minor": "#00D4AA", "major": "#FFD600", "fatal": "#FF4B4B"},
                labels={"weather": "Weather", "count": "Incidents", "accident_severity": "Severity"},
            )
            fig_ws.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_ws, use_container_width=True)

    with col_r:
        section_header("Road Type vs Mean Risk Score", "🛣️")
        road_risk_sql = f"""
        SELECT 
            road_type,
            COUNT(*) AS count,
            ROUND(AVG(risk_score), 3) AS mean_risk,
            SUM(casualties) AS total_casualties
        FROM accidents_india
        WHERE {where_sql}
        GROUP BY road_type;
        """
        df_rr = execute_query(road_risk_sql, params)
        if not df_rr.empty:
            fig_rr = px.bar(
                df_rr,
                x="road_type",
                y="mean_risk",
                color="total_casualties",
                color_continuous_scale=["#4DA8FF", "#FF8C42", "#FF4B4B"],
                labels={"road_type": "Road Classification", "mean_risk": "Mean Risk Score"},
            )
            fig_rr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_rr, use_container_width=True)


# ── TAB 4: Collision Dynamics ────────────────────────────────────────────────
with tab_vehicles:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        section_header("Vehicles Involved vs Average Casualties", "🚙")
        df_veh = get_vehicle_collision_breakdown()
        if not df_veh.empty:
            fig_v = px.line(
                df_veh,
                x="vehicles_involved",
                y="avg_casualties_per_accident",
                markers=True,
                labels={"vehicles_involved": "Vehicles Involved", "avg_casualties_per_accident": "Mean Casualties"},
            )
            fig_v.update_traces(line_color="#FF4B4B", line_width=3, marker=dict(size=8, color="#FFD600"))
            fig_v.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_v, use_container_width=True)

    with col_v2:
        section_header("Traffic Density Distribution", "🚦")
        dens_sql = f"""
        SELECT traffic_density, COUNT(*) AS count, SUM(casualties) AS casualties
        FROM accidents_india
        WHERE {where_sql}
        GROUP BY traffic_density;
        """
        df_dens = execute_query(dens_sql, params)
        if not df_dens.empty:
            fig_dens = px.bar(
                df_dens,
                x="traffic_density",
                y="count",
                color="traffic_density",
                color_discrete_sequence=["#00D4AA", "#FFD600", "#FF4B4B"],
                labels={"traffic_density": "Traffic Density", "count": "Incident Volume"},
            )
            fig_dens.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_dens, use_container_width=True)


# ── 3. Data Download & Limitations Disclaimer ────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Download Filtered Records", "📥")

download_sql = f"""
SELECT 
    accident_id, city, state, latitude, longitude, accident_date,
    road_type, weather, visibility, cause, accident_severity,
    vehicles_involved, casualties, risk_score
FROM accidents_india
WHERE {where_sql}
LIMIT 5000;
"""
df_filtered_export = execute_query(download_sql, params)

col_d1, col_d2 = st.columns([2, 3])
with col_d1:
    st.write(f"Filtered analytical slice contains **{len(df_filtered_export):,}** rows (capped at 5,000 for export).")
with col_d2:
    csv_bytes = df_filtered_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="roadrisk_filtered_analytics.csv",
        mime="text/csv",
    )

st.markdown("<br>", unsafe_allow_html=True)
info_card(
    "📌 <strong>Methodological Limitation:</strong> Observed accident volumes highlight recorded incidents. "
    "True danger rates require baseline exposure data (such as Average Daily Traffic Volume and lane mileage), "
    "which are absent from raw incident registries."
)
