"""
📊 Dashboard — Interactive Analytics & KPI Overview
=====================================================
Module A: Displays interactive summary statistics, temporal trends,
city breakdowns, and atmospheric associations from the database.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.config import PAGE_CONFIG, COLORS
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    render_kpi_row,
    info_card,
)
from src.database import execute_query, DATABASE_PATH, load_data_to_database
from src.sql_queries import get_overall_kpis, get_monthly_trends

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Ensure Database is Initialized ───────────────────────────────────────────
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
    Exploratory intelligence, trend trajectories, and risk patterns across 20,000 audited Indian road incidents.
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Analytics Filters")

    # City filter
    cities_df = execute_query("SELECT DISTINCT city FROM accidents_india ORDER BY city ASC;")
    city_options = ["All"] + cities_df["city"].tolist()
    selected_city = st.selectbox("Select City:", city_options, index=0)

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

    st.markdown("---")
    st.caption("Filters update all dashboard KPIs and visualizations in real time.")


# ── Filtered Query Construction ──────────────────────────────────────────────
where_clauses = ["1=1"]
params = []

if selected_city != "All":
    where_clauses.append("city = ?")
    params.append(selected_city)
if selected_year != "All":
    where_clauses.append("year = ?")
    params.append(int(selected_year))
if selected_severity != "All":
    where_clauses.append("accident_severity = ?")
    params.append(selected_severity.lower())
if selected_weather != "All":
    where_clauses.append("weather = ?")
    params.append(selected_weather.lower())

where_sql = " AND ".join(where_clauses)


# ── 1. KPI Cards ─────────────────────────────────────────────────────────────
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


# ── 2. Temporal Trends & Severity Breakdown ──────────────────────────────────
col_trend, col_sev = st.columns([3, 2])

with col_trend:
    section_header("Monthly Accident & Casualty Trajectory", "📈")
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
            name="Accidents",
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
        st.info("No records matching the selected filter criteria.")

with col_sev:
    section_header("Severity Share", "🎯")
    sev_sql = f"""
    SELECT 
        accident_severity,
        COUNT(*) AS count
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
            hole=0.55,
            color="accident_severity",
            color_discrete_map={
                "minor": "#00D4AA",
                "major": "#FFD600",
                "fatal": "#FF4B4B",
            },
        )
        fig_sev.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_sev, use_container_width=True)


# ── 3. City Breakdown & Weather Impact ───────────────────────────────────────
col_city, col_cause = st.columns(2)

with col_city:
    section_header("Incident Density Across Cities", "🏙️")
    city_sql = f"""
    SELECT 
        city,
        COUNT(*) AS incidents,
        SUM(CASE WHEN accident_severity = 'fatal' THEN 1 ELSE 0 END) AS fatal_count
    FROM accidents_india
    WHERE {where_sql}
    GROUP BY city
    ORDER BY incidents ASC;
    """
    df_city = execute_query(city_sql, params)
    if not df_city.empty:
        fig_city = px.bar(
            df_city,
            x="incidents",
            y="city",
            orientation="h",
            color="fatal_count",
            color_continuous_scale=["#00D4AA", "#FF8C42", "#FF4B4B"],
            labels={"incidents": "Incident Count", "city": "City", "fatal_count": "Fatal Count"},
        )
        fig_city.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_city, use_container_width=True)

with col_cause:
    section_header("Reported Primary Causes", "⚠️")
    cause_sql = f"""
    SELECT 
        cause,
        COUNT(*) AS count
    FROM accidents_india
    WHERE {where_sql}
    GROUP BY cause
    ORDER BY count DESC;
    """
    df_cause = execute_query(cause_sql, params)
    if not df_cause.empty:
        fig_cause = px.bar(
            df_cause,
            x="cause",
            y="count",
            color="cause",
            color_discrete_sequence=["#FF4B4B", "#FF8C42", "#FFD600", "#4DA8FF", "#A855F7"],
            labels={"cause": "Cause", "count": "Incident Count"},
        )
        fig_cause.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_cause, use_container_width=True)


# ── 4. Filtered Data Download ────────────────────────────────────────────────
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

col1, col2 = st.columns([2, 3])
with col1:
    st.write(f"Filtered subset contains **{len(df_filtered_export):,}** rows (capped at 5,000 for export).")
with col2:
    csv_bytes = df_filtered_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="roadrisk_filtered_analytics.csv",
        mime="text/csv",
    )

st.markdown("<br>", unsafe_allow_html=True)
info_card(
    "📌 <strong>Analytical Guidance:</strong> Observed accident frequencies reflect recorded incidents. "
    "To establish genuine risk indices, localized road exposure metrics (Average Annual Daily Traffic) "
    "are required alongside historical incidents."
)
