"""
🌦️ Weather Analysis — Historical Weather & Accident Risk Intelligence
========================================================================
Module G: Open-Meteo Historical Weather API integration, SQLite caching,
WMO atmospheric classification, and empirical accident correlation analysis.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PAGE_CONFIG, DATABASE_PATH, PROCESSED_DIR
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    metric_card,
    info_card,
    status_badge,
    export_dataframe_ui,
    empty_state,
    error_state,
    service_status_row,
)
from src.database import get_db_connection
from src.weather_client import (
    fetch_historical_weather,
    get_incident_weather_match,
    compute_weather_risk_metrics,
    init_weather_cache,
    weather_service_status,
)

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# Initialize cache table
init_weather_cache(DATABASE_PATH)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #4DA8FF 0%, #A855F7 50%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    ">🌦️ Historical Weather & Atmospheric Risk Intelligence</h1>
    <p style="color: #8899A6; font-size: 0.95rem; line-height: 1.5;">
        Correlate accident occurrences with atmospheric conditions using the Open-Meteo Historical Archive API,
        local SQLite caching, and empirical weather-severity distribution analytics.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Weather Integration Status (secret-free) ─────────────────────────────────
service_status_row([weather_service_status(probe=False)])

# ── Load Accident Data ───────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_accident_weather_data() -> pd.DataFrame:
    """Load cleaned Indian accident data with weather and severity attributes."""
    try:
        conn = get_db_connection(DATABASE_PATH)
        query = """
        SELECT
            accident_id, city, state, latitude, longitude,
            accident_date, accident_time, hour, time_of_day,
            weather, visibility, temperature, accident_severity,
            vehicles_involved, casualties, road_type
        FROM accidents_india
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception:
        csv_path = PROCESSED_DIR / "indian_roads_cleaned.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()


df_accidents = load_accident_weather_data()

if df_accidents.empty:
    error_state(
        "Accident dataset unavailable",
        "The accident dataset could not be loaded from the SQLite database or the processed CSV files.",
        hint="Rebuild the database from the Home page, or run the preprocessing pipeline to regenerate the cleaned CSVs.",
        icon="🌦️",
    )
    st.stop()

# ── Compute Core Weather Metrics ─────────────────────────────────────────────
metrics = compute_weather_risk_metrics(df_accidents)

total_acc = metrics.get("total_accidents", len(df_accidents))
adv_comp = metrics.get("adverse_comparison", {})
adv_key = "Adverse Weather (Rain/Fog/Storm)"
norm_key = "Normal Weather (Clear/Cloudy)"

adv_stats = adv_comp.get(adv_key, {"pct_of_all": 0.0, "severe_combined_rate": 0.0})
norm_stats = adv_comp.get(norm_key, {"severe_combined_rate": 0.0})
rr = metrics.get("relative_severe_risk_ratio", 1.0)

# ── Top-Level KPI Ribbon ─────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Total Analyzed Incidents", f"{total_acc:,}", "Cleaned dataset records", border_color="#4DA8FF")
with k2:
    metric_card("Adverse Weather Share", f"{adv_stats.get('pct_of_all', 0):.1f}%", "Rain, Fog, or Storm", border_color="#A855F7")
with k3:
    metric_card(
        "Severe Rate (Adverse)",
        f"{adv_stats.get('severe_combined_rate', 0):.1f}%",
        f"vs {norm_stats.get('severe_combined_rate', 0):.1f}% Normal",
        border_color="#FF4B4B" if adv_stats.get('severe_combined_rate', 0) > norm_stats.get('severe_combined_rate', 0) else "#00C853"
    )
with k4:
    metric_card(
        "Severe Risk Ratio",
        f"{rr:.2f}x",
        "Adverse vs Clear condition rate",
        border_color="#FF8C42"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Tabbed Interface ────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Macro Weather Correlations",
    "🌐 Open-Meteo Historical Explorer",
    "🎯 Incident Atmospheric Re-enrichment",
    "⚠️ Scientific Governance & Exposure Bias",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Macro Weather Correlations
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("Atmospheric Conditions & Incident Distributions", "🌦️")

    # City filter
    all_cities = ["All Cities"] + sorted(df_accidents["city"].dropna().unique().tolist())
    selected_city = st.selectbox("Filter by City:", all_cities, index=0)

    if selected_city != "All Cities":
        df_filtered = df_accidents[df_accidents["city"] == selected_city].copy()
    else:
        df_filtered = df_accidents.copy()

    col_left, col_right = st.columns(2)

    with col_left:
        # Weather distribution
        w_counts = df_filtered["weather"].value_counts().reset_index()
        w_counts.columns = ["Weather Condition", "Incident Count"]
        w_counts["Weather Condition"] = w_counts["Weather Condition"].str.title()

        fig_w = px.bar(
            w_counts,
            x="Weather Condition",
            y="Incident Count",
            color="Weather Condition",
            color_discrete_sequence=["#4DA8FF", "#00C853", "#FF8C42", "#A855F7", "#FFD600"],
            title=f"Accident Volume by Weather Condition ({selected_city})",
            text="Incident Count",
        )
        fig_w.update_traces(textposition="outside")
        fig_w.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            height=350,
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with col_right:
        # Severity by Weather Stacked 100% Bar
        df_sev = df_filtered.copy()
        df_sev["weather_clean"] = df_sev["weather"].astype(str).str.title()
        df_sev["severity_clean"] = df_sev["accident_severity"].astype(str).str.title()

        sev_cross = pd.crosstab(
            df_sev["weather_clean"], df_sev["severity_clean"], normalize="index"
        ) * 100
        sev_cross = sev_cross.reset_index().melt(
            id_vars="weather_clean", var_name="Severity", value_name="Percentage"
        )

        sev_colors = {"Fatal": "#FF4B4B", "Serious": "#FF8C42", "Minor": "#00C853"}
        fig_sev = px.bar(
            sev_cross,
            x="weather_clean",
            y="Percentage",
            color="Severity",
            color_discrete_map=sev_colors,
            title=f"Severity Distribution by Weather (%) ({selected_city})",
            text="Percentage",
        )
        fig_sev.update_traces(texttemplate="%{y:.1f}%", textposition="inside")
        fig_sev.update_layout(
            barmode="stack",
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            yaxis=dict(title="Percentage (%)", range=[0, 100]),
            xaxis=dict(title="Weather Condition"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    # Temperature & Visibility Distributions
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        # Temperature Box Plot across Severity
        if "temperature" in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered["temperature"]):
            fig_temp = px.box(
                df_filtered,
                x="accident_severity",
                y="temperature",
                color="accident_severity",
                color_discrete_map={"Fatal": "#FF4B4B", "Serious": "#FF8C42", "Minor": "#00C853"},
                title="Ambient Temperature (°C) vs Accident Severity",
                points=False,
            )
            fig_temp.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                showlegend=False,
                yaxis=dict(title="Temperature (°C)"),
                xaxis=dict(title="Severity"),
                margin=dict(l=20, r=20, t=50, b=20),
                height=340,
            )
            st.plotly_chart(fig_temp, use_container_width=True)

    with col_t2:
        # Visibility Breakdown
        if "visibility" in df_filtered.columns:
            vis_counts = df_filtered.groupby(["visibility", "accident_severity"]).size().reset_index(name="Count")
            vis_counts["visibility"] = vis_counts["visibility"].astype(str).str.title()
            vis_counts["accident_severity"] = vis_counts["accident_severity"].astype(str).str.title()

            fig_vis = px.bar(
                vis_counts,
                x="visibility",
                y="Count",
                color="accident_severity",
                color_discrete_map={"Fatal": "#FF4B4B", "Serious": "#FF8C42", "Minor": "#00C853"},
                title="Visibility Level vs Incident Counts",
                barmode="group",
            )
            fig_vis.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                xaxis=dict(title="Visibility"),
                yaxis=dict(title="Count"),
                margin=dict(l=20, r=20, t=50, b=20),
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_vis, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Open-Meteo Historical Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Live Open-Meteo Historical Archive API Explorer", "🌐")
    st.markdown("""
    Query real historical atmospheric observations from the **Open-Meteo Historical Weather API**.
    Responses are automatically cached in the local SQLite database (`weather_cache` table) with spatial and temporal keying.
    """)

    preset_cities = {
        "Mumbai, Maharashtra": (19.0760, 72.8777),
        "Delhi (NCR)": (28.6139, 77.2090),
        "Bengaluru, Karnataka": (12.9716, 77.5946),
        "Kolkata, West Bengal": (22.5726, 88.3639),
        "Chennai, Tamil Nadu": (13.0827, 80.2707),
        "Hyderabad, Telangana": (17.3850, 78.4867),
        "Pune, Maharashtra": (18.5204, 73.8567),
    }

    c_city, c_date, c_action = st.columns([2, 1, 1])

    with c_city:
        chosen_preset = st.selectbox("Select Target Urban Center:", list(preset_cities.keys()), index=0)
        preset_lat, preset_lon = preset_cities[chosen_preset]

    with c_date:
        query_date = st.date_input("Historical Date:", value=date(2024, 7, 15), min_value=date(2020, 1, 1), max_value=date(2025, 4, 1))

    with c_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        fetch_btn = st.button("⚡ Query Atmospheric Profile", type="primary", use_container_width=True)

    date_str = query_date.strftime("%Y-%m-%d")

    # Fetch data (or use cached sample)
    if fetch_btn or "last_weather_df" not in st.session_state:
        with st.spinner(f"Querying Open-Meteo Historical API for {chosen_preset} on {date_str}..."):
            df_weather_day = fetch_historical_weather(
                lat=preset_lat,
                lon=preset_lon,
                start_date=date_str,
                end_date=date_str,
                use_cache=True,
                db_path=DATABASE_PATH,
            )
            st.session_state["last_weather_df"] = df_weather_day
            st.session_state["last_weather_location"] = chosen_preset
            st.session_state["last_weather_date"] = date_str
    else:
        df_weather_day = st.session_state.get("last_weather_df", pd.DataFrame())

    if not df_weather_day.empty:
        source_label = df_weather_day.iloc[0].get("source", "SQLite Cache")
        st.markdown(f"""
        <div style="
            background: #161B22; border: 1px solid #30363D; border-radius: 8px;
            padding: 0.8rem 1.2rem; margin-bottom: 1rem; display: flex;
            justify-content: space-between; align-items: center;
        ">
            <div>
                <strong>📍 {st.session_state.get('last_weather_location', chosen_preset)}</strong> •
                <code>{st.session_state.get('last_weather_date', date_str)}</code> •
                24 Hourly Observations
            </div>
            <div>
                {status_badge(source_label, 'ready' if 'Open-Meteo' in source_label else 'info')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dual-Axis Plotly: Temperature & Precipitation
        fig_profile = go.Figure()

        # Temperature line
        fig_profile.add_trace(go.Scatter(
            x=df_weather_day["hour"],
            y=df_weather_day["temperature_2m"],
            name="Temperature (°C)",
            mode="lines+markers",
            line=dict(color="#FF8C42", width=3),
            marker=dict(size=6),
            yaxis="y1",
        ))

        # Precipitation bar
        fig_profile.add_trace(go.Bar(
            x=df_weather_day["hour"],
            y=df_weather_day["precipitation"],
            name="Precipitation (mm)",
            marker=dict(color="#4DA8FF", opacity=0.6),
            yaxis="y2",
        ))

        # Dual Y-axis layout
        fig_profile.update_layout(
            title="24-Hour Diurnal Temperature (°C) & Precipitation (mm) Progression",
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            xaxis=dict(title="Hour of Day (00:00 - 23:00)", dtick=1),
            yaxis=dict(title="Temperature (°C)", side="left", showgrid=True),
            yaxis2=dict(
                title="Precipitation (mm)",
                side="right",
                overlaying="y",
                showgrid=False,
                rangemode="tozero",
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_profile, use_container_width=True)

        # Atmospheric Conditions Strip / Table
        with st.expander("📋 View Full 24-Hour Atmospheric Observation Table", expanded=False):
            display_cols = [
                "hour", "temperature_2m", "relative_humidity_2m",
                "precipitation", "wind_speed_10m", "wmo_emoji", "wmo_description", "is_adverse"
            ]
            valid_cols = [c for c in display_cols if c in df_weather_day.columns]
            st.dataframe(df_weather_day[valid_cols], use_container_width=True)
            export_dataframe_ui(df_weather_day[valid_cols], f"weather_{chosen_preset}_{date_str}")
    else:
        empty_state(
            "No atmospheric data returned",
            f"Open-Meteo returned no hourly observations for {chosen_preset} on {date_str}.",
            hint="Check the selected date is within the supported historical range, then retry the query.",
            icon="🌤️",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Incident Atmospheric Re-enrichment
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Single-Incident Atmospheric Re-enrichment", "🎯")
    st.markdown("""
    Select any real crash record from the database. RoadRisk AI will query the historical
    Open-Meteo meteorological database for the **exact GPS coordinates, calendar date, and hour**
    of the crash to cross-validate dataset attributes against verified reanalysis observations.
    """)

    sample_incidents = df_accidents.head(50).copy()
    sample_options = {
        row["accident_id"]: f"ID #{row['accident_id']} — {row['city']} ({row['accident_date']} {row['accident_time']}) [{row['accident_severity']}]"
        for _, row in sample_incidents.iterrows()
    }

    selected_id = st.selectbox(
        "Choose an incident to enrich:",
        options=list(sample_options.keys()),
        format_func=lambda x: sample_options[x],
    )

    incident_row = df_accidents[df_accidents["accident_id"] == selected_id].iloc[0]

    i_col1, i_col2 = st.columns(2)

    with i_col1:
        st.markdown(f"""
        <div class="info-card" style="border-left-color: #4DA8FF;">
            <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem;">
                📄 Dataset Recorded Incident Dossier
            </div>
            <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                <tr><td style="color: #8899A6; padding: 4px 0;">Accident ID:</td><td><strong>#{incident_row['accident_id']}</strong></td></tr>
                <tr><td style="color: #8899A6; padding: 4px 0;">Location:</td><td>{incident_row['city']}, {incident_row['state']}</td></tr>
                <tr><td style="color: #8899A6; padding: 4px 0;">Coordinates:</td><td><code>{incident_row['latitude']:.4f}, {incident_row['longitude']:.4f}</code></td></tr>
                <tr><td style="color: #8899A6; padding: 4px 0;">Date & Time:</td><td>{incident_row['accident_date']} at {incident_row['accident_time']}</td></tr>
                <tr><td style="color: #8899A6; padding: 4px 0;">Reported Weather:</td><td><strong>{str(incident_row['weather']).title()}</strong></td></tr>
                <tr><td style="color: #8899A6; padding: 4px 0;">Reported Temp:</td><td>{incident_row['temperature']}°C</td></tr>
                <tr><td style="color: #8899A6; padding: 4px 0;">Reported Visibility:</td><td>{str(incident_row['visibility']).title()}</td></tr>
                <tr><td style="color: #8899A6; padding: 4px 0;">Severity Outcome:</td><td><span style="color: {'#FF4B4B' if incident_row['accident_severity']=='Fatal' else '#FF8C42'}; font-weight:700;">{incident_row['accident_severity']}</span> ({incident_row['casualties']} casualties)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with i_col2:
        # Perform historical lookup
        with st.spinner("Aligning exact historical atmospheric reanalysis..."):
            weather_match = get_incident_weather_match(
                lat=incident_row["latitude"],
                lon=incident_row["longitude"],
                date_str=str(incident_row["accident_date"]),
                hour=int(incident_row["hour"]),
                use_cache=True,
                db_path=DATABASE_PATH,
            )

        if weather_match:
            is_adv = weather_match.get("is_adverse", 0) == 1
            st.markdown(f"""
            <div class="info-card" style="border-left-color: {'#FF4B4B' if is_adv else '#00C853'};">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem;">
                    🌐 Open-Meteo Reanalysis Match ({weather_match.get('wmo_emoji', '🌤️')})
                </div>
                <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                    <tr><td style="color: #8899A6; padding: 4px 0;">WMO Condition:</td><td><strong>{weather_match.get('wmo_description', 'N/A')}</strong></td></tr>
                    <tr><td style="color: #8899A6; padding: 4px 0;">Adverse Hazard:</td><td>{'⚠️ Yes (Hazardous)' if is_adv else '✓ No (Benign)'}</td></tr>
                    <tr><td style="color: #8899A6; padding: 4px 0;">Reanalysis Temp:</td><td><strong>{weather_match.get('temperature_2m', 'N/A')}°C</strong></td></tr>
                    <tr><td style="color: #8899A6; padding: 4px 0;">Precipitation:</td><td>{weather_match.get('precipitation', 0.0)} mm/hr</td></tr>
                    <tr><td style="color: #8899A6; padding: 4px 0;">Relative Humidity:</td><td>{weather_match.get('relative_humidity_2m', 'N/A')}%</td></tr>
                    <tr><td style="color: #8899A6; padding: 4px 0;">Wind Speed:</td><td>{weather_match.get('wind_speed_10m', 'N/A')} km/h</td></tr>
                    <tr><td style="color: #8899A6; padding: 4px 0;">Cache Status:</td><td><code>{weather_match.get('cache_key', 'Live')}</code></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            error_state(
                "No historical weather match",
                "No archived Open-Meteo reanalysis record could be aligned to this incident's location and time.",
                hint="The historical archive may not cover this date, or the service may be temporarily unreachable. The rest of the page is unaffected.",
                icon="🌐",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Scientific Governance & Confounding Variables
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("The Danger of Naive Correlation: Confounding & Exposure Bias", "⚠️")

    st.markdown("""
    ### Why Clear Weather Has More Total Accidents than Storms: The Exposure Fallacy

    In empirical crash databases, a classic trap for novice data scientists is observing that **60–80% of all accidents occur in clear, dry weather**,
    and naively concluding that *"clear weather is more dangerous than heavy rain or fog."*

    This is an example of **Exposure Bias** and **Confounder Omission**:
    """)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("""
        <div class="kpi-card" style="text-align: left; min-height: 220px; border-left: 4px solid #FF8C42;">
            <div style="font-weight: 700; color: #FF8C42; font-size: 1rem; margin-bottom: 0.5rem;">
                🚗 1. Vehicle-Kilometers Traveled (VKT)
            </div>
            <p style="font-size: 0.85rem; color: #8899A6; line-height: 1.6;">
                Over 90% of all annual vehicle trips take place in benign weather. When torrential rain or dense fog strikes,
                drivers cancel discretionary trips, slow down, or take alternate routes.
                <br><br>
                Because exposure (vehicle-kilometers traveled) plummets during extreme storms, the <em>absolute count</em> of crashes drops,
                even while the <strong>per-kilometer crash probability</strong> spikes drastically.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("""
        <div class="kpi-card" style="text-align: left; min-height: 220px; border-left: 4px solid #4DA8FF;">
            <div style="font-weight: 700; color: #4DA8FF; font-size: 1rem; margin-bottom: 0.5rem;">
                🛡️ 2. Risk Compensation (Peltzman Effect)
            </div>
            <p style="font-size: 0.85rem; color: #8899A6; line-height: 1.6;">
                In dense fog or monsoon downpours, drivers perceive heightened danger and instinctively reduce speed,
                increase headway, and turn on hazard lamps.
                <br><br>
                Conversely, on straight, dry, sunlit highways, drivers experience higher confidence, leading to excessive speeding,
                tailgating, and higher kinetic energy crashes when an unexpected collision does occur.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    info_card(
        "📚 <strong>Academic Methodology Takeaway:</strong><br>"
        "To establish true causal safety risk, road safety engineers use the <strong>Accident Exposure Rate Formula</strong>:<br><br>"
        "$$\\text{Crash Risk Rate} = \\frac{\\text{Reported Crashes in Condition } W}{\\text{Vehicle Kilometers Traveled (VKT) in Condition } W}$$<br>"
        "Without continuous AADT (Annual Average Daily Traffic) sensor counts, raw accident counts should <em>never</em> be used "
        "as a direct surrogate for road safety risk."
    )
