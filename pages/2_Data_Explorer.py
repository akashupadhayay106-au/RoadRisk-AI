"""
🔍 Accident Explorer & Data Quality Studio
===========================================
Module B & D: Deep multi-dataset inspection, faceted search,
single-record dossier inspector, pagination, and data export.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import PAGE_CONFIG, DATASETS, RAW_DATA_DIR, PROCESSED_DIR
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    render_kpi_row,
    info_card,
    load_csv_cached,
)
from src.database import execute_query, DATABASE_PATH, load_data_to_database

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Title & Header ───────────────────────────────────────────────────────────
st.markdown("""
<h1 style="
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #4DA8FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🔍 Accident Explorer & Record Inspector</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Faceted query filters, schema audits, single-incident dossier inspector, and certified dataset exports.
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Ensure Database is Loaded ────────────────────────────────────────────────
if not DATABASE_PATH.exists():
    with st.spinner("Preparing analytical database…"):
        load_data_to_database()

# ── Dataset Selector ─────────────────────────────────────────────────────────
DATASET_OPTIONS = {
    "1. Indian Road Accidents (Cleaned & Indexed)": "accidents_india",
    "2. Highway Corridor Incidents (OpenCity)": "highway_accidents_opencity",
    "3. ML Severity Dataset (Train Set)": "severity_train",
}

section_header("Choose Source Dataset", "📂")
selected_dataset_label = st.selectbox(
    "Select dataset to audit and explore:",
    list(DATASET_OPTIONS.keys()),
    index=0,
)
selected_dataset_key = DATASET_OPTIONS[selected_dataset_label]

# Load active dataframe
if selected_dataset_key in ["accidents_india", "highway_accidents_opencity"]:
    df_raw = execute_query(f"SELECT * FROM {selected_dataset_key};")
else:
    csv_path = PROCESSED_DIR / "severity_train_cleaned.csv"
    if not csv_path.exists():
        csv_path = RAW_DATA_DIR / "train.csv"
    df_raw = load_csv_cached(str(csv_path))

if df_raw is None or df_raw.empty:
    st.warning("Selected dataset could not be loaded. Please ensure preprocessing has run.")
    st.stop()


# ── Active Dataset KPIs ──────────────────────────────────────────────────────
render_kpi_row([
    ("📋", "Total Rows", len(df_raw)),
    ("📊", "Columns", len(df_raw.columns)),
    ("❌", "Missing Cells", int(df_raw.isnull().sum().sum())),
    ("🔁", "Duplicate Rows", int(df_raw.duplicated().sum())),
])
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs: Explorer, Record Dossier, Schema Audit ──────────────────────────────
tab_explorer, tab_dossier, tab_schema = st.tabs([
    "🔎 Faceted Search & Explorer",
    "📋 Single Incident Dossier",
    "🩺 Schema & Data Quality Audit"
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Faceted Search & Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    section_header("Filter & Query Records", "⚙️")

    df_filtered = df_raw.copy()

    # Dynamic filters for Indian accidents
    if selected_dataset_key == "accidents_india":
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            cities = ["All"] + sorted(df_raw["city"].unique().tolist())
            f_city = st.selectbox("Filter City:", cities, key="exp_city")
            if f_city != "All":
                df_filtered = df_filtered[df_filtered["city"] == f_city]
        with col_f2:
            severities = ["All"] + sorted(df_raw["accident_severity"].unique().tolist())
            f_sev = st.selectbox("Filter Severity:", severities, key="exp_sev")
            if f_sev != "All":
                df_filtered = df_filtered[df_filtered["accident_severity"] == f_sev]
        with col_f3:
            weathers = ["All"] + sorted(df_raw["weather"].unique().tolist())
            f_weather = st.selectbox("Filter Weather:", weathers, key="exp_weather")
            if f_weather != "All":
                df_filtered = df_filtered[df_filtered["weather"] == f_weather]
        with col_f4:
            roads = ["All"] + sorted(df_raw["road_type"].unique().tolist())
            f_road = st.selectbox("Filter Road Type:", roads, key="exp_road")
            if f_road != "All":
                df_filtered = df_filtered[df_filtered["road_type"] == f_road]

    # Full text keyword search
    search_query = st.text_input(
        "🔍 Full-text search across all attributes:",
        placeholder="Type state, road type, cause, or keyword…",
    )
    if search_query:
        mask = df_filtered.astype(str).apply(
            lambda col: col.str.contains(search_query, case=False, na=False)
        ).any(axis=1)
        df_filtered = df_filtered[mask]

    st.markdown(f"**Found {len(df_filtered):,} matching incident records** (out of {len(df_raw):,}).")

    # Column Visibility
    with st.expander("👁️ Customize Visible Columns", expanded=False):
        all_cols = df_filtered.columns.tolist()
        default_cols = all_cols[:10]
        visible_cols = st.multiselect("Select columns to display in table:", all_cols, default=default_cols)
        if visible_cols:
            df_filtered_view = df_filtered[visible_cols]
        else:
            df_filtered_view = df_filtered

    # Pagination Controls
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        page_size = st.selectbox("Records per page:", [20, 50, 100, 250], index=0)
    with col_p2:
        total_pages = max(1, (len(df_filtered) - 1) // page_size + 1)
        page_num = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    st.dataframe(
        df_filtered_view.iloc[start_idx:end_idx],
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing rows {start_idx + 1}–{min(end_idx, len(df_filtered))} of {len(df_filtered):,}")

    # Export Download Button
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Slice ({len(df_filtered):,} rows as CSV)",
        data=csv_bytes,
        file_name=f"{selected_dataset_key}_filtered.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Single Incident Dossier
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dossier:
    section_header("Single Incident Dossier Inspector", "📋")
    
    if selected_dataset_key == "accidents_india":
        available_ids = df_filtered["accident_id"].tolist()[:100]
        if available_ids:
            sel_id = st.selectbox("Select Accident ID to inspect:", available_ids)
            rec = df_raw[df_raw["accident_id"] == sel_id].iloc[0].to_dict()

            # Dossier Card
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.markdown("#### 📍 Geospatial & Location")
                st.write(f"**City:** {rec.get('city')}")
                st.write(f"**State:** {rec.get('state')}")
                st.write(f"**Coordinates:** `{rec.get('latitude'):.5f}, {rec.get('longitude'):.5f}`")
                st.write(f"**Road Type:** {rec.get('road_type')}")
                st.write(f"**Lanes:** {rec.get('lanes')}")
                st.write(f"**Traffic Signal:** {'Yes' if rec.get('traffic_signal') == 1 else 'No'}")

            with col_d2:
                st.markdown("#### 🕒 Temporal & Atmospheric")
                st.write(f"**Date:** {rec.get('accident_date')}")
                st.write(f"**Time:** {rec.get('accident_time')} ({rec.get('time_of_day')})")
                st.write(f"**Day of Week:** {rec.get('day_of_week')}")
                st.write(f"**Weekend Crash:** {'Yes' if rec.get('is_weekend') == 1 else 'No'}")
                st.write(f"**Weather:** {rec.get('weather')}")
                st.write(f"**Visibility:** {rec.get('visibility')}")
                st.write(f"**Temperature:** {rec.get('temperature')}°C")

            with col_d3:
                st.markdown("#### ⚠️ Impact & Severity Outcomes")
                sev_color = "#FF4B4B" if rec.get('accident_severity') == 'fatal' else ("#FFD600" if rec.get('accident_severity') == 'major' else "#00D4AA")
                st.markdown(f"**Severity Level:** <span style='color: {sev_color}; font-weight: 700; text-transform: uppercase;'>{rec.get('accident_severity')}</span>", unsafe_allow_html=True)
                st.write(f"**Vehicles Involved:** {rec.get('vehicles_involved')}")
                st.write(f"**Casualties Reported:** {rec.get('casualties')}")
                st.write(f"**Traffic Density:** {rec.get('traffic_density')}")
                st.write(f"**Reported Cause:** {rec.get('cause')}")
                st.write(f"**Risk Score Index:** `{rec.get('risk_score')}`")
        else:
            st.info("No records in current filter selection.")
    else:
        st.info("Dossier inspector is currently configured for the primary Indian Road Accidents dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Schema & Data Quality Audit
# ═══════════════════════════════════════════════════════════════════════════════
with tab_schema:
    section_header("Schema & Completeness Audit", "🩺")

    schema_df = pd.DataFrame({
        "Column Name": df_raw.columns,
        "Data Type": df_raw.dtypes.astype(str).values,
        "Non-Null Count": df_raw.notna().sum().values,
        "Missing Cells": df_raw.isna().sum().values,
        "Null Percentage": (df_raw.isna().sum().values / len(df_raw) * 100).round(2),
        "Unique Entries": [df_raw[c].nunique() for c in df_raw.columns],
        "Sample Value": [str(df_raw[c].dropna().iloc[0])[:60] if df_raw[c].notna().any() else "None" for c in df_raw.columns],
    })
    st.dataframe(schema_df, use_container_width=True, hide_index=True, height=450)
