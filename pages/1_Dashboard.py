"""
📊 Dashboard — Interactive Analytics & KPI Overview
=====================================================
Module A: Displays summary statistics, trends, and breakdowns
from the loaded accident datasets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.config import PAGE_CONFIG, COLORS, PROJECT_ROOT, RAW_DATA_DIR
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    render_kpi_row,
    info_card,
    load_csv_cached,
)

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Page Title ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #FF4B4B, #FF8C42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">📊 Analytics Dashboard</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Explore accident statistics, trends, and breakdowns from the loaded datasets.
</p>
""", unsafe_allow_html=True)
st.divider()


# ── Load Data ────────────────────────────────────────────────────────────────

def _find_csv(name: str) -> Path | None:
    """Look in raw/ then project root."""
    for parent in [RAW_DATA_DIR, PROJECT_ROOT]:
        p = parent / name
        if p.exists():
            return p
    return None

indian_path = _find_csv("indian_roads_dataset.csv")

if indian_path is None:
    st.warning(
        "⚠️ **Indian Road Accident Dataset not found.**\n\n"
        "Please download it from "
        "[Kaggle](https://www.kaggle.com/datasets/sehaj1104/indian-road-accident-dataset-20222025) "
        "and place `indian_roads_dataset.csv` in the `data/raw/` folder.\n\n"
        "The dashboard will populate once data is available."
    )
    st.stop()

df = load_csv_cached(str(indian_path))

if df is None or df.empty:
    st.error("Dataset loaded but appears empty. Please check the file.")
    st.stop()

# ── Show Available Columns (temporary — useful for Phase 2 inspection) ───────
with st.expander("🔎 Raw Column Inspection (for development)"):
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.write("**Columns:**", list(df.columns))
    st.write("**Data types:**")
    st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}))
    st.write("**First 5 rows:**")
    st.dataframe(df.head())

# ── Basic KPIs ───────────────────────────────────────────────────────────────
section_header("Key Metrics", "📈")

total_records = len(df)
num_columns = len(df.columns)

# Try to detect common column names (we'll refine in Phase 2)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
kpis = [
    ("📋", "Total Records", total_records),
    ("📊", "Features", num_columns),
    ("🔢", "Numeric Columns", len(numeric_cols)),
    ("📝", "Text Columns", num_columns - len(numeric_cols)),
]
render_kpi_row(kpis)

st.markdown("<br>", unsafe_allow_html=True)

# ── Column Value Previews ────────────────────────────────────────────────────
section_header("Column Summaries", "📋")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Numeric Column Statistics**")
    st.dataframe(df.describe().T.style.format("{:.1f}"), use_container_width=True)

with col2:
    st.markdown("**Categorical Column Unique Counts**")
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        cat_info = pd.DataFrame({
            "Column": cat_cols,
            "Unique Values": [df[c].nunique() for c in cat_cols],
            "Missing": [df[c].isna().sum() for c in cat_cols],
            "Sample Value": [df[c].dropna().iloc[0] if df[c].notna().any() else "N/A" for c in cat_cols],
        })
        st.dataframe(cat_info, use_container_width=True, hide_index=True)
    else:
        st.info("No categorical columns detected.")

# ── Missing Values ───────────────────────────────────────────────────────────
section_header("Missing Values Overview", "🔍")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values, "Pct (%)": missing_pct.values})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)

if missing_df.empty:
    st.success("✅ No missing values in the dataset!")
else:
    fig = px.bar(
        missing_df,
        x="Column", y="Pct (%)",
        title="Missing Values by Column",
        color="Pct (%)",
        color_continuous_scale=["#00D4AA", "#FFD600", "#FF4B4B"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Note ─────────────────────────────────────────────────────────────────────
info_card(
    "📌 <strong>Note:</strong> This is the Phase 1 skeleton dashboard. "
    "Full analytics (trends, severity breakdowns, filters, and geographic views) "
    "will be added in Phase 3–4 after data cleaning and schema validation."
)
