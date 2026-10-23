"""
🔍 Data Explorer — Inspect, Filter & Download Datasets
========================================================
Module B & D: Data quality checks, schema inspection,
search/filter, and downloadable results.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import PAGE_CONFIG, DATASETS, PROJECT_ROOT, RAW_DATA_DIR
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
    background: linear-gradient(135deg, #4DA8FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🔍 Data Explorer</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Inspect schemas, check data quality, filter records, and download results.
</p>
""", unsafe_allow_html=True)
st.divider()


# ── Dataset Selection ────────────────────────────────────────────────────────
def discover_datasets() -> dict[str, Path]:
    """Find all CSV/Excel files in raw/ and project root."""
    found = {}
    for search_dir in [RAW_DATA_DIR, PROJECT_ROOT]:
        if search_dir.exists():
            for f in search_dir.iterdir():
                if f.suffix.lower() in {".csv", ".xlsx", ".xls"} and f.name not in found:
                    found[f.name] = f
    return found

available = discover_datasets()

if not available:
    st.warning(
        "⚠️ No datasets found. Place CSV or Excel files in `data/raw/` to begin."
    )
    st.stop()

section_header("Select Dataset", "📂")
selected_name = st.selectbox(
    "Choose a dataset to explore:",
    options=list(available.keys()),
    help="All CSV/Excel files found in data/raw/ and the project root.",
)

selected_path = available[selected_name]

# ── Load ─────────────────────────────────────────────────────────────────────
if selected_path.suffix.lower() == ".csv":
    df = load_csv_cached(str(selected_path))
else:
    try:
        df = pd.read_excel(selected_path)
    except Exception as e:
        st.error(f"Failed to read {selected_name}: {e}")
        st.stop()

if df is None or df.empty:
    st.error("File is empty or could not be read.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────
render_kpi_row([
    ("📋", "Rows", len(df)),
    ("📊", "Columns", len(df.columns)),
    ("❌", "Missing Cells", int(df.isnull().sum().sum())),
    ("🔁", "Duplicate Rows", int(df.duplicated().sum())),
])
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_schema, tab_quality, tab_filter, tab_download = st.tabs([
    "📐 Schema", "🩺 Data Quality", "🔎 Filter & Search", "📥 Download"
])

# --- Schema Tab ---
with tab_schema:
    section_header("Column Schema", "📐")
    schema_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notna().sum().values,
        "Null Count": df.isna().sum().values,
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Sample": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else "N/A" for c in df.columns],
    })
    st.dataframe(schema_df, use_container_width=True, hide_index=True, height=400)

# --- Quality Tab ---
with tab_quality:
    section_header("Data Quality Report", "🩺")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Missing Values per Column**")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values!")
        else:
            st.dataframe(
                pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                              "% Missing": (missing.values / len(df) * 100).round(2)}),
                use_container_width=True, hide_index=True,
            )

    with col2:
        st.markdown("**Duplicate Rows**")
        dup_count = df.duplicated().sum()
        if dup_count == 0:
            st.success("No duplicate rows!")
        else:
            st.warning(f"{dup_count:,} duplicate rows found ({dup_count/len(df)*100:.1f}%)")
            if st.checkbox("Show duplicate rows"):
                st.dataframe(df[df.duplicated(keep="first")].head(20), use_container_width=True)

    st.markdown("**Numeric Column Statistics**")
    num_desc = df.describe()
    if not num_desc.empty:
        st.dataframe(num_desc.T.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("No numeric columns in this dataset.")

# --- Filter Tab ---
with tab_filter:
    section_header("Filter & Search", "🔎")

    # Column filter
    filter_cols = st.multiselect("Select columns to display:", df.columns.tolist(), default=df.columns.tolist()[:8])

    # Row filter
    st.markdown("**Apply row filters:**")
    filtered_df = df[filter_cols].copy() if filter_cols else df.copy()

    # Simple text search
    search_term = st.text_input("🔍 Search across all columns:", placeholder="Type to search…")
    if search_term:
        mask = filtered_df.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered_df = filtered_df[mask]
        st.info(f"Found {len(filtered_df):,} matching rows.")

    # Pagination
    page_size = st.selectbox("Rows per page:", [25, 50, 100, 250], index=0)
    total_pages = max(1, (len(filtered_df) - 1) // page_size + 1)
    page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True, height=400)
    st.caption(f"Showing rows {start_idx+1}–{min(end_idx, len(filtered_df))} of {len(filtered_df):,}")

# --- Download Tab ---
with tab_download:
    section_header("Download Data", "📥")
    info_card("Download the currently loaded dataset or a filtered version as CSV.")

    col1, col2 = st.columns(2)
    with col1:
        csv_full = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv_full,
            file_name=f"{selected_name.rsplit('.', 1)[0]}_full.csv",
            mime="text/csv",
        )
    with col2:
        if search_term or (filter_cols and len(filter_cols) < len(df.columns)):
            csv_filt = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_filt,
                file_name=f"{selected_name.rsplit('.', 1)[0]}_filtered.csv",
                mime="text/csv",
            )
        else:
            st.info("Apply filters above to enable filtered download.")
