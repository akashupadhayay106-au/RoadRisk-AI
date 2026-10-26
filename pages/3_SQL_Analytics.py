"""
🗄️ SQL Analytics Studio — Query the Accident Database
======================================================
Module C: Interactive SQLite-backed analytical studio.
Execute predefined safety analyses, run custom read-only SQL queries,
inspect the relational schema, and download analytical datasets.
"""

import time
import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import PAGE_CONFIG, DATABASE_PATH
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    info_card,
)
from src.database import (
    execute_query,
    get_database_summary,
    load_data_to_database,
)
from src.sql_queries import (
    PREDEFINED_QUERIES,
)

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Title & Header ───────────────────────────────────────────────────────────
st.markdown("""
<h1 style="
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #A855F7, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🗄️ SQL Analytics Studio</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    High-performance, parameterized SQLite query engine for multi-source road safety analytics.
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Ensure Database Exists ───────────────────────────────────────────────────
if not DATABASE_PATH.exists():
    with st.spinner("Initializing SQLite database and indexing tables…"):
        load_data_to_database()

# ── Database Health Strip ────────────────────────────────────────────────────
db_summary = get_database_summary()
total_tables = len(db_summary["tables"])
total_records = sum(t["row_count"] for t in db_summary["tables"].values())
db_size_mb = round(db_summary["size_bytes"] / (1024 * 1024), 2)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Database:** `roadrisk.db`")
with col2:
    st.markdown(f"**Total Records:** `{total_records:,}`")
with col3:
    st.markdown(f"**Indexed Tables:** `{total_tables}`")
with col4:
    st.markdown(f"**Storage:** `{db_size_mb} MB`")

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation Tabs ──────────────────────────────────────────────────────────
tab_predefined, tab_custom, tab_schema = st.tabs([
    "📊 Predefined Analytical Questions",
    "💻 Interactive SQL Console",
    "🗄️ Database Schema & Indexes"
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Predefined Analytical Questions
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predefined:
    section_header("Curated Safety Analyses", "🔍")
    
    query_names = list(PREDEFINED_QUERIES.keys())
    selected_q = st.selectbox(
        "Choose an analytical question to execute:",
        query_names,
        index=0,
    )
    
    q_data = PREDEFINED_QUERIES[selected_q]
    
    st.markdown(f"### {q_data['title']}")
    st.markdown(f"<p style='color: #8899A6;'>{q_data['description']}</p>", unsafe_allow_html=True)
    
    with st.expander("📝 View Underlying SQL Query", expanded=False):
        st.code(q_data["sql"], language="sql")
    
    # Execute query with timing
    t0 = time.time()
    try:
        df_result = execute_query(q_data["sql"])
        latency_ms = round((time.time() - t0) * 1000, 2)
        st.caption(f"⚡ Executed in **{latency_ms} ms** | Returned **{len(df_result)}** rows")
    except Exception as e:
        st.error(f"Error executing query: {e}")
        df_result = pd.DataFrame()

    if not df_result.empty:
        # Visual presentation based on selected question
        if "city" in df_result.columns and "total_accidents" in df_result.columns:
            fig = px.bar(
                df_result,
                x="city",
                y="total_accidents",
                color="fatal_percentage" if "fatal_percentage" in df_result.columns else "total_casualties",
                title="Accident Volume and Severity Across Cities",
                labels={"city": "City", "total_accidents": "Total Accidents"},
                color_continuous_scale=["#00D4AA", "#FFD600", "#FF4B4B"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif "weather" in df_result.columns and "visibility" in df_result.columns:
            fig = px.bar(
                df_result,
                x="weather",
                y="total_accidents",
                color="visibility",
                barmode="group",
                title="Accidents by Weather & Visibility Combination",
                color_discrete_sequence=["#4DA8FF", "#FF8C42", "#FF4B4B"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif "km_bracket" in df_result.columns:
            fig = px.bar(
                df_result,
                x="km_bracket",
                y="incident_count",
                color="total_casualties",
                title="Highway Accidents by Kilometer Marker (OpenCity)",
                color_continuous_scale="Reds",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tabular Result
        st.dataframe(df_result, use_container_width=True)

        # Export Button
        csv_data = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Query Result (CSV)",
            data=csv_data,
            file_name=f"sql_analysis_{selected_q[:2].strip()}.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Interactive SQL Console
# ═══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    section_header("Ad-Hoc SQL Query Console", "💻")
    
    info_card(
        "🛡️ <strong>Safety Protection:</strong> Only read-only <code>SELECT</code> queries "
        "are permitted. Destructive DDL/DML statements (<code>DROP</code>, <code>DELETE</code>, "
        "<code>ALTER</code>, <code>UPDATE</code>) are strictly blocked."
    )
    
    default_query = """SELECT 
    city,
    road_type,
    weather,
    COUNT(*) AS incident_count,
    ROUND(AVG(casualties), 2) AS mean_casualties,
    ROUND(AVG(risk_score), 3) AS mean_risk_score
FROM accidents_india
WHERE accident_severity = 'fatal'
GROUP BY city, road_type, weather
ORDER BY incident_count DESC
LIMIT 20;"""

    query_input = st.text_area(
        "Enter your SQL SELECT statement:",
        value=default_query,
        height=180,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Execute Query", type="primary")

    if run_btn:
        t0 = time.time()
        try:
            df_custom = execute_query(query_input)
            latency_ms = round((time.time() - t0) * 1000, 2)
            st.success(f"✓ Query executed successfully in **{latency_ms} ms** ({len(df_custom):,} rows returned).")
            
            st.dataframe(df_custom, use_container_width=True, height=350)

            # Download Result
            custom_csv = df_custom.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Result (CSV)",
                data=custom_csv,
                file_name="custom_query_result.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Execution Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Database Schema & Indexes
# ═══════════════════════════════════════════════════════════════════════════════
with tab_schema:
    section_header("Relational Schema & Index Architecture", "🗄️")
    
    st.markdown("""
    The database architecture follows a clean relational structure with proper column typing,
    primary keys, and B-tree indexes on frequently filtered fields.
    """)
    
    for tbl_name, tbl_data in db_summary["tables"].items():
        with st.expander(f"📁 Table: `{tbl_name}` — {tbl_data['row_count']:,} records", expanded=(tbl_name == "accidents_india")):
            df_cols = pd.DataFrame(tbl_data["columns"])
            st.dataframe(df_cols, use_container_width=True, hide_index=True)

    st.markdown("#### Database Indexes (B-Tree)")
    if db_summary["indexes"]:
        df_idx = pd.DataFrame(db_summary["indexes"])
        st.dataframe(df_idx, use_container_width=True, hide_index=True)
    else:
        st.info("No custom indexes created.")
