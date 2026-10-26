"""
ℹ️ About — Project Information & Documentation
=================================================
Module K: Project purpose, sources, methodology, and responsible AI.
"""

import streamlit as st
from src.config import PAGE_CONFIG, APP_VERSION, DATASETS, COLORS
from src.utils import inject_custom_css, setup_sidebar, section_header, info_card

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Page Title ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #8899A6, #FAFAFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">ℹ️ About RoadRisk AI</h1>
""", unsafe_allow_html=True)
st.divider()

# ── Project Purpose ──────────────────────────────────────────────────────────
section_header("Project Purpose", "🎯")
st.markdown("""
**RoadRisk AI** is an end-to-end AI-powered road accident analytics and intelligence
platform built as an educational portfolio project. It demonstrates the integration of:

- **Data Engineering** — Cleaning, validating, and managing multi-source accident data
- **SQL Analytics** — Structured querying with SQLite for trend and severity analysis
- **Machine Learning** — Supervised classification of accident severity
- **Geospatial Analysis** — Spatial clustering and hotspot mapping
- **Computer Vision** — CCTV frame classification for accident detection
- **GenAI** — Pluggable LLM provider layer (Gemini / Groq / Hugging Face) with an offline grounded RAG assistant
- **Explainable AI** — SHAP-based model interpretation
- **Interactive Dashboards** — Professional Streamlit-based analytics interface
""")

# ── Technology Stack ─────────────────────────────────────────────────────────
section_header("Technology Stack", "🛠️")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Data & ML**
    - Python 3.12
    - Pandas & NumPy
    - Scikit-learn
    - XGBoost / LightGBM
    - SHAP
    - SQLite + SQLAlchemy
    """)
with col2:
    st.markdown("""
    **Visualization & Geo**
    - Plotly
    - Folium
    - Streamlit
    - GeoPandas
    - HeiGIT openrouteservice
    """)
with col3:
    st.markdown("""
    **AI & Vision**
    - Google Gemini API (google-genai)
    - Groq / Hugging Face (optional)
    - PyTorch
    - OpenCV
    - Open-Meteo API
    """)

# ── Data Sources ─────────────────────────────────────────────────────────────
section_header("Data Sources & Licenses", "📜")

for key, meta in DATASETS.items():
    with st.expander(f"📊 {meta['name']}"):
        st.markdown(f"""
        | Field | Detail |
        |---|---|
        | **Source** | [{meta['source']}]({meta['url']}) |
        | **License** | {meta['license']} |
        | **Description** | {meta['description']} |
        """)

# ── Methodology ──────────────────────────────────────────────────────────────
section_header("Methodology Overview", "📐")
st.markdown("""
1. **Data Inspection** — Each dataset is independently inspected for schema, quality, and provenance
2. **Cleaning & Validation** — Missing values, duplicates, type errors, and coordinate ranges are handled
3. **Separate Pipelines** — Datasets are treated as separate sources; joins are only made when defensible
4. **ML Pipeline** — Stratified splits, baseline + advanced models, cross-validation, macro-F1 evaluation
5. **Spatial Analysis** — DBSCAN clustering on validated coordinates with clear count-vs-risk labeling
6. **CV Pipeline** — Transfer learning on CCTV frames with proper train/val/test separation
7. **GenAI Integration** — Grounded responses using actual data and controlled database queries
""")

# ── Known Limitations ────────────────────────────────────────────────────────
section_header("Known Limitations", "⚠️")
st.markdown("""
- **Accident counts ≠ risk rates** — Without traffic exposure data, raw counts may reflect reporting bias
- **Weather association ≠ causation** — Correlated conditions don't prove causal relationships
- **ML predictions are demonstrations** — Not suitable for operational safety decisions without expert review
- **CCTV classification is offline** — Frame-level classification, not real-time video detection
- **Data coverage gaps** — Each dataset covers specific regions, timeframes, and schemas
- **Free-tier API limits** — LLM (Gemini/Groq/Hugging Face), routing (openrouteservice), and weather APIs have rate limits and quotas; the app falls back to offline mode
- **Deployment constraints** — Streamlit Community Cloud has resource and storage limits
""")

# ── Responsible AI ───────────────────────────────────────────────────────────
section_header("Responsible AI Statement", "🤝")
st.markdown("""
<div style="
    background: rgba(255,75,75,0.08); border: 1px solid rgba(255,75,75,0.2);
    border-radius: 10px; padding: 1.2rem; font-size: 0.85rem; color: #FAFAFA;
    line-height: 1.7;
">
    This platform is designed for <strong>educational and exploratory purposes</strong>.
    It should not be used as a substitute for professional road-safety analysis,
    emergency response systems, or policy-making tools.<br><br>
    All models, statistics, and visualizations are based on publicly available data
    with inherent limitations in coverage, accuracy, and timeliness.
    The developers do not make claims about the predictive accuracy of the models
    beyond the reported evaluation metrics on the specific datasets used.<br><br>
    Users should critically evaluate all outputs and consult domain experts before
    making any decisions based on this platform's results.
</div>
""", unsafe_allow_html=True)

# ── Version & Credits ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    **Version:** {APP_VERSION}
    **License:** MIT
    **Python:** 3.12
    """)
with col2:
    st.markdown("""
    **Built with:** [Streamlit](https://streamlit.io) •
    [Plotly](https://plotly.com) •
    [Scikit-learn](https://scikit-learn.org) •
    [Google Gemini](https://ai.google.dev)
    """)
