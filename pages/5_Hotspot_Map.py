"""
🗺️ Hotspot Map — Geospatial Accident Cluster Explorer
=======================================================
Module F: Map-based visualization with DBSCAN clustering.
Will be fully implemented in Phase 6.
"""

import streamlit as st
from src.config import PAGE_CONFIG
from src.utils import inject_custom_css, setup_sidebar, section_header, info_card

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

st.markdown("""
<h1 style="
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #00D4AA, #4DA8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🗺️ Hotspot Map</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Explore accident spatial clusters and heatmaps.
</p>
""", unsafe_allow_html=True)
st.divider()

info_card(
    "🚧 <strong>Coming in Phase 6</strong><br>"
    "This module will include:<br>"
    "• Accident point maps with valid coordinates<br>"
    "• Heatmaps showing density<br>"
    "• DBSCAN spatial clustering<br>"
    "• Cluster summaries and statistics<br>"
    "• Separate UrbanGB (Great Britain) analysis<br>"
    "• Exportable hotspot results"
)

section_header("Important Note", "⚠️")
st.markdown("""
> **Accident counts ≠ Risk rates.**
> Spatial clusters show where accidents were *reported*, not necessarily
> where roads are most *dangerous*. Without traffic volume and reporting
> coverage data, high-count areas may simply reflect heavier traffic or
> better reporting. This module will clearly label results as
> "observed accident concentrations."
""")
