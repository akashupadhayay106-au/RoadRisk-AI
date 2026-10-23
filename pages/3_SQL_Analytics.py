"""
🗄️ SQL Analytics — Query the Accident Database
=================================================
Module C: SQLite-backed analytical queries.
Will be fully implemented in Phase 3.
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
    background: linear-gradient(135deg, #A855F7, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🗄️ SQL Analytics</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Run analytical queries against the accident database.
</p>
""", unsafe_allow_html=True)
st.divider()

info_card(
    "🚧 <strong>Coming in Phase 3</strong><br>"
    "This module will include:<br>"
    "• Normalized SQLite schema based on real dataset columns<br>"
    "• Database creation and data loading pipeline<br>"
    "• Parameterized queries for trends, severity, and regional analysis<br>"
    "• Interactive query interface for predefined analytical questions<br>"
    "• Query results with downloadable tables"
)

section_header("Planned Features", "📋")
st.markdown("""
| Feature | Status |
|---|---|
| SQLite database creation | 🔜 Phase 3 |
| Data loading pipeline | 🔜 Phase 3 |
| Monthly/yearly trend queries | 🔜 Phase 3 |
| Severity distribution queries | 🔜 Phase 3 |
| Regional comparison queries | 🔜 Phase 3 |
| Custom query interface | 🔜 Phase 3 |
""")
