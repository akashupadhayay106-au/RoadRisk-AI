"""
🌦️ Weather Analysis — Historical Weather & Accident Correlations
==================================================================
Module G: Open-Meteo API integration and weather-accident analysis.
Will be fully implemented in Phase 7.
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
    background: linear-gradient(135deg, #4DA8FF, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🌦️ Weather Analysis</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Analyze historical weather conditions and their association with accidents.
</p>
""", unsafe_allow_html=True)
st.divider()

info_card(
    "🚧 <strong>Coming in Phase 7</strong><br>"
    "This module will include:<br>"
    "• Open-Meteo historical weather data retrieval<br>"
    "• Temperature, precipitation, humidity, and wind analysis<br>"
    "• Weather-accident joins with defensible time/location matching<br>"
    "• Caching to minimize API calls<br>"
    "• Rate-limit and network error handling<br>"
    "• Exploratory correlation plots"
)

section_header("Important Note", "⚠️")
st.markdown("""
> **Correlation ≠ Causation.**
> Weather conditions observed around accident times show *association*,
> not proven causal relationships. Many confounding factors (traffic volume,
> road conditions, driver behavior) are not captured in the data.
""")
