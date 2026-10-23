"""
📈 Model Explainability — SHAP, Feature Importance & Monitoring
================================================================
Module J: Explainable AI and model performance tracking.
Will be fully implemented in Phase 10.
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
    background: linear-gradient(135deg, #FFD600, #FF8C42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">📈 Model Explainability</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Understand model decisions with SHAP values, feature importance, and monitoring.
</p>
""", unsafe_allow_html=True)
st.divider()

info_card(
    "🚧 <strong>Coming in Phase 10</strong><br>"
    "This module will include:<br>"
    "• Global feature importance plots<br>"
    "• SHAP summary and dependence plots<br>"
    "• Individual prediction explanations<br>"
    "• Model performance summary dashboard<br>"
    "• Input validation and confidence analysis<br>"
    "• Data drift checks (if feasible)"
)

section_header("Important Note", "⚠️")
st.markdown("""
> **Feature importance ≠ Causality.**
> A feature being important for the model's predictions does not mean
> it *causes* the predicted outcome. ML models capture statistical
> associations in the training data, which may reflect confounders,
> reporting biases, or data artifacts.
""")
