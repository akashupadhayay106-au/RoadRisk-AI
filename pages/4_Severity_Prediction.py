"""
🎯 Severity Prediction — ML Accident Severity Classification
==============================================================
Module E: Complete ML pipeline for severity prediction.
Will be fully implemented in Phase 5.
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
    background: linear-gradient(135deg, #FF4B4B, #FFD600);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🎯 Severity Prediction</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Classify accident severity using machine learning models.
</p>
""", unsafe_allow_html=True)
st.divider()

info_card(
    "🚧 <strong>Coming in Phase 5</strong><br>"
    "This module will include a complete ML pipeline:<br>"
    "• Target inspection and leakage prevention<br>"
    "• Feature encoding, scaling, and missing-value handling<br>"
    "• Stratified train/test split<br>"
    "• Logistic Regression baseline → Random Forest → XGBoost/LightGBM<br>"
    "• Macro-F1, precision, recall, confusion matrix evaluation<br>"
    "• Saved pipeline loaded in Streamlit for interactive predictions<br>"
    "• SHAP-based feature explanations"
)

section_header("ML Pipeline Roadmap", "🗺️")

st.markdown("""
| Step | Description | Status |
|---|---|---|
| 1. Data inspection | Inspect target labels and feature types | 🔜 Phase 5 |
| 2. Feature engineering | Encode categoricals, scale numerics | 🔜 Phase 5 |
| 3. Baseline model | Logistic Regression | 🔜 Phase 5 |
| 4. Advanced models | Random Forest, XGBoost | 🔜 Phase 5 |
| 5. Evaluation | Metrics, confusion matrix, cross-validation | 🔜 Phase 5 |
| 6. Deployment | Saved pipeline + Streamlit prediction UI | 🔜 Phase 5 |
| 7. Explainability | SHAP values and feature importance | 🔜 Phase 5 |
""")

st.markdown("---")
st.markdown("""
<div style="
    background: rgba(255,75,75,0.08); border: 1px solid rgba(255,75,75,0.2);
    border-radius: 10px; padding: 1rem; font-size: 0.8rem; color: #8899A6;
">
    <strong style="color: #FF4B4B;">⚠️ Important:</strong>
    This is a decision-support demonstration, not a certified safety system.
    Predictions should not be used for emergency response or policy decisions
    without expert validation.
</div>
""", unsafe_allow_html=True)
