"""
📈 Model Explainability — Feature Importance, SHAP Concepts & Causal Governance
================================================================================
Module J: Interpretability suite for the accident severity classification model.
Visualizes global feature contributions, explains individual incident risk factors,
and enforces scientific boundaries regarding correlation vs physical causality.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px

from src.config import PAGE_CONFIG
from src.utils import inject_custom_css, setup_sidebar, section_header, metric_card, info_card
from src.explainability import (
    get_global_feature_importance,
    explain_single_prediction,
    load_severity_artifact,
)

# ── Streamlit Setup ───────────────────────────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.2rem;">
    <h1 style="
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(135deg, #FFD600 0%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    ">📈 Model Explainability & Interpretability Studio</h1>
    <p style="color: #8899A6; font-size: 0.95rem; line-height: 1.5;">
        Deconstruct random forest severity decisions using Gini feature importances,
        local incident factor attributions, and rigorous causal governance disclaimers.
    </p>
</div>
""", unsafe_allow_html=True)

# Causal Disclaimer Card
st.markdown("""
<div style="
    background: rgba(255, 214, 0, 0.08); border: 1px solid rgba(255, 214, 0, 0.3);
    border-left: 4px solid #FFD600; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 1.2rem;
">
    <div style="font-weight: 700; color: #FFD600; font-size: 0.88rem;">
        ⚖️ Critical Scientific Foundation: Feature Importance ≠ Physical Causality
    </div>
    <div style="color: #B0BEC5; font-size: 0.82rem; margin-top: 3px; line-height: 1.45;">
        Feature importance ranks statistical associations inside the training data.
        A feature with high importance (e.g. <code>Light_conditions</code>) is highly predictive in police logs,
        but does not solely prove that changing lighting mechanically eliminates crashes without addressing confounders (e.g., driver fatigue, alcohol, road geometry).
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Model Artifact & Feature Importance ──────────────────────────────────
try:
    artifact = load_severity_artifact()
    df_imp = get_global_feature_importance(artifact)
except Exception as e:
    st.error(f"Could not load severity model: {e}")
    st.stop()

# ── KPI Ribbon ────────────────────────────────────────────────────────────────
top3 = df_imp.head(3)["feature"].tolist()
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Top Predictive Feature", top3[0] if len(top3) > 0 else "N/A", f"{df_imp.iloc[0]['importance_pct']:.1f}% Relative Weight", border_color="#FFD600")
with c2:
    metric_card("Second Factor", top3[1] if len(top3) > 1 else "N/A", f"{df_imp.iloc[1]['importance_pct']:.1f}% Relative Weight", border_color="#FF8C42")
with c3:
    metric_card("Third Factor", top3[2] if len(top3) > 2 else "N/A", f"{df_imp.iloc[2]['importance_pct']:.1f}% Relative Weight", border_color="#4DA8FF")
with c4:
    metric_card("Top 5 Feature Mass", f"{df_imp.iloc[4]['cumulative_pct']:.1f}%", "Cumulative predictive power", border_color="#00C853")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Global Feature Importance",
    "🔍 Interactive Incident Attribution",
    "📚 Methodological Governance",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Global Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("Global Feature Importance Hierarchy (Random Forest MDI)", "📊")

    top_n = st.slider("Number of top features to display:", 5, min(20, len(df_imp)), 10)
    df_top = df_imp.head(top_n).sort_values(by="importance_pct", ascending=True)

    fig_bar = px.bar(
        df_top,
        x="importance_pct",
        y="feature",
        orientation="h",
        color="importance_pct",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Features by Relative Predictive Weight (%)",
        labels={"importance_pct": "Relative Importance (%)", "feature": "Incident Feature"},
    )
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161B22",
        plot_bgcolor="#161B22",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Ranked Data Table
    st.markdown("**Complete Feature Importance Breakdown:**")
    st.dataframe(
        df_imp[["rank", "feature", "importance_pct", "cumulative_pct"]].rename(columns={
            "rank": "Rank",
            "feature": "Feature Name",
            "importance_pct": "Importance Share (%)",
            "cumulative_pct": "Cumulative Share (%)",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Interactive Incident Attribution
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Single Incident Risk Factor Decomposition", "🔍")

    st.markdown("""
    Select crash circumstances below to evaluate how the model weighs specific risk indicators
    versus protective factors for an individual incident scenario.
    """)

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        light = st.selectbox(
            "Lighting Conditions:",
            ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting"],
        )
        weather = st.selectbox(
            "Weather Conditions:",
            ["Normal", "Raining", "Fog or mist", "Windy", "Snow"],
        )
    with ic2:
        lanes = st.selectbox(
            "Roadway Lanes / Median:",
            ["Two-way (divided with median)", "Undivided Two way", "One way", "Double carriageway"],
        )
        driving_exp = st.selectbox(
            "Driver Experience:",
            ["Above 10yr", "5-10yr", "2-5yr", "1-2yr", "Under 1yr"],
        )
    with ic3:
        hour = st.slider("Incident Hour (0-23):", 0, 23, 21)
        area = st.selectbox(
            "Accident Location Area:",
            ["Other", "Office areas", "Residential areas", "Industrial areas", "Church areas", "Hospital areas"],
        )

    # Assemble test record with safe defaults
    sample_input = {
        "Number_of_vehicles_involved": 2,
        "Number_of_casualties": 1,
        "Hour": hour,
        "Day_of_week": "Friday",
        "Age_band_of_driver": "18-30",
        "Sex_of_driver": "Male",
        "Educational_level": "Junior high school",
        "Vehicle_driver_relation": "Employee",
        "Driving_experience": driving_exp,
        "Type_of_vehicle": "Automobile",
        "Owner_of_vehicle": "Owner",
        "Service_year_of_vehicle": "1-2yr",
        "Defect_of_vehicle": "No defect",
        "Area_accident_occured": area,
        "Lanes_or_Medians": lanes,
        "Road_allignment": "Tangent road with flat terrain",
        "Types_of_Junction": "No junction",
        "Road_surface_type": "Asphalt roads",
        "Road_surface_conditions": "Dry",
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Type_of_collision": "Vehicle with vehicle collision",
        "Vehicle_movement": "Going straight",
        "Cause_of_accident": "Moving Backward",
        "Time_of_Day": "Night" if (hour >= 20 or hour <= 5) else "Evening Rush",
    }

    if st.button("🔍 Explain Prediction for this Incident", use_container_width=True):
        explanation = explain_single_prediction(sample_input, artifact=artifact)

        pred_class = explanation["predicted_class"]
        probs = explanation["probabilities"]

        st.markdown(f"### Predicted Severity: **{pred_class}**")

        # Class Probability Display
        prob_cols = st.columns(len(probs))
        for idx, (cls_name, prob_val) in enumerate(probs.items()):
            with prob_cols[idx]:
                metric_card(f"Probability: {cls_name}", f"{prob_val*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            st.markdown("#### 🚨 Primary Compounding Risk Factors:")
            risks = explanation["primary_risk_factors"]
            if risks:
                for r in risks:
                    st.markdown(f"- 🔴 {r}")
            else:
                st.markdown("*(No severe environmental hazard flags triggered)*")

        with r2:
            st.markdown("#### 🛡️ Mitigating / Protective Factors:")
            protects = explanation["protective_factors"]
            if protects:
                for p in protects:
                    st.markdown(f"- 🟢 {p}")
            else:
                st.markdown("*(Baseline environmental conditions)*")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Methodological Governance
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Scientific Governance: Interpretability Limits & Ethics", "📚")

    info_card(
        "### Limitations of Model Interpretability Methods in Road Safety:<br><br>"
        "1. <strong>Tree Impurity (MDI) Bias:</strong> Mean Decrease in Impurity favors high-cardinality categorical features. "
        "In this project, categories were standardized to minimize artificial cardinality inflation.<br><br>"
        "2. <strong>Confounding and Proxy Variables:</strong> A feature like <code>Hour</code> often acts as a proxy for fatigue, "
        "alcohol consumption, and diminished enforcement rather than illumination alone.<br><br>"
        "3. <strong>Reporting & Survivor Biases:</strong> Minor accidents often go unrecorded in police databases, skewing "
        "the baseline severity distribution toward more severe crashes.<br><br>"
        "4. <strong>Intervention Design Caution:</strong> Do not design traffic engineering policies solely on ML feature rankings "
        "without conducting randomized controlled field trials or rigorous econometric counterfactual studies."
    )
