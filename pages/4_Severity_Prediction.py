"""
🎯 Severity Prediction — ML Accident Severity Classification
==============================================================
Module E: Interactive inference engine for predicting crash severity
from pre-incident driver, vehicular, atmospheric, and roadway factors.
Includes comparative benchmark diagnostics, confusion matrices, and probabilities.
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import PAGE_CONFIG, MODELS_DIR
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    error_state,
)
from src.features import get_time_of_day
from src.evaluate_model import plot_confusion_matrix_plotly

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #FF4B4B, #FFD600);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">🎯 Accident Severity Prediction Studio</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Supervised machine learning classification with class-balanced weighting, leakage prevention, and model benchmarking.
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Load Model Artifacts ─────────────────────────────────────────────────────
pipeline_path = MODELS_DIR / "severity_pipeline.joblib"
metrics_path = MODELS_DIR / "severity_model_comparison.json"

from src.train_severity_model import load_severity_model

try:
    model = load_severity_model(pipeline_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_meta = json.load(f)
except Exception as e:
    error_state(
        "Model artifacts unavailable",
        f"The severity model could not be loaded ({type(e).__name__}).",
        hint="Run the training pipeline to regenerate `severity_pipeline.joblib` and `severity_model_comparison.json`, then reload.",
        icon="🧯",
    )
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_predict, tab_benchmarks, tab_methodology = st.tabs([
    "🎯 Interactive Predictor",
    "📊 Model Comparison & Benchmarks",
    "🛡️ Methodology & Leakage Prevention"
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Interactive Predictor
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    section_header("Configure Incident Parameters", "⚙️")

    st.markdown("""
    Select the pre-incident operational attributes to evaluate the predicted accident severity.
    *Notice: Post-outcome casualty severity features are strictly omitted to prevent target leakage.*
    """)

    with st.form("severity_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Driver & Temporal Context")
            age_band = st.selectbox(
                "Age Band of Driver:",
                ["18-30", "31-50", "Over 51", "Under 18", "Unknown"],
                index=0,
            )
            sex_driver = st.selectbox("Sex of Driver:", ["Male", "Female", "Unknown"], index=0)
            edu_level = st.selectbox(
                "Driver Educational Level:",
                ["Junior high school", "Elementary school", "High school", "Above high school", "Writing & reading", "Illiterate", "Unknown"],
                index=0,
            )
            driving_exp = st.selectbox(
                "Driving Experience:",
                ["5-10yr", "2-5yr", "Above 10yr", "1-2yr", "Below 1yr", "No Licence", "Unknown"],
                index=0,
            )
            day_week = st.selectbox(
                "Day of Week:",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=4,
            )
            hour_val = st.slider("Hour of Day (0–23):", 0, 23, 18)

        with col2:
            st.markdown("##### 🚙 Vehicular & Road Conditions")
            vehicle_type = st.selectbox(
                "Type of Vehicle:",
                ["Automobile", "Lorry (41?100Q)", "Other", "Pick up upto 10Q", "Public (12?45 seats)", "Lorry (11?40Q)", "Long lorry", "Public (> 45 seats)", "Motorcycle", "Special vehicle", "Ridden horse", "Turbo", "Unknown"],
                index=0,
            )
            service_year = st.selectbox(
                "Service Year of Vehicle:",
                ["Unknown", "Above 10yr", "5-10yrs", "1-2yr", "2-5yrs", "Below 1yr"],
                index=0,
            )
            defect_vehicle = st.selectbox(
                "Defect of Vehicle:",
                ["No defect", "7", "5", "Unknown"],
                index=0,
            )
            road_surface_type = st.selectbox(
                "Road Surface Type:",
                ["Asphalt roads", "Earth roads", "Gravel roads", "Asphalt roads with some problem", "Other", "Unknown"],
                index=0,
            )
            road_surface_cond = st.selectbox(
                "Road Surface Conditions:",
                ["Dry", "Wet or damp", "Snow", "Flood over 3cm. deep"],
                index=0,
            )
            light_cond = st.selectbox(
                "Light Conditions:",
                ["Daylight", "Darkness - lights lit", "Darkness - no lighting", "Darkness - lights unlit"],
                index=0,
            )
            weather_cond = st.selectbox(
                "Weather Conditions:",
                ["Normal", "Raining", "Raining and Windy", "Cloudy", "Other", "Windy", "Snow", "Fog or mist"],
                index=0,
            )

        with col3:
            st.markdown("##### 🛣️ Spatial & Collision Dynamics")
            area_accident = st.selectbox(
                "Area Accident Occurred:",
                ["Other", "Office areas", "Residential areas", "Church areas", "  Medical facilities", "Industrial areas", "School areas", "Outside rural", "Hospital areas", "Recreational areas", "Unknown"],
                index=1,
            )
            lanes = st.selectbox(
                "Lanes or Medians:",
                ["Two-way (divided with broken lines road marking)", "Undivided Two way", "other", "Double carriageway (median)", "One way", "Two-way (divided with solid lines road marking)", "Unknown"],
                index=0,
            )
            junction_type = st.selectbox(
                "Type of Junction:",
                ["Y Shape", "No junction", "Crossing", "Other", "O Shape", "T Shape", "X Shape", "Unknown"],
                index=0,
            )
            road_align = st.selectbox(
                "Road Alignment:",
                ["Tangent road with flat terrain", "Gentle horizontal curve", "Escarpments", "Tangent road with mild grade and flat terrain", "Steep grade upward with poor visual", "Unknown"],
                index=0,
            )
            type_collision = st.selectbox(
                "Type of Collision:",
                ["Vehicle with vehicle collision", "Collision with roadside objects", "Collision with pedestrians", "Rollover", "Collision with animals", "Other", "Unknown"],
                index=0,
            )
            veh_movement = st.selectbox(
                "Vehicle Movement:",
                ["Going straight", "Moving Backward", "Turn left", "Turn right", "Parked", "Overtaking", "Other", "Unknown"],
                index=0,
            )
            cause_accident = st.selectbox(
                "Cause of Accident:",
                ["No distancing", "Changing lane to the right", "Changing lane to the left", "Driving carelessly", "No priority to vehicle", "Moving Backward", "Overtaking", "Driving to the left", "Overloading", "Unknown"],
                index=0,
            )
            num_vehicles = st.number_input("Number of Vehicles Involved:", min_value=1, max_value=10, value=2)
            num_casualties = st.number_input("Number of Casualties:", min_value=0, max_value=15, value=1)

        submitted = st.form_submit_button("🚀 Run Severity Inference", type="primary")

    if submitted:
        # Construct single-row DataFrame matching the model features
        time_period = get_time_of_day(hour_val)
        input_dict = {
            "Day_of_week": day_week,
            "Age_band_of_driver": age_band,
            "Sex_of_driver": sex_driver,
            "Educational_level": edu_level,
            "Vehicle_driver_relation": "Employee",
            "Driving_experience": driving_exp,
            "Type_of_vehicle": vehicle_type,
            "Owner_of_vehicle": "Owner",
            "Service_year_of_vehicle": service_year,
            "Defect_of_vehicle": defect_vehicle,
            "Area_accident_occured": area_accident,
            "Lanes_or_Medians": lanes,
            "Road_allignment": road_align,
            "Types_of_Junction": junction_type,
            "Road_surface_type": road_surface_type,
            "Road_surface_conditions": road_surface_cond,
            "Light_conditions": light_cond,
            "Weather_conditions": weather_cond,
            "Type_of_collision": type_collision,
            "Vehicle_movement": veh_movement,
            "Cause_of_accident": cause_accident,
            "Time_of_Day": time_period,
            "Number_of_vehicles_involved": int(num_vehicles),
            "Number_of_casualties": int(num_casualties),
            "Hour": int(hour_val),
        }
        X_infer = pd.DataFrame([input_dict])

        # Prediction
        pred_label = model.predict(X_infer)[0]
        pred_probs = model.predict_proba(X_infer)[0]
        classes = list(model.classes_)

        # Color mapping for outcome
        color_map = {
            "Slight Injury": "#00D4AA",
            "Serious Injury": "#FFD600",
            "Fatal injury": "#FF4B4B",
        }
        badge_color = color_map.get(pred_label, "#4DA8FF")

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Inference Results", "📊")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border: 2px solid {badge_color};">
                <div class="kpi-label">Predicted Severity Classification</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {badge_color}; margin: 0.6rem 0;">
                    {pred_label}
                </div>
                <div style="font-size: 0.8rem; color: #8899A6;">
                    Optimal Model: <strong>{metrics_meta.get('best_model', 'Random Forest')}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if pred_label == "Fatal injury":
                st.error("⚠️ High critical risk factors detected. Pre-crash environmental and vehicular indicators signal elevated likelihood of severe outcomes.")
            elif pred_label == "Serious Injury":
                st.warning("⚡ Moderate to high impact risk. Key predictors show substantial probability of grievous injuries.")
            else:
                st.success("✅ Standard slight injury baseline. Environmental attributes favor low probability of fatal consequence.")

        with res_col2:
            st.markdown("##### Model Probability Distribution")
            df_prob = pd.DataFrame({
                "Severity Class": classes,
                "Probability": [round(float(p) * 100, 2) for p in pred_probs],
            })
            fig_prob = px.bar(
                df_prob,
                x="Probability",
                y="Severity Class",
                orientation="h",
                text="Probability",
                color="Severity Class",
                color_discrete_map=color_map,
            )
            fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_prob.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(range=[0, 100], title="Class Probability (%)"),
                margin=dict(l=20, r=40, t=10, b=20),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_prob, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Model Comparison & Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════
with tab_benchmarks:
    section_header("Algorithm Benchmark Diagnostics", "📈")

    st.markdown("""
    Because the dataset has extreme class imbalance (**86.3% Slight, 12.7% Serious, 1.0% Fatal**),
    raw accuracy is misleading. A naive classifier predicting "Slight" on every sample achieves 86.3% accuracy
    while detecting 0% of fatal crashes. **Macro-F1** treats all classes equally and serves as the primary decision metric.
    """)

    comp_data = metrics_meta.get("comparison", {})
    summary_rows = []
    for model_name, m in comp_data.items():
        summary_rows.append({
            "Algorithm": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Macro-F1": f"{m['macro_f1']:.4f}",
            "Weighted-F1": f"{m['weighted_f1']:.4f}",
            "Macro-Precision": f"{m['macro_precision']:.4f}",
            "Macro-Recall": f"{m['macro_recall']:.4f}",
        })
    df_comp = pd.DataFrame(summary_rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Confusion Matrix Comparison", "🔲")

    col_cm1, col_cm2 = st.columns(2)
    selected_eval_model = st.selectbox(
        "Select model to view Confusion Matrix:",
        list(comp_data.keys()),
        index=1,
    )
    m_info = comp_data[selected_eval_model]
    
    col_raw, col_norm = st.columns(2)
    with col_raw:
        fig_raw = plot_confusion_matrix_plotly(
            m_info["confusion_matrix"],
            class_names=m_info["classes"],
            normalize=False,
            title=f"{selected_eval_model} — Raw Counts",
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    with col_norm:
        fig_norm = plot_confusion_matrix_plotly(
            m_info["confusion_matrix_normalized"],
            class_names=m_info["classes"],
            normalize=True,
            title=f"{selected_eval_model} — Normalized Recall",
        )
        st.plotly_chart(fig_norm, use_container_width=True)

    # Per-Class Metrics Table
    st.markdown("##### Per-Class Precision, Recall, and F1-Score")
    per_class_rows = []
    for cls_name, pcm in m_info["per_class"].items():
        per_class_rows.append({
            "Class": cls_name,
            "Precision": f"{pcm['precision']:.4f}",
            "Recall": f"{pcm['recall']:.4f}",
            "F1-Score": f"{pcm['f1_score']:.4f}",
        })
    st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Methodology & Leakage Prevention
# ═══════════════════════════════════════════════════════════════════════════════
with tab_methodology:
    section_header("Methodological Rigor & Data Leakage Prevention", "🛡️")

    st.markdown("""
    ### 1. Data Leakage Prevention Protocol
    In many published Kaggle notebooks on this dataset, authors mistakenly include post-crash casualty columns:
    - `Casualty_severity` (which directly codes whether someone died or was slightly hurt)
    - `Casualty_class`
    - `Sex_of_casualty`
    - `Age_band_of_casualty`
    
    If `Casualty_severity` is used, the model achieves near-100% artificial accuracy because it learns a trivial rule:
    *If casualty severity is 1 (Fatal), predict Fatal injury.*
    
    **In RoadRisk AI:**
    All post-crash features are **strictly isolated and excluded** from the preprocessor. The model only learns from
    attributes that exist prior to the collision (road condition, weather, driver profile, vehicle type, junction geometry).

    ---
    ### 2. Extreme Class Imbalance Handling
    - **Slight Injury:** ~86%
    - **Serious Injury:** ~13%
    - **Fatal injury:** ~1%
    
    We apply **cost-sensitive learning** via `class_weight='balanced'`, penalizing misclassifications on minority fatal cases
    inversely proportional to their class frequency.
    
    ---
    ### 3. Responsible AI & Advisory Disclaimer
    > **Educational Decision-Support Prototype:** This machine learning pipeline is designed for portfolio demonstration
    > and risk factor evaluation. It is **not** certified for emergency dispatch or road infrastructure modifications
    > without formal engineering review and real-world clinical validation.
    """)
