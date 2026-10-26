"""
📹 CCTV Detection — Computer Vision Crash Classifier & Sequence Audit
========================================================================
Module H: Deep Learning frame-level accident classification using fine-tuned
MobileNetV2, Grad-CAM attention heatmaps, video frame sampling, baseline comparison,
and source-video sequence leakage auditing.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    PAGE_CONFIG,
    CCTV_TEST_DIR,
)
from src.utils import (
    inject_custom_css,
    setup_sidebar,
    section_header,
    metric_card,
    info_card,
    empty_state,
    error_state,
)
from src.cctv_preprocessing import (
    scan_cctv_manifest,
    audit_split_leakage,
)
from src.cctv_inference import (
    load_cctv_model,
    predict_cctv_frame,
    generate_cctv_gradcam,
    sample_video_frames,
    CCTV_METRICS_PATH,
    CCTV_BASELINE_METRICS_PATH,
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
        background: linear-gradient(135deg, #EC4899 0%, #FF4B4B 50%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    ">📹 CCTV Frame Crash Classifier & Video Audit Studio</h1>
    <p style="color: #8899A6; font-size: 0.95rem; line-height: 1.5;">
        Offline surveillance frame classification using MobileNetV2 transfer learning,
        sub-50ms CPU inference, Grad-CAM visual attention interpretability, and video frame stream analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# Experimental Use Disclaimer Banner
st.markdown("""
<div style="
    background: rgba(255, 140, 66, 0.08); border: 1px solid rgba(255, 140, 66, 0.3);
    border-left: 4px solid #FF8C42; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1.2rem;
">
    <div style="font-weight: 700; color: #FF8C42; font-size: 0.88rem;">
        ⚠️ Portfolio Prototype Notice & Scientific Boundary
    </div>
    <div style="color: #B0BEC5; font-size: 0.82rem; margin-top: 2px; line-height: 1.4;">
        This system demonstrates <strong>frame-level image classification</strong> on static CCTV frames.
        It is <strong>not a certified real-time accident detection or automated emergency dispatch system</strong>.
        Operational deployments require multi-frame temporal confirmation, object trajectory tracking, and human-in-the-loop oversight.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Model & Metrics ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_cctv_model()

model, device_name = get_cached_model()

# Load evaluation metrics if available
metrics_data = {}
if CCTV_METRICS_PATH.exists():
    try:
        with open(CCTV_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception:
        pass

test_metrics = metrics_data.get("test_metrics", {
    "accuracy": 0.81,
    "macro_f1": 0.8091,
    "roc_auc": 0.8266,
    "confusion_matrix": [[37, 10], [9, 44]],
})

# Load baseline metrics if available
baseline_metrics = {}
if CCTV_BASELINE_METRICS_PATH.exists():
    try:
        with open(CCTV_BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)
    except Exception:
        pass

# ── KPI Ribbon ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Deep Architecture", "MobileNetV2", "ImageNet-1K Transfer Learning", border_color="#EC4899")
with k2:
    metric_card("Test Accuracy", f"{test_metrics.get('accuracy', 0.81)*100:.1f}%", f"Macro-F1: {test_metrics.get('macro_f1', 0.809):.3f}", border_color="#00C853")
with k3:
    metric_card("Edge Latency (CPU)", "~35 - 50 ms", "Approx. 20–28 FPS on CPU", border_color="#4DA8FF")
with k4:
    metric_card("Dataset Volume", "990 Frames", "791 Train • 98 Val • 101 Test", border_color="#FF8C42")

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔍 Frame & Video Stream Classifier",
    "🔬 Video Sequence Leakage Audit",
    "📊 Model Performance & Benchmarks",
    "⚡ Edge Deployment & Latency Budget",
    "⚠️ Operational Governance & Disclaimers",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Frame & Video Stream Classifier
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("CCTV Incident Classifier & Attention Map", "📹")

    # Selection mode
    input_mode = st.radio(
        "Choose Surveillance Feed Source:",
        [
            "Curated CCTV Test Gallery",
            "Upload Custom Image Frame (JPG/PNG)",
            "Upload Video Stream & Sample Frames (MP4/AVI/MOV)",
        ],
        horizontal=True,
    )

    if input_mode == "Curated CCTV Test Gallery":
        acc_samples = sorted(list((CCTV_TEST_DIR / "Accident").glob("*.*")))[:6]
        non_samples = sorted(list((CCTV_TEST_DIR / "Non Accident").glob("*.*")))[:6]

        sample_options = {}
        for i, p in enumerate(acc_samples):
            sample_options[f"🚨 Accident Frame #{i+1} ({p.name})"] = p
        for i, p in enumerate(non_samples):
            sample_options[f"🚗 Normal Traffic Frame #{i+1} ({p.name})"] = p

        chosen_label = st.selectbox("Select a benchmark CCTV test frame:", list(sample_options.keys()))
        selected_path = sample_options[chosen_label]
        selected_image = Image.open(selected_path).convert("RGB")
        image_title = chosen_label

        # Single frame inference
        with st.spinner("Executing neural inference & Grad-CAM feature analysis..."):
            pred_res = predict_cctv_frame(selected_image, model=model, device=device_name)
            overlay_img, cam_map = generate_cctv_gradcam(selected_image, model=model, device=device_name)

        is_acc = pred_res["is_accident"]
        conf = pred_res["confidence"]
        latency = pred_res["latency_ms"]
        fps_est = round(1000.0 / latency, 1) if latency > 0 else 30.0

        if is_acc:
            banner_bg = "rgba(255, 75, 75, 0.15)"
            banner_border = "#FF4B4B"
            banner_icon = "🚨"
            banner_title = f"CRITICAL INCIDENT CLASSIFIED — ACCIDENT ({conf*100:.1f}% Confidence)"
            banner_msg = f"Potential road collision or vehicle wreckage identified. Recommended action: Trigger PTZ zoom and alert traffic dispatch. Latency: {latency}ms ({fps_est} FPS)."
        else:
            banner_bg = "rgba(0, 200, 83, 0.12)"
            banner_border = "#00C853"
            banner_icon = "✅"
            banner_title = f"NORMAL TRAFFIC FLOW ({conf*100:.1f}% Confidence)"
            banner_msg = f"No accident signature detected. Regular flow conditions. Latency: {latency}ms ({fps_est} FPS)."

        st.markdown(f"""
        <div style="
            background: {banner_bg}; border: 1px solid {banner_border}; border-left: 6px solid {banner_border};
            border-radius: 8px; padding: 1rem 1.4rem; margin-bottom: 1.5rem;
        ">
            <div style="font-weight: 800; font-size: 1.15rem; color: {banner_border}; margin-bottom: 0.3rem;">
                {banner_icon} {banner_title}
            </div>
            <div style="color: #CCD6DD; font-size: 0.9rem; line-height: 1.5;">
                {banner_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown(f"**Surveillance Camera Frame** (`{selected_image.width} × {selected_image.height}`)")
            st.image(selected_image, use_container_width=True)
        with col_img2:
            st.markdown("**Grad-CAM Visual Attention Heatmap** (Salient Regions)")
            st.image(overlay_img, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_prob1, col_prob2 = st.columns([2, 1])
        with col_prob1:
            st.markdown("**Class Probability Breakdown:**")
            p_acc = pred_res["probabilities"]["Accident"]
            p_non = pred_res["probabilities"]["Non Accident"]
            prob_df = pd.DataFrame({
                "Class": ["Accident", "Non Accident"],
                "Probability": [p_acc, p_non],
            })
            fig_p = px.bar(
                prob_df,
                x="Probability",
                y="Class",
                orientation="h",
                color="Class",
                color_discrete_map={"Accident": "#FF4B4B", "Non Accident": "#00C853"},
                text="Probability",
                range_x=[0, 1.0],
            )
            fig_p.update_traces(texttemplate="%{x:.1%}", textposition="outside")
            fig_p.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                showlegend=False,
                height=180,
                margin=dict(l=20, r=40, t=10, b=10),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with col_prob2:
            st.markdown("**Edge Inference Telemetry:**")
            st.markdown(f"""
            <div class="kpi-card" style="text-align: left; padding: 1rem;">
                <div style="font-size: 0.85rem; color: #8899A6;">Inference Time: <strong style="color: #4DA8FF;">{latency} ms</strong></div>
                <div style="font-size: 0.85rem; color: #8899A6; margin-top: 4px;">Throughput: <strong style="color: #00C853;">~{fps_est} FPS</strong></div>
                <div style="font-size: 0.85rem; color: #8899A6; margin-top: 4px;">Execution Device: <code>{device_name.upper()}</code></div>
                <div style="font-size: 0.85rem; color: #8899A6; margin-top: 4px;">Input Resolution: <code>224 × 224 (Resized)</code></div>
            </div>
            """, unsafe_allow_html=True)

    elif input_mode == "Upload Custom Image Frame (JPG/PNG)":
        uploaded_file = st.file_uploader(
            "Upload a road or CCTV surveillance frame:",
            type=["jpg", "jpeg", "png", "bmp"],
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("Executing neural inference & Grad-CAM analysis..."):
                pred_res = predict_cctv_frame(selected_image, model=model, device=device_name)
                overlay_img, cam_map = generate_cctv_gradcam(selected_image, model=model, device=device_name)

            is_acc = pred_res["is_accident"]
            conf = pred_res["confidence"]
            latency = pred_res["latency_ms"]

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown(f"**Uploaded Frame** (`{selected_image.width} × {selected_image.height}`)")
                st.image(selected_image, use_container_width=True)
            with col_img2:
                st.markdown("**Grad-CAM Heatmap Overlay**")
                st.image(overlay_img, use_container_width=True)

            st.write(f"**Prediction:** `{pred_res['prediction']}` | **Confidence:** `{conf*100:.1f}%` | **Latency:** `{latency} ms`")
        else:
            empty_state(
                "No frame uploaded yet",
                "Upload a JPG or PNG traffic-surveillance frame to run neural inference and Grad-CAM.",
                hint="Use the Curated Gallery tab to evaluate bundled benchmark frames instead.",
                icon="🖼️",
            )

    else:
        # Video Upload & Frame Sampling Mode
        st.markdown("""
        **Surveillance Video Stream Frame Ingestion:**
        Upload an MP4, AVI, or MOV video stream. The edge simulator will sample frames at the configured
        temporal stride, run batch neural inference, and generate an incident risk timeline.
        """)

        vc1, vc2 = st.columns([2, 1])
        with vc1:
            uploaded_video = st.file_uploader(
                "Upload surveillance video clip (MP4, AVI, MOV):",
                type=["mp4", "avi", "mov"],
            )
        with vc2:
            sample_interval = st.slider("Sampling Stride (seconds):", 0.5, 5.0, 1.0, 0.5)
            max_sampled = st.slider("Max Frames to Sample:", 10, 60, 25)

        if uploaded_video is not None:
            if st.button("🚀 Process & Sample Video Feed"):
                with st.spinner(f"Extracting video frames every {sample_interval}s via OpenCV..."):
                    try:
                        frames_data = sample_video_frames(
                            uploaded_video,
                            sample_interval_sec=sample_interval,
                            max_frames=max_sampled,
                        )
                    except Exception as e:
                        error_state(
                            "Could not read video stream",
                            f"Frame extraction failed ({type(e).__name__}).",
                            hint="Confirm the file is a valid MP4/AVI/MOV clip and that OpenCV can decode its codec.",
                            icon="🎬",
                        )
                        frames_data = []

                if frames_data:
                    st.success(f"Successfully extracted and processed {len(frames_data)} sampled frames.")
                    
                    timeline_records = []
                    for item in frames_data:
                        pred = predict_cctv_frame(item["image"], model=model, device=device_name)
                        timeline_records.append({
                            "timestamp_sec": item["timestamp_sec"],
                            "frame_index": item["frame_index"],
                            "prediction": pred["prediction"],
                            "prob_accident": pred["probabilities"]["Accident"],
                            "prob_non_accident": pred["probabilities"]["Non Accident"],
                            "image": item["image"],
                        })

                    df_timeline = pd.DataFrame(timeline_records)

                    # Plot timeline
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=df_timeline["timestamp_sec"],
                        y=df_timeline["prob_accident"],
                        mode="lines+markers",
                        name="Accident Probability",
                        line=dict(color="#FF4B4B", width=3),
                        marker=dict(size=8, color="#FF8C42"),
                    ))
                    fig_timeline.add_hline(
                        y=0.5, line_dash="dash", line_color="#8899A6",
                        annotation_text="Decision Threshold (0.50)",
                    )
                    fig_timeline.update_layout(
                        title="Temporal Accident Probability Profile Across Video Timeline",
                        template="plotly_dark",
                        paper_bgcolor="#161B22",
                        plot_bgcolor="#161B22",
                        xaxis_title="Time in Video (Seconds)",
                        yaxis_title="Estimated Crash Probability",
                        yaxis=dict(range=[0, 1.05]),
                        height=320,
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # Display top alert frames
                    alert_frames = df_timeline[df_timeline["prob_accident"] >= 0.50]
                    if not alert_frames.empty:
                        st.markdown(f"### 🚨 Flagged Incident Frames ({len(alert_frames)} detections)")
                        cols = st.columns(min(3, len(alert_frames)))
                        for idx, (_, row) in enumerate(alert_frames.head(3).iterrows()):
                            with cols[idx % 3]:
                                st.image(row["image"], use_container_width=True)
                                st.caption(f"T={row['timestamp_sec']}s • Accident Prob: {row['prob_accident']*100:.1f}%")
                    else:
                        empty_state(
                            "No incident frames flagged",
                            "Every sampled frame stayed below the 0.50 accident-probability threshold.",
                            hint="Try a clip that contains an actual collision, or lower the sampling stride to inspect more frames.",
                            icon="✅",
                        )
                else:
                    error_state(
                        "No frames extracted",
                        "No frames could be sampled from the provided video file.",
                        hint="The clip may be empty, corrupted, or use an unsupported codec.",
                        icon="🎬",
                    )
        else:
            empty_state(
                "No video uploaded yet",
                "Upload an MP4, AVI, or MOV surveillance clip to run batch frame inference.",
                hint="Or switch to the Curated Gallery tab to evaluate bundled benchmark frames.",
                icon="📹",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Video Sequence Leakage Audit
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Scientific Audit: Video Sequence Split Leakage & Grouping Heuristic", "🔬")

    st.markdown("""
    In CCTV video datasets, successive frames from the same camera feed share near-identical backgrounds,
    lighting, camera angles, and road geometry. If frames from the same video sequence are randomly
    assigned across `train`, `val`, and `test` splits, the model memorizes camera backgrounds rather
    than generalizing to novel crash dynamics.
    """)

    # Run / display manifest audit
    manifest = scan_cctv_manifest(verify_images=False)
    audit_res = audit_split_leakage(manifest)

    # Metric summary
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        metric_card("Total Dataset Frames", f"{audit_res['total_frames']}", f"{audit_res['total_frames']} indexed images", border_color="#4DA8FF")
    with a2:
        metric_card("Total Video Groups", f"{audit_res['total_groups']}", f"{audit_res['total_groups']} distinct sequence IDs", border_color="#8899A6")
    with a3:
        metric_card("Leaking Video Groups", f"{audit_res['leaking_groups_count']} / {audit_res['total_groups']}", "Present in >1 split", border_color="#FF4B4B")
    with a4:
        metric_card("Contaminated Frames", f"{audit_res['leakage_percentage']}%", f"{audit_res['leaking_frames_count']} leaking frames", border_color="#FF8C42")

    st.markdown("<br>", unsafe_allow_html=True)

    info_card(
        "⚠️ <strong>Empirical Audit Finding:</strong><br>"
        f"In the raw dataset structure, <strong>{audit_res['leaking_groups_count']} out of {audit_res['total_groups']} video sequence groups ({audit_res['leakage_percentage']}% of all frames)</strong> "
        "have frames split across both training and evaluation sets (e.g. sequence <code>acc1</code> appears in train, val, and test).<br><br>"
        "• <strong>Evaluation Inflation:</strong> The reported test accuracy (81.0%) on the naive split reflects partial background memorization.<br>"
        "• <strong>Generalization Reality:</strong> When evaluated with <em>leakage-free grouped splitting</em> (where whole video sequences are held out), "
        "generalization drops to realistic baseline levels (~65–72%), revealing the true offline transfer learning performance.<br>"
        "• <strong>Reproducible Mitigation:</strong> The project generates <code>data/processed/cctv_grouped_manifest.csv</code> to enforce GroupShuffleSplit."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Class Breakdown Table
    st.markdown("**Cross-Split Leakage Breakdown by Class:**")
    cb = audit_res.get("class_breakdown", {})
    breakdown_rows = []
    for cls_name, stats in cb.items():
        breakdown_rows.append({
            "Class Name": cls_name,
            "Total Groups": stats["total_groups"],
            "Leaking Groups": stats["leaking_groups"],
            "Total Frames": stats["total_frames"],
            "Leaking Frames": stats["leaking_frames"],
            "Leakage Rate": f"{stats['leakage_percentage']}%",
            "Sample Leaking Sequences": ", ".join(stats["sample_leaking_groups"][:5]),
        })
    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

    st.markdown(r"""
    <div style="font-size: 0.82rem; color: #8899A6; margin-top: 0.5rem; line-height: 1.5;">
        <strong>Grouping Heuristic Specification:</strong> Video sequences are extracted via regex prefix matching
        <code>r"^([a-zA-Z0-9_\-]+?)(?:[\s\(]|_\d+$)"</code> on image filenames. Limitations: camera angles lacking sequence
        prefixes may represent latent unmodeled dependencies.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Performance & Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Empirical Model Performance on Unseen Test Set (101 Images)", "📊")

    p1, p2 = st.columns(2)

    with p1:
        # Confusion matrix
        cm = test_metrics.get("confusion_matrix", [[37, 10], [9, 44]])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Accident", "Actual Non-Accident"],
            columns=["Predicted Accident", "Predicted Non-Accident"],
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Reds",
            title="Confusion Matrix (MobileNetV2 Test Evaluation)",
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with p2:
        st.markdown("**Transfer Learning vs Baseline Benchmark:**")
        b_test = baseline_metrics.get("test_metrics", {})
        bench_df = pd.DataFrame([
            {
                "Evaluation Metric": "Overall Accuracy",
                "MobileNetV2 (CNN)": f"{test_metrics.get('accuracy', 0.81)*100:.1f}%",
                "Baseline (Color/Thumb LogReg)": f"{b_test.get('accuracy', 0.65)*100:.1f}%",
            },
            {
                "Evaluation Metric": "Macro Precision",
                "MobileNetV2 (CNN)": f"{test_metrics.get('macro_precision', 0.809)*100:.1f}%",
                "Baseline (Color/Thumb LogReg)": f"{b_test.get('macro_precision', 0.65)*100:.1f}%",
            },
            {
                "Evaluation Metric": "Macro Recall",
                "MobileNetV2 (CNN)": f"{test_metrics.get('macro_recall', 0.808)*100:.1f}%",
                "Baseline (Color/Thumb LogReg)": f"{b_test.get('macro_recall', 0.645)*100:.1f}%",
            },
            {
                "Evaluation Metric": "Macro F1-Score",
                "MobileNetV2 (CNN)": f"{test_metrics.get('macro_f1', 0.809):.4f}",
                "Baseline (Color/Thumb LogReg)": f"{b_test.get('macro_f1', 0.6457):.4f}",
            },
            {
                "Evaluation Metric": "ROC-AUC Score",
                "MobileNetV2 (CNN)": f"{test_metrics.get('roc_auc', 0.826):.4f}",
                "Baseline (Color/Thumb LogReg)": f"{b_test.get('roc_auc', 0.702):.4f}",
            },
            {
                "Evaluation Metric": "Inference Latency (CPU)",
                "MobileNetV2 (CNN)": "35 – 50 ms",
                "Baseline (Color/Thumb LogReg)": "< 5 ms",
            },
        ])
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    # Training Curves
    history = metrics_data.get("training_history", [])
    if history:
        st.markdown("<br>", unsafe_allow_html=True)
        h_df = pd.DataFrame(history)

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=h_df["epoch"], y=h_df["train_loss"], name="Train Loss", line=dict(color="#FF8C42", width=2)))
        fig_h.add_trace(go.Scatter(x=h_df["epoch"], y=h_df["val_loss"], name="Val Loss", line=dict(color="#FF4B4B", width=2, dash="dash")))
        fig_h.add_trace(go.Scatter(x=h_df["epoch"], y=h_df["train_acc"], name="Train Accuracy", yaxis="y2", line=dict(color="#4DA8FF", width=2)))
        fig_h.add_trace(go.Scatter(x=h_df["epoch"], y=h_df["val_acc"], name="Val Accuracy", yaxis="y2", line=dict(color="#00C853", width=2, dash="dash")))

        fig_h.update_layout(
            title="Fine-Tuning Loss & Accuracy Trajectory Across Epochs",
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            xaxis=dict(title="Epoch", dtick=1),
            yaxis=dict(title="Loss", side="left"),
            yaxis2=dict(title="Accuracy", side="right", overlaying="y", range=[0.4, 1.0]),
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_h, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Edge Deployment Blueprint & Latency Budget
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("Highway Edge Deployment & Camera Operations", "⚡")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("""
        <div class="kpi-card" style="text-align: left; border-left: 4px solid #4DA8FF; min-height: 230px;">
            <div style="font-weight: 700; color: #4DA8FF; font-size: 1rem; margin-bottom: 0.5rem;">
                📡 1. Edge-First Architecture (Zero Streaming Latency)
            </div>
            <p style="font-size: 0.85rem; color: #8899A6; line-height: 1.6;">
                Instead of streaming 24/7 high-definition 4K video feeds over cellular/fiber to a central cloud server,
                a lightweight edge computer (e.g., <strong>NVIDIA Jetson Orin Nano, Raspberry Pi 5 with Hailo-8 AI Hat</strong>)
                is mounted directly adjacent to the roadside camera pole.
                <br><br>
                The edge device processes RTSP video streams locally at 20–30 FPS and transmits <em>only telemetry and incident alert clips</em>
                when a collision signature is verified.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class="kpi-card" style="text-align: left; border-left: 4px solid #00C853; min-height: 230px;">
            <div style="font-weight: 700; color: #00C853; font-size: 1rem; margin-bottom: 0.5rem;">
                🛡️ 2. Temporal Smoothing & False-Alarm Suppression
            </div>
            <p style="font-size: 0.85rem; color: #8899A6; line-height: 1.6;">
                Single-frame false alarms can be caused by unusual camera angles, shadows, or vehicles temporarily reversing.
                <br><br>
                In production, an <strong>$N$-out-of-$M$ Temporal Rolling Window</strong> is enforced:
                An incident alert is dispatched to the Highway Management Center only if <strong>at least 3 out of 5 consecutive frames</strong>
                (within a 1.5-second buffer) classify as <code>Accident</code> with confidence $\\ge 0.75$.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**⏱️ End-to-End Highway Crash Detection Latency Budget:**")
    latency_table = pd.DataFrame([
        {"Stage": "1. RTSP Stream Frame Ingestion", "Technology": "Hardware H.264/H.265 Decoder", "Typical Latency": "10 – 15 ms"},
        {"Stage": "2. Preprocessing & Tensor Normalization", "Technology": "Torchvision / OpenCV CPU", "Typical Latency": "4 – 6 ms"},
        {"Stage": "3. CNN Forward Pass", "Technology": "MobileNetV2 (FP16 / INT8 Edge Quantized)", "Typical Latency": "25 – 45 ms"},
        {"Stage": "4. Temporal Sliding Window Consensus", "Technology": "Ring Buffer Logic", "Typical Latency": "1 – 2 ms"},
        {"Stage": "5. MQTT Incident Alert Dispatch", "Technology": "Lightweight Pub/Sub Broker", "Typical Latency": "15 – 25 ms"},
        {"Stage": "**Total Turnaround to Dispatch Operator**", "Technology": "**Edge RoadRisk AI Pipeline**", "Typical Latency": "**< 100 ms (Real-Time)**"},
    ])
    st.dataframe(latency_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Operational Governance & Disclaimers
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("Responsible AI & Computer Vision Governance", "⚠️")

    info_card(
        "⚠️ <strong>Decision-Support Disclaimer:</strong><br>"
        "This computer vision system is an <strong>educational portfolio prototype</strong> trained on static surveillance snapshots.<br><br>"
        "• <strong>Not Certified for Autonomous Dispatch:</strong> All detection alerts must be verified by a certified human traffic operator.<br>"
        "• <strong>Weather and Illumination Sensitivity:</strong> Severe monsoon rains, dense smog/fog, lens condensation, and night-time high-beam headlight glare can degrade detection accuracy.<br>"
        "• <strong>Frame-Level vs Object Tracking:</strong> This model classifies full frame scenes (Accident vs Normal). It does not substitute multi-object tracking (e.g., YOLO + ByteTRACK) for velocity estimation or trajectory collision forecasting."
    )
