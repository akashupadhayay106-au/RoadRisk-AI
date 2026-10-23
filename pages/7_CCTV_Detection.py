"""
📹 CCTV Detection — Computer Vision Accident Classification
=============================================================
Module H: Image-based accident/non-accident classification.
Will be fully implemented in Phase 8.
"""

import streamlit as st
from src.config import PAGE_CONFIG, CCTV_TRAIN_DIR, CCTV_TEST_DIR, CCTV_VAL_DIR
from src.utils import inject_custom_css, setup_sidebar, section_header, info_card, render_kpi_row

st.set_page_config(**PAGE_CONFIG)
inject_custom_css()
setup_sidebar()

st.markdown("""
<h1 style="
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #EC4899, #FF4B4B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">📹 CCTV Accident Detection</h1>
<p style="color: #8899A6; margin-top: -0.5rem;">
    Classify CCTV frames as Accident or Non-Accident using computer vision.
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Show CCTV Dataset Status ────────────────────────────────────────────────
section_header("Dataset Status", "📂")

def count_images(directory):
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

train_acc = count_images(CCTV_TRAIN_DIR / "Accident")
train_non = count_images(CCTV_TRAIN_DIR / "Non Accident")
test_acc = count_images(CCTV_TEST_DIR / "Accident")
test_non = count_images(CCTV_TEST_DIR / "Non Accident")
val_acc = count_images(CCTV_VAL_DIR / "Accident")
val_non = count_images(CCTV_VAL_DIR / "Non Accident")

total = train_acc + train_non + test_acc + test_non + val_acc + val_non

if total > 0:
    render_kpi_row([
        ("📁", "Total Images", total),
        ("🚗", "Train (Acc/Non)", f"{train_acc}/{train_non}"),
        ("🧪", "Test (Acc/Non)", f"{test_acc}/{test_non}"),
        ("✅", "Val (Acc/Non)", f"{val_acc}/{val_non}"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.warning(
        "⚠️ CCTV dataset not found. Please download from "
        "[Kaggle](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage) "
        "and extract the train/test/val folders into `data/`."
    )

info_card(
    "🚧 <strong>Coming in Phase 8</strong><br>"
    "This module will include:<br>"
    "• Image preprocessing and augmentation<br>"
    "• Transfer learning with pretrained CNN (e.g., ResNet/EfficientNet)<br>"
    "• Training with proper train/val/test split<br>"
    "• Model evaluation with confusion matrix<br>"
    "• Streamlit image upload for inference<br>"
    "• Frame-level predictions with confidence scores"
)

section_header("Important Limitations", "⚠️")
st.markdown("""
> **This is offline image classification, not real-time detection.**
> The model classifies individual frames as "Accident" or "Non-Accident."
> It does not perform object detection, tracking, or real-time video analysis.
> Confidence scores should be interpreted carefully — uncertain predictions
> will be clearly flagged.
""")
