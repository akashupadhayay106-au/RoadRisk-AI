"""
🚦 RoadRisk AI — Main Application Entry Point
================================================
AI-Powered Road Accident Analysis & Risk Intelligence Platform

Run locally:
    streamlit run app.py

This file sets up the Streamlit multipage navigation and renders
the Home / Overview Dashboard (Module A).
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import (
    PAGE_CONFIG,
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    COLORS,
    DATASETS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    INDIAN_ACCIDENTS_FILE,
    SEVERITY_TRAIN_FILE,
    URBAN_GB_COORDS_FILE,
    OPENCITY_FILE,
    CCTV_TRAIN_DIR,
    ensure_directories,
)
from src.utils import (
    inject_custom_css,
    kpi_card,
    render_kpi_row,
    section_header,
    info_card,
    status_badge,
    setup_sidebar,
)

# ── Page Config (must be the first Streamlit call) ───────────────────────────
st.set_page_config(**PAGE_CONFIG)

# ── Ensure directories exist ────────────────────────────────────────────────
ensure_directories()

# ── Global CSS ──────────────────────────────────────────────────────────────
inject_custom_css()

# ── Sidebar ─────────────────────────────────────────────────────────────────
setup_sidebar()


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE — Overview Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def check_dataset_status() -> dict[str, dict]:
    """
    Check which datasets are available on disk.
    Returns a dict with dataset key → {exists, path, rows (if CSV)}.
    """
    status = {}

    # Indian Accidents
    indian_path = _find_file("indian_roads_dataset.csv")
    status["indian_accidents"] = _csv_status(indian_path, "Indian Accidents (2022–2025)")

    # Severity Train
    sev_path = _find_file("train.csv")
    status["severity_train"] = _csv_status(sev_path, "Severity Train Set")

    # Severity Test
    sev_test = _find_file("test.csv")
    status["severity_test"] = _csv_status(sev_test, "Severity Test Set")

    # UrbanGB
    ugb_path = _find_file("urbanGB.txt")
    status["urban_gb"] = _txt_status(ugb_path, "UrbanGB Coordinates")

    # OpenCity
    oc_path = _find_file("Accdataset_hk_PS_BAEL_Combined.csv")
    status["opencity"] = _csv_status(oc_path, "OpenCity India 2023")

    # CCTV Images
    cctv_exists = CCTV_TRAIN_DIR.exists() and any(CCTV_TRAIN_DIR.rglob("*.jpg")) or any(CCTV_TRAIN_DIR.rglob("*.png"))
    cctv_count = 0
    if cctv_exists:
        cctv_count = sum(1 for _ in CCTV_TRAIN_DIR.rglob("*") if _.is_file())
    status["cctv"] = {
        "label": "CCTV Accident Images",
        "exists": cctv_exists,
        "detail": f"{cctv_count} train images" if cctv_exists else "Not found",
    }

    return status


def _find_file(filename: str) -> Path | None:
    """Search for a file in raw/ or project root."""
    raw = RAW_DATA_DIR / filename
    if raw.exists():
        return raw
    root = PROJECT_ROOT / filename
    if root.exists():
        return root
    return None


def _csv_status(path: Path | None, label: str) -> dict:
    """Get status info for a CSV file."""
    if path and path.exists():
        try:
            # Read just the header + 1 row to count columns; count rows cheaply
            nrows = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
            ncols = len(pd.read_csv(path, nrows=0).columns)
            return {
                "label": label,
                "exists": True,
                "detail": f"{nrows:,} rows × {ncols} cols",
                "path": str(path),
            }
        except Exception as e:
            return {"label": label, "exists": True, "detail": f"Error reading: {e}"}
    return {"label": label, "exists": False, "detail": "Not found"}


def _txt_status(path: Path | None, label: str) -> dict:
    """Get status info for a text file."""
    if path and path.exists():
        nrows = sum(1 for _ in open(path, encoding="utf-8", errors="replace"))
        return {
            "label": label,
            "exists": True,
            "detail": f"{nrows:,} rows",
        }
    return {"label": label, "exists": False, "detail": "Not found"}


def render_home():
    """Render the home page / overview dashboard."""

    # ── Hero Section ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0 1rem;">
        <h1 style="
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #FF4B4B 0%, #FF8C42 50%, #FFD600 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        ">🚦 RoadRisk AI</h1>
        <p style="
            font-size: 1.15rem;
            color: #8899A6;
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        ">{APP_SUBTITLE}</p>
        <p style="
            font-size: 0.8rem;
            color: #556677;
            margin-top: 0.5rem;
        ">v{APP_VERSION} • Built with Streamlit, Scikit-learn, PyTorch & a pluggable LLM provider layer</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Module Navigation Cards ──────────────────────────────────────────────
    section_header("Platform Modules", "🧩")

    modules = [
        ("📊", "Dashboard",           "Interactive analytics & KPIs",         "1_Dashboard",           "ready"),
        ("🔍", "Data Explorer",       "Inspect, filter & download data",      "2_Data_Explorer",       "ready"),
        ("🗄️", "SQL Analytics",       "Query accident database with SQL",     "3_SQL_Analytics",       "ready"),
        ("🎯", "Severity Prediction", "ML classification of accident severity","4_Severity_Prediction","ready"),
        ("🗺️", "Hotspot Map",        "Geospatial clustering, heatmaps & routing","5_Hotspot_Map",      "ready"),
        ("🌦️", "Weather Analysis",   "Historical weather & accident links",  "6_Weather_Analysis",    "ready"),
        ("📹", "CCTV Detection",      "Computer vision accident detection",   "7_CCTV_Detection",      "ready"),
        ("🤖", "AI Assistant",        "Grounded road safety Q&A (live or offline)","8_AI_Assistant",   "ready"),
        ("📈", "Explainability",      "SHAP, feature importance & monitoring","9_Model_Explainability","ready"),
        ("ℹ️", "About",              "Project info, sources & methodology",  "10_About",              "ready"),
    ]

    # Render in rows of 5
    for row_start in range(0, len(modules), 5):
        row = modules[row_start:row_start + 5]
        cols = st.columns(len(row))
        for col, (icon, name, desc, page, mod_status) in zip(cols, row):
            badge = status_badge("Ready", "ready") if mod_status == "ready" else status_badge("Coming Soon", "pending")
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="min-height: 160px; text-align: left;">
                    <div style="font-size: 1.8rem; margin-bottom: 0.4rem;">{icon}</div>
                    <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem;">{name}</div>
                    <div style="font-size: 0.78rem; color: #8899A6; margin-bottom: 0.6rem;">{desc}</div>
                    {badge}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset Status ───────────────────────────────────────────────────────
    section_header("Dataset Status", "📂")

    dataset_status = check_dataset_status()

    cols = st.columns(3)
    for i, (key, info) in enumerate(dataset_status.items()):
        with cols[i % 3]:
            badge_html = status_badge("✓ Available", "ready") if info["exists"] else status_badge("✗ Missing", "error")
            st.markdown(f"""
            <div class="info-card" style="border-left-color: {'#00C853' if info['exists'] else '#FF1744'};">
                <div style="font-weight: 600; margin-bottom: 0.3rem;">{info['label']}</div>
                <div style="font-size: 0.8rem; color: #8899A6; margin-bottom: 0.4rem;">{info['detail']}</div>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data Sources ─────────────────────────────────────────────────────────
    section_header("Data Sources & Attribution", "📜")

    for key, meta in DATASETS.items():
        with st.expander(f"**{meta['name']}** — {meta['source']}"):
            st.markdown(f"""
            - **Source:** [{meta['source']}]({meta['url']})
            - **License:** {meta['license']}
            - **Description:** {meta['description']}
            """)

    # ── Disclaimers ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="
        background: rgba(255,75,75,0.08);
        border: 1px solid rgba(255,75,75,0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.8rem;
        color: #8899A6;
        line-height: 1.6;
    ">
        <strong style="color: #FF4B4B;">⚠️ Responsible Use Disclaimer</strong><br>
        RoadRisk AI is an <strong>educational portfolio project</strong> for exploring road accident data
        and demonstrating data science techniques. It is <strong>not</strong> a certified road-safety
        system and should not be used for emergency response, policy decisions, or real-time accident
        detection without proper validation, expert review, and regulatory approval.<br><br>
        Accident counts reflect reported data, not true risk rates. Spatial clusters show observed
        concentrations, not exposure-adjusted danger. ML predictions are decision-support demonstrations,
        not ground truth.
    </div>
    """, unsafe_allow_html=True)


# ── Run ──────────────────────────────────────────────────────────────────────
render_home()
