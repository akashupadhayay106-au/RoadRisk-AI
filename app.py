"""
🚦 RoadRisk AI — Main Application Entry Point
================================================
AI-Powered Road Accident Analysis & Risk Intelligence Platform

Run locally:
    streamlit run app.py

This file sets up the Streamlit multipage navigation and renders the
Home / Overview dashboard with live analytical KPIs, honest dataset
availability status, and the platform module grid.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    APP_VERSION,
    CCTV_TRAIN_DIR,
    DATASETS,
    PAGE_CONFIG,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    ensure_directories,
)
from src.database import DATABASE_PATH, load_data_to_database
from src.sql_queries import get_overall_kpis
from src.utils import (
    empty_state,
    hero_kicker,
    info_card,
    inject_custom_css,
    page_header,
    render_status_strip,
    section_header,
    setup_sidebar,
    skeleton,
    status_badge,
    status_chip,
)

# ── Page Config (must be the first Streamlit call) ───────────────────────────
st.set_page_config(**PAGE_CONFIG)

# ── Ensure directories exist ────────────────────────────────────────────────
ensure_directories()

# ── Global CSS + ambient layers ──────────────────────────────────────────────
inject_custom_css()

# ── Sidebar ──────────────────────────────────────────────────────────────────
setup_sidebar()


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA-SOURCE AVAILABILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_file(filename: str) -> Path | None:
    """Search for a file in raw/ or project root."""
    for candidate in (RAW_DATA_DIR / filename, PROJECT_ROOT / filename):
        if candidate.exists():
            return candidate
    return None


def _csv_status(path: Path | None, label: str) -> dict:
    """Get status info for a CSV file (rows/columns counted cheaply)."""
    if path and path.exists():
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                nrows = sum(1 for _ in fh) - 1
            ncols = len(pd.read_csv(path, nrows=0).columns)
            return {
                "label": label,
                "exists": True,
                "detail": f"{max(nrows, 0):,} rows × {ncols} cols",
            }
        except Exception:
            return {"label": label, "exists": True, "detail": "unreadable file"}
    return {"label": label, "exists": False, "detail": "Not found — add to data/raw/"}


def _txt_status(path: Path | None, label: str) -> dict:
    """Get status info for a text-based dataset (e.g. UrbanGB)."""
    if path and path.exists():
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                nrows = sum(1 for _ in fh)
            return {"label": label, "exists": True, "detail": f"{nrows:,} rows"}
        except Exception:
            return {"label": label, "exists": True, "detail": "unreadable file"}
    return {"label": label, "exists": False, "detail": "Not found — add to data/raw/"}


def check_dataset_status() -> dict[str, dict]:
    """
    Check which source datasets are available on disk.
    Returns a dict with dataset key → {label, exists, detail}.
    """
    return {
        "indian_accidents": _csv_status(_find_file("indian_roads_dataset.csv"), "Indian Accidents (2022–2025)"),
        "severity_train": _csv_status(_find_file("train.csv"), "Severity Train Set"),
        "severity_test": _csv_status(_find_file("test.csv"), "Severity Test Set"),
        "urban_gb": _txt_status(_find_file("urbanGB.txt"), "UrbanGB Coordinates (UK)"),
        "opencity": _csv_status(_find_file("Accdataset_hk_PS_BAEL_Combined.csv"), "OpenCity India 2023"),
    }


def _cctv_status() -> dict:
    """Report CCTV image split availability (images are intentionally not shipped)."""
    jpg_any = any(CCTV_TRAIN_DIR.rglob("*.jpg")) or any(CCTV_TRAIN_DIR.rglob("*.png"))
    if CCTV_TRAIN_DIR.exists() and jpg_any:
        count = sum(1 for _ in CCTV_TRAIN_DIR.rglob("*") if _.is_file())
        return {"label": "CCTV Accident Images", "exists": True, "detail": f"{count:,} train images"}
    return {"label": "CCTV Accident Images", "exists": False, "detail": "Not found — model still works via upload"}


def _load_live_kpis() -> dict:
    """Return real aggregated KPIs from the SQLite analytics database ({} when empty)."""
    try:
        if not DATABASE_PATH.exists():
            load_data_to_database()
        return get_overall_kpis() or {}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE — Overview Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def render_home() -> None:
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(hero_kicker("AI-Powered Road Safety Intelligence Platform"), unsafe_allow_html=True)
    page_header(
        "RoadRisk AI",
        section="Overview",
        subtitle=(
            "Multi-source accident analytics, ML severity classification, geospatial hotspot discovery, "
            "weather & CCTV intelligence, explainable AI, and a grounded GenAI assistant."
        ),
        icon="🚦",
        accent="cyan",
    )

    kpis = _load_live_kpis()
    total = int(kpis.get("total_accidents", 0) or 0)

    if total:
        render_status_strip([
            status_chip(f"LIVE ANALYTICS · {total:,} incidents indexed", "ok"),
            status_chip(f"v{APP_VERSION}", "info"),
        ])
    else:
        render_status_strip([
            status_chip("DEMO / OFFLINE MODE — analytics database not loaded", "warn"),
            status_chip(f"v{APP_VERSION}", "info"),
        ])

    # ── KPI Cards (backed by real data when present) ─────────────────────────
    section_header("Executive Risk Snapshot", "📊")
    if total:
        fatal = int(kpis.get("fatal_accidents", 0) or 0)
        casualties = int(kpis.get("total_casualties", 0) or 0)
        avg_risk = float(kpis.get("avg_risk_score", 0.0) or 0.0)
        fatal_rate = float(kpis.get("fatality_rate_pct", 0.0) or 0.0)

        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""
            <div class="kpi-card" title="Total reported incidents in the indexed SQLite database">
                <div class="kpi-icon">🚗</div>
                <div class="kpi-value">{total:,}</div>
                <div class="kpi-label">Total Accidents</div>
            </div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="kpi-card" title="Total casualties reported across all incidents">
                <div class="kpi-icon">🚑</div>
                <div class="kpi-value">{casualties:,}</div>
                <div class="kpi-label">Casualties</div>
            </div>""", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
            <div class="kpi-card" title="Fatal incidents (share of all incidents)">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-value" style="background:linear-gradient(90deg,#FDA4AF,#FB7185);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{fatal:,}</div>
                <div class="kpi-label">Fatal ({fatal_rate:.1f}%)</div>
            </div>""", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""
            <div class="kpi-card" title="Mean risk-score index of the indexed incidents (0–1 scale)">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-value">{avg_risk:.2f}</div>
                <div class="kpi-label">Avg Risk Index</div>
            </div>""", unsafe_allow_html=True)
        st.caption(
            "KPI cards are computed live from the local analytics database. "
            "Metrics reflect reported incidents, not exposure-adjusted risk rates."
        )
    else:
        skeleton(1)
        empty_state(
            "Analytics database not loaded on this instance",
            "The public deployment intentionally ships without raw datasets (licence terms). "
            "Add the datasets to `data/raw/` and rebuild the database to see live KPIs here.",
            hint="The severity model, CCTV model and grounded assistant remain fully functional in demo mode.",
            icon="📊",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Intelligence Module Grid ─────────────────────────────────────────────
    section_header("Intelligence Modules", "🧩")

    modules = [
        ("📊", "Dashboard", "Executive analytics, trends and KPI filters", "1_Dashboard", "ready"),
        ("🔍", "Data Explorer", "Faceted search, dossiers & schema audits", "2_Data_Explorer", "ready"),
        ("🗄️", "SQL Analytics", "Parameterized read-only query studio", "3_SQL_Analytics", "ready"),
        ("🎯", "Severity Prediction", "Leakage-free ML severity classifier", "4_Severity_Prediction", "ready"),
        ("🗺️", "Hotspot Map", "DBSCAN geospatial clustering & routing", "5_Hotspot_Map", "ready"),
        ("🌦️", "Weather Analysis", "Open-Meteo atmospheric risk intelligence", "6_Weather_Analysis", "ready"),
        ("📹", "CCTV Detection", "MobileNetV2 crash classifier & Grad-CAM", "7_CCTV_Detection", "ready"),
        ("🤖", "AI Assistant", "Grounded GenAI + offline RAG engine", "8_AI_Assistant", "ready"),
        ("📈", "Explainability", "Feature importance & causal governance", "9_Model_Explainability", "ready"),
        ("ℹ️", "About", "Architecture, data cards & limitations", "10_About", "ready"),
    ]

    for row_start in range(0, len(modules), 5):
        row = modules[row_start:row_start + 5]
        cols = st.columns(len(row))
        for col, (icon, name, desc, page, mod_status) in zip(cols, row):
            badge = status_badge("Ready", "ready") if mod_status == "ready" else status_badge("Coming Soon", "pending")
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="min-height: 168px; text-align: left;">
                    <div style="font-size: 1.8rem; margin-bottom: 0.4rem;">{icon}</div>
                    <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; color: #E6EDF7;">{name}</div>
                    <div style="font-size: 0.78rem; color: #8CA3C3; margin-bottom: 0.6rem;">{desc}</div>
                    {badge}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset Status (honest, data-less safe) ──────────────────────────────
    section_header("Dataset Status", "📂")

    dataset_status = check_dataset_status()
    dataset_status["cctv"] = _cctv_status()

    cols = st.columns(3)
    for i, (key, info) in enumerate(dataset_status.items()):
        with cols[i % 3]:
            badge_html = status_badge("✓ Available", "ready") if info["exists"] else status_badge("✗ Missing", "error")
            st.markdown(f"""
            <div class="info-card" style="border-left-color: {'#34D399' if info['exists'] else '#FB7185'};">
                <div style="font-weight: 600; margin-bottom: 0.3rem; color:#E6EDF7;">{info['label']}</div>
                <div style="font-size: 0.8rem; color: #8CA3C3; margin-bottom: 0.4rem;">{info['detail']}</div>
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

    # ── Responsible Use Disclaimer ───────────────────────────────────────────
    st.divider()
    info_card(
        "<strong style='color:#FB7185;'>⚠️ Responsible Use Disclaimer</strong><br>"
        "RoadRisk AI is an <strong>educational portfolio project</strong> for exploring road accident data "
        "and demonstrating data science techniques. It is <strong>not</strong> a certified road-safety "
        "system and should not be used for emergency response, policy decisions, or real-time accident "
        "detection without proper validation, expert review, and regulatory approval.<br><br>"
        "Accident counts reflect reported data, not true risk rates. Spatial clusters show observed "
        "concentrations, not exposure-adjusted danger. ML predictions are decision-support demonstrations, "
        "not ground truth."
    )


# ── Run ──────────────────────────────────────────────────────────────────────
render_home()