"""
RoadRisk AI — Utility Functions
================================
Shared helpers used across the application: caching, formatting, UI widgets.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from src.config import COLORS, APP_TITLE


# ── CSS Injection ────────────────────────────────────────────────────────────

def inject_custom_css() -> None:
    """Inject the global custom CSS for the RoadRisk AI dark theme."""
    st.markdown("""
    <style>
    /* ── Import Google Font ───────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── KPI Card ──────────────────────────────────────────────── */
    .kpi-card {
        background: linear-gradient(135deg, #1A1F2C 0%, #252B3B 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        background: linear-gradient(135deg, #FF4B4B, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8899A6;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .kpi-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }

    /* ── Section Header ────────────────────────────────────────── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #FF4B4B;
        display: inline-block;
    }

    /* ── Info Card ─────────────────────────────────────────────── */
    .info-card {
        background: linear-gradient(135deg, #1A1F2C 0%, #252B3B 100%);
        border-left: 4px solid #4DA8FF;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #FAFAFA;
    }

    /* ── Sidebar Branding ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2C 100%);
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0 0.5rem;
        background: linear-gradient(135deg, #FF4B4B, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-subtitle {
        font-size: 0.75rem;
        color: #8899A6;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* ── Status Badge ─────────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(0,200,83,0.15);
        color: #00C853;
        border: 1px solid rgba(0,200,83,0.3);
    }
    .status-pending {
        background: rgba(255,214,0,0.15);
        color: #FFD600;
        border: 1px solid rgba(255,214,0,0.3);
    }
    .status-error {
        background: rgba(255,23,68,0.15);
        color: #FF1744;
        border: 1px solid rgba(255,23,68,0.3);
    }

    /* ── Hide Streamlit Defaults ──────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ── KPI Card Widget ──────────────────────────────────────────────────────────

def kpi_card(icon: str, label: str, value: str | int | float) -> str:
    """Return HTML for a single KPI card."""
    if isinstance(value, (int, float)):
        formatted = f"{value:,.0f}" if isinstance(value, (int, float)) and value > 999 else str(value)
    else:
        formatted = str(value)
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{formatted}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def render_kpi_row(kpis: list[tuple[str, str, str | int | float]]) -> None:
    """
    Render a row of KPI cards.
    kpis: list of (icon, label, value) tuples.
    """
    cols = st.columns(len(kpis))
    for col, (icon, label, value) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(icon, label, value), unsafe_allow_html=True)


# ── Section Header ───────────────────────────────────────────────────────────

def section_header(title: str, icon: str = "") -> None:
    """Render a styled section header."""
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


# ── Info Box ─────────────────────────────────────────────────────────────────

def info_card(text: str) -> None:
    """Render a styled information card."""
    st.markdown(
        f'<div class="info-card">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Status Badge ─────────────────────────────────────────────────────────────

def status_badge(label: str, status: str = "ready") -> str:
    """Return HTML for a status badge. status: 'ready' | 'pending' | 'error'."""
    return f'<span class="status-badge status-{status}">{label}</span>'


# ── Sidebar Setup ────────────────────────────────────────────────────────────

def setup_sidebar() -> None:
    """Render the branded sidebar header."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">🚦 RoadRisk AI</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-subtitle">Road Accident Analysis & Intelligence</div>',
            unsafe_allow_html=True,
        )
        st.divider()


# ── Data Loading Helper ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_csv_cached(path: str | Path, **kwargs) -> pd.DataFrame | None:
    """Load a CSV with Streamlit caching. Returns None if the file is missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, **kwargs)
    except Exception as e:
        st.error(f"Failed to load {p.name}: {e}")
        return None


# ── Formatting Helpers ───────────────────────────────────────────────────────

def format_number(n: int | float) -> str:
    """Human-readable number formatting (e.g., 1,234,567)."""
    if pd.isna(n):
        return "N/A"
    return f"{n:,.0f}"


def format_percentage(n: float, decimals: int = 1) -> str:
    """Format a fraction (0–1) as a percentage string."""
    if pd.isna(n):
        return "N/A"
    return f"{n * 100:.{decimals}f}%"
