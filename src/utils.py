"""
RoadRisk AI — Utility Functions
================================
Shared helpers used across the application: caching, formatting, UI widgets.
"""

import streamlit as st
import pandas as pd
from pathlib import Path


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

    /* ── Empty / Error / Loading States ───────────────────────── */
    .state-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.35rem;
        padding: 2rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 14px;
        background: linear-gradient(135deg, #1A1F2C 0%, #252B3B 100%);
        border: 1px dashed rgba(255,255,255,0.12);
    }
    .state-box.state-error {
        border: 1px solid rgba(255,23,68,0.45);
        border-left: 4px solid #FF4B4B;
    }
    .state-box.state-empty {
        border: 1px solid rgba(77,168,255,0.30);
        border-left: 4px solid #4DA8FF;
    }
    .state-icon { font-size: 1.8rem; line-height: 1; }
    .state-title { font-size: 1rem; font-weight: 700; color: #FAFAFA; }
    .state-msg { font-size: 0.88rem; color: #B8C4D0; max-width: 60ch; line-height: 1.5; }
    .state-hint {
        font-size: 0.78rem;
        color: #8899A6;
        margin-top: 0.35rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 0.5rem;
    }

    /* ── Service / API Status Pills ───────────────────────────── */
    .svc-row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 0.35rem 0 0.75rem 0; }
    .svc-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .svc-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: currentColor; flex: none; }
    .svc-ok   { background: rgba(0,200,83,0.15);  color: #4ADE80; border-color: rgba(0,200,83,0.35); }
    .svc-off  { background: rgba(255,214,0,0.12); color: #FFD600; border-color: rgba(255,214,0,0.30); }
    .svc-bad  { background: rgba(255,23,68,0.15); color: #FF6B81; border-color: rgba(255,23,68,0.35); }

    /* ── Accessibility: visible keyboard focus ────────────────── */
    a:focus-visible,
    button:focus-visible,
    [role="button"]:focus-visible,
    input:focus-visible,
    select:focus-visible,
    textarea:focus-visible,
    summary:focus-visible {
        outline: 2px solid #4DA8FF;
        outline-offset: 2px;
        border-radius: 6px;
    }

    /* ── Responsive / Mobile Coherence ────────────────────────── */
    @media (max-width: 768px) {
        .kpi-card { padding: 1rem 0.75rem; border-radius: 12px; }
        .kpi-value { font-size: 1.55rem; }
        .kpi-label { font-size: 0.72rem; letter-spacing: 0.05em; }
        .kpi-icon { font-size: 1.3rem; }
        .section-header { font-size: 1.05rem; margin: 1.1rem 0 0.6rem 0; }
        .state-box { padding: 1.35rem 1rem; }
        .state-msg { font-size: 0.83rem; }
        .info-card { padding: 0.85rem 0.95rem; font-size: 0.9rem; }
        .sidebar-title { font-size: 1.2rem; }
    }

    /* ── Respect reduced-motion preferences ───────────────────── */
    @media (prefers-reduced-motion: reduce) {
        .kpi-card { transition: none; }
        .kpi-card:hover { transform: none; }
    }

    /* ── Hide Streamlit Defaults ──────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ── KPI Card Widget ──────────────────────────────────────────────────────────

def kpi_card(icon: str, label: str, value: str | int | float, help_text: str | None = None) -> str:
    """Return HTML for a single KPI card. `help_text` renders as an accessible tooltip."""
    if isinstance(value, (int, float)):
        formatted = f"{value:,.0f}" if isinstance(value, (int, float)) and value > 999 else str(value)
    else:
        formatted = str(value)
    tooltip = f' title="{help_text}"' if help_text else ""
    return f"""
    <div class="kpi-card"{tooltip}>
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


# ── Empty / Error / Loading States ───────────────────────────────────────────

def empty_state(title: str, message: str = "", hint: str = "", icon: str = "🔍") -> None:
    """Render a consistent empty-state block (e.g. no rows match the filters)."""
    hint_html = f'<div class="state-hint">{hint}</div>' if hint else ""
    msg_html = f'<div class="state-msg">{message}</div>' if message else ""
    st.markdown(
        f'<div class="state-box state-empty">'
        f'<div class="state-icon">{icon}</div>'
        f'<div class="state-title">{title}</div>'
        f'{msg_html}{hint_html}'
        f"</div>",
        unsafe_allow_html=True,
    )


def error_state(title: str, message: str = "", hint: str = "", icon: str = "⚠️") -> None:
    """Render a consistent error-state block for recoverable failures."""
    hint_html = f'<div class="state-hint">{hint}</div>' if hint else ""
    msg_html = f'<div class="state-msg">{message}</div>' if message else ""
    st.markdown(
        f'<div class="state-box state-error">'
        f'<div class="state-icon">{icon}</div>'
        f'<div class="state-title">{title}</div>'
        f'{msg_html}{hint_html}'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Service / API Status Indicator ───────────────────────────────────────────

def service_status_row(services: list[dict]) -> None:
    """
    Render a row of secret-free service status pills.

    Each entry must be a dict with:
      - label (str): display name, e.g. "Weather (Open-Meteo)"
      - configured (bool): whether any credentials/endpoint are set up
      - available (bool | None): None = not probed, True/False = probe result
      - detail (str, optional): short non-sensitive explanation, shown as tooltip

    Never accepts or renders credential values.
    """
    pills = []
    for svc in services:
        configured = bool(svc.get("configured"))
        available = svc.get("available")
        if not configured:
            state_class, state_word = "svc-off", "not configured"
        elif available is True:
            state_class, state_word = "svc-ok", "available"
        elif available is False:
            state_class, state_word = "svc-bad", "unavailable"
        else:
            state_class, state_word = "svc-off", "not tested"

        detail = svc.get("detail") or ""
        tooltip = f' title="{detail}"' if detail else ""
        label = svc.get("label", "Service")
        pills.append(
            f'<span class="svc-pill {state_class}"{tooltip}>'
            f'<span class="svc-dot"></span>{label}: {state_word}</span>'
        )

    st.markdown(f'<div class="svc-row">{"".join(pills)}</div>', unsafe_allow_html=True)


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
